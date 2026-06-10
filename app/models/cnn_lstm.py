import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.dropout = nn.Dropout2d(0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        return x


class CNNLSTMModel(nn.Module):
    def __init__(
        self,
        time_steps: int = 30,
        grid_height: int = 16,
        grid_width: int = 24,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.time_steps = time_steps
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers

        self.cnn = CNNFeatureExtractor(in_channels=1)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, grid_height, grid_width)
            cnn_out_dim = self.cnn(dummy).shape[1]
            self._conv3_out_h = dummy.shape[2] // 8
            self._conv3_out_w = dummy.shape[3] // 8
        self.cnn_out_dim = cnn_out_dim

        self.lstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        self._cached_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._gradcam_hooks: list = []
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._gradcam_enabled: bool = False

    def _forward_cnn_per_step(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, H, W = x.shape
        x = x.unsqueeze(2)
        x = x.view(batch_size * time_steps, 1, H, W)
        cnn_features = self.cnn(x)
        cnn_features = cnn_features.view(batch_size, time_steps, -1)
        return cnn_features

    def _build_zero_hidden(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h0 = torch.zeros(
            self.lstm_num_layers, batch_size, self.lstm_hidden_size, device=device
        )
        c0 = torch.zeros(
            self.lstm_num_layers, batch_size, self.lstm_hidden_size, device=device
        )
        return h0, c0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        device = x.device
        cnn_features = self._forward_cnn_per_step(x)

        h0, c0 = self._build_zero_hidden(batch_size, device)
        lstm_out, _ = self.lstm(cnn_features, (h0, c0))
        last_hidden = lstm_out[:, -1, :]
        logits = self.classifier(last_hidden)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        return probs.detach()

    @torch.no_grad()
    def predict_breakout(
        self,
        x: torch.Tensor,
        clear_intermediates: bool = True,
    ) -> float:
        probs = self.predict_proba(x)
        probs_cpu = probs.detach().cpu()
        prob_np = probs_cpu.numpy()
        prob_scalar = float(prob_np.reshape(-1)[0])

        if clear_intermediates:
            del probs, probs_cpu, prob_np
            if x.device.type == "cuda":
                with torch.cuda.device(x.device):
                    torch.cuda.empty_cache()
            gc.collect()

        return prob_scalar

    def _save_activation_hook(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient_hook(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def enable_gradcam(self) -> None:
        if self._gradcam_enabled:
            return
        target_layer = self.cnn.conv3
        fw_hook = target_layer.register_forward_hook(self._save_activation_hook)
        bw_hook = target_layer.register_full_backward_hook(self._save_gradient_hook)
        self._gradcam_hooks = [fw_hook, bw_hook]
        self._gradcam_enabled = True

    def disable_gradcam(self) -> None:
        for hook in self._gradcam_hooks:
            hook.remove()
        self._gradcam_hooks.clear()
        self._activations = None
        self._gradients = None
        self._gradcam_enabled = False

    def _run_with_grad(self, x: torch.Tensor) -> Tuple[float, torch.Tensor]:
        batch_size = x.size(0)
        device = x.device

        batch_flat = x.unsqueeze(2).view(batch_size * self.time_steps, 1, self.grid_height, self.grid_width)
        conv_out = self.cnn.conv1(batch_flat)
        conv_out = self.cnn.pool1(F.relu(self.cnn.bn1(conv_out)))
        conv_out = self.cnn.conv2(conv_out)
        conv_out = self.cnn.pool2(F.relu(self.cnn.bn2(conv_out)))
        conv_out = self.cnn.conv3(conv_out)
        post_conv = F.relu(self.cnn.bn3(conv_out))
        self._activations = conv_out

        conv_features = self.cnn.pool3(post_conv)
        conv_features = self.cnn.dropout(conv_features)
        conv_features = conv_features.view(batch_size, self.time_steps, -1)

        h0, c0 = self._build_zero_hidden(batch_size, device)
        lstm_out, _ = self.lstm(conv_features, (h0, c0))
        last_hidden = lstm_out[:, -1, :]
        logits = self.classifier(last_hidden)
        probs = torch.sigmoid(logits)
        prob_scalar = float(probs.detach().cpu().item())

        return prob_scalar, conv_out

    def compute_gradcam(
        self,
        x: torch.Tensor,
        target_class: int = 0,
        upsample_to_input: bool = True,
    ) -> Tuple[float, np.ndarray]:
        if not self._gradcam_enabled:
            self.enable_gradcam()

        self.eval()
        was_training = self.training
        device = x.device
        batch_size = x.size(0)

        self.zero_grad(set_to_none=True)

        x_var = x.clone().detach().requires_grad_(True)

        batch_flat = x_var.unsqueeze(2).view(
            batch_size * self.time_steps, 1, self.grid_height, self.grid_width
        )

        c1 = self.cnn.conv1(batch_flat)
        c1 = self.cnn.pool1(F.relu(self.cnn.bn1(c1)))
        c2 = self.cnn.conv2(c1)
        c2 = self.cnn.pool2(F.relu(self.cnn.bn2(c2)))
        c3 = self.cnn.conv3(c2)
        activations = c3
        activations.retain_grad()
        post_c3 = F.relu(self.cnn.bn3(c3))
        cnn_out = self.cnn.pool3(post_c3)
        cnn_out = self.cnn.dropout(cnn_out)
        cnn_out = cnn_out.view(batch_size, self.time_steps, -1)

        h0, c0 = self._build_zero_hidden(batch_size, device)
        lstm_out, _ = self.lstm(cnn_out, (h0, c0))
        last_hidden = lstm_out[:, -1, :]
        logits = self.classifier(last_hidden)
        probs = torch.sigmoid(logits)
        probs_np = probs.detach().cpu().numpy().reshape(-1)
        prob_scalar = float(probs_np[0])

        target = torch.zeros_like(logits)
        target[:, target_class] = 1.0
        logits.backward(gradient=target, retain_graph=False)

        gradients = activations.grad
        if gradients is None:
            gradients = torch.zeros_like(activations)

        act = activations.detach()
        grad = gradients.detach()

        b_t, c, h, w = act.shape
        act_reshaped = act.view(batch_size, self.time_steps, c, h, w)
        grad_reshaped = grad.view(batch_size, self.time_steps, c, h, w)

        act_last = act_reshaped[:, -1, :, :, :]
        grad_last = grad_reshaped[:, -1, :, :, :]

        weights = torch.mean(grad_last, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * act_last, dim=1, keepdim=True)
        cam = F.relu(cam)

        if upsample_to_input:
            cam = F.interpolate(
                cam,
                size=(self.grid_height, self.grid_width),
                mode="bilinear",
                align_corners=False,
            )

        cam = cam[0:1, ...]
        cam_np = cam.squeeze().cpu().numpy()
        if cam_np.ndim == 0:
            cam_np = np.array([[float(cam_np)]])
        elif cam_np.ndim == 1:
            cam_np = cam_np.reshape(1, -1)
        cam_max = cam_np.max() if cam_np.max() > 0 else 1e-8
        cam_np = cam_np / cam_max
        cam_np = np.clip(cam_np, 0.0, 1.0)

        del (
            x_var, batch_flat, c1, c2, c3, activations, post_c3,
            cnn_out, lstm_out, last_hidden, logits, probs, probs_np, target,
            gradients, act, grad, act_reshaped, grad_reshaped,
            act_last, grad_last, weights, cam,
        )
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
        gc.collect()

        self.zero_grad(set_to_none=True)
        if was_training:
            self.train()

        return prob_scalar, cam_np.astype(np.float32)

    def reset_hidden_states(self) -> None:
        if self._cached_hidden is not None:
            h, c = self._cached_hidden
            del h, c
            self._cached_hidden = None
        if next(self.parameters()).device.type == "cuda":
            dev = next(self.parameters()).device
            with torch.cuda.device(dev):
                torch.cuda.empty_cache()
        gc.collect()

    def reset_batch_state(self) -> None:
        self.reset_hidden_states()
        for module in self.modules():
            if hasattr(module, "reset_running_stats") and callable(module.reset_running_stats):
                try:
                    module.reset_running_stats()
                except Exception:
                    pass


def load_model(
    model_path: Optional[str] = None,
    time_steps: int = 30,
    grid_height: int = 16,
    grid_width: int = 24,
    device: Optional[str] = None,
) -> CNNLSTMModel:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CNNLSTMModel(
        time_steps=time_steps,
        grid_height=grid_height,
        grid_width=grid_width,
    )

    if model_path is not None:
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict, strict=True)
        except Exception:
            pass

    model.to(device)
    model.eval()
    model.reset_batch_state()
    return model
