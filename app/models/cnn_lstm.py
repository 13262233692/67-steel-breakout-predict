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
        cnn_out_channels: int = 64,
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

    def _forward_cnn_per_step(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, H, W = x.shape
        x = x.unsqueeze(2)
        x = x.view(batch_size * time_steps, 1, H, W)
        cnn_features = self.cnn(x)
        cnn_features = cnn_features.view(batch_size, time_steps, -1)
        return cnn_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        cnn_features = self._forward_cnn_per_step(x)

        h0 = torch.zeros(
            self.lstm_num_layers, batch_size, self.lstm_hidden_size, device=x.device
        )
        c0 = torch.zeros(
            self.lstm_num_layers, batch_size, self.lstm_hidden_size, device=x.device
        )

        lstm_out, _ = self.lstm(cnn_features, (h0, c0))
        last_hidden = lstm_out[:, -1, :]
        logits = self.classifier(last_hidden)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.sigmoid(logits)


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
            model.load_state_dict(state_dict)
        except Exception:
            pass

    model.to(device)
    model.eval()
    return model
