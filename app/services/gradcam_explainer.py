import gc
import numpy as np
import torch
from typing import Optional, Tuple, List
from app.models.cnn_lstm import CNNLSTMModel
from app.utils.sensor_mapping import SensorMapper, SensorInfo
from app.schemas.prediction import (
    BreakoutPrediction,
    GradCAMHeatmap,
    AnomalySensor,
    BreakoutExplanation,
)
from app.config import settings


class GradCAMExplainer:
    def __init__(
        self,
        model: CNNLSTMModel,
        sensor_mapper: SensorMapper,
        alert_threshold: float = 0.85,
        top_k_sensors: int = 10,
    ):
        self.model = model
        self.sensor_mapper = sensor_mapper
        self.alert_threshold = alert_threshold
        self.top_k_sensors = top_k_sensors
        self.grid_height = settings.grid_height
        self.grid_width = settings.grid_width
        self._gradcam_enabled = False
        self._ensure_gradcam_enabled()

    def _ensure_gradcam_enabled(self) -> None:
        if not self._gradcam_enabled:
            self.model.enable_gradcam()
            self._gradcam_enabled = True

    def _map_heatmap_to_sensors(
        self,
        heatmap: np.ndarray,
        current_grid: Optional[np.ndarray] = None,
    ) -> List[AnomalySensor]:
        assert heatmap.shape == (self.grid_height, self.grid_width), \
            f"Heatmap shape {heatmap.shape} != expected ({self.grid_height}, {self.grid_width})"

        sensors: List[AnomalySensor] = []

        for row in range(self.grid_height):
            for col in range(self.grid_width):
                sensor_id = self.sensor_mapper.get_sensor_id(row, col)
                if sensor_id is None:
                    continue
                info = self.sensor_mapper.get_sensor_info(sensor_id)
                if info is None:
                    continue
                weight = float(heatmap[row, col])
                current_temp = None
                if current_grid is not None and current_grid.shape == (self.grid_height, self.grid_width):
                    norm_val = float(current_grid[row, col])
                    current_temp = settings.min_temp + norm_val * (settings.max_temp - settings.min_temp)
                sensor = AnomalySensor(
                    sensor_id=sensor_id,
                    row=row,
                    col=col,
                    physical_x_mm=info.physical_x_mm,
                    physical_y_mm=info.physical_y_mm,
                    attention_weight=weight,
                    current_temperature=current_temp,
                    is_highlight=False,
                )
                sensors.append(sensor)

        sensors.sort(key=lambda s: (-s.attention_weight, s.row, s.col))
        top_k = min(self.top_k_sensors, len(sensors))
        for i in range(top_k):
            sensors[i].is_highlight = True

        return sensors

    def _heatmap_to_list(self, heatmap: np.ndarray) -> List[List[float]]:
        return [[float(v) for v in row] for row in heatmap]

    def explain(
        self,
        input_tensor: torch.Tensor,
        current_grid: Optional[np.ndarray] = None,
        force: bool = False,
    ) -> Optional[BreakoutExplanation]:
        if input_tensor is None:
            return None
        self._ensure_gradcam_enabled()

        self.model.eval()
        device = next(self.model.parameters()).device
        x = input_tensor.to(device)

        try:
            prob, heatmap = self.model.compute_gradcam(
                x,
                target_class=0,
                upsample_to_input=True,
            )
        except Exception:
            return None

        is_alert = prob >= self.alert_threshold
        if not is_alert and not force:
            return None

        prediction = BreakoutPrediction(
            breakout_probability=prob,
            is_alert=is_alert,
            alert_threshold=self.alert_threshold,
        )

        gradcam_heatmap = GradCAMHeatmap(
            grid_height=self.grid_height,
            grid_width=self.grid_width,
            heatmap_values=self._heatmap_to_list(heatmap),
        )

        anomaly_sensors = self._map_heatmap_to_sensors(heatmap, current_grid)

        explanation = BreakoutExplanation(
            prediction=prediction,
            heatmap=gradcam_heatmap,
            anomaly_sensors=anomaly_sensors,
            top_k=self.top_k_sensors,
            gradcam_enabled=True,
        )

        del x, heatmap
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
        gc.collect()

        return explanation

    def explain_from_grid(
        self,
        grid_tensor: torch.Tensor,
        force: bool = False,
    ) -> Optional[BreakoutExplanation]:
        if grid_tensor is None:
            return None

        current_grid = None
        if grid_tensor.ndim == 4:
            current_grid_np = grid_tensor.detach().cpu().numpy()
            if current_grid_np.shape[0] == 1 and current_grid_np.shape[1] >= 1:
                current_grid = current_grid_np[0, -1, :, :]
        elif grid_tensor.ndim == 3:
            current_grid_np = grid_tensor.detach().cpu().numpy()
            if current_grid_np.shape[0] >= 1:
                current_grid = current_grid_np[-1, :, :]

        return self.explain(grid_tensor, current_grid, force=force)

    def cleanup(self) -> None:
        if self._gradcam_enabled:
            self.model.disable_gradcam()
            self._gradcam_enabled = False
