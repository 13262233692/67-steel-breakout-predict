import numpy as np
import time
from typing import List, Optional, Tuple, Deque
from collections import deque
from dataclasses import dataclass, field

from app.utils.sensor_mapping import SensorMapper
from app.config import settings
from app.schemas.prediction import ThermocoupleReading


@dataclass
class TimeStep:
    timestamp: float
    grid: np.ndarray
    filled_mask: np.ndarray


class SpatiotemporalGrid:
    def __init__(
        self,
        sensor_mapper: SensorMapper,
        time_window: int = settings.time_window,
        grid_height: int = settings.grid_height,
        grid_width: int = settings.grid_width,
        min_temp: float = settings.min_temp,
        max_temp: float = settings.max_temp,
    ):
        self.sensor_mapper = sensor_mapper
        self.time_window = time_window
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.temp_range = max_temp - min_temp

        self._time_steps: Deque[TimeStep] = deque(maxlen=time_window)
        self._current_step: Optional[TimeStep] = None
        self._current_step_bucket: Optional[float] = None
        self._bucket_duration: float = 1.0

        self.total_readings_processed = 0

    def _normalize_temperature(self, temp: float) -> float:
        clipped = np.clip(temp, self.min_temp, self.max_temp)
        return (clipped - self.min_temp) / self.temp_range

    def _create_empty_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        grid = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        mask = np.zeros((self.grid_height, self.grid_width), dtype=bool)
        return grid, mask

    def _get_time_bucket(self, timestamp: float) -> float:
        return float(np.floor(timestamp / self._bucket_duration) * self._bucket_duration)

    def _finalize_current_step(self) -> None:
        if self._current_step is not None:
            self._fill_missing_values(self._current_step)
            self._time_steps.append(self._current_step)
            self._current_step = None
            self._current_step_bucket = None

    def _fill_missing_values(self, step: TimeStep) -> None:
        if not np.any(step.filled_mask):
            return
        mean_val = float(np.mean(step.grid[step.filled_mask]))
        step.grid[~step.filled_mask] = mean_val
        step.filled_mask[~step.filled_mask] = True

    def _ensure_time_bucket(self, timestamp: float) -> None:
        bucket = self._get_time_bucket(timestamp)
        if self._current_step_bucket is None:
            self._current_step_bucket = bucket
            grid, mask = self._create_empty_grid()
            self._current_step = TimeStep(timestamp=bucket, grid=grid, filled_mask=mask)
            return
        if bucket > self._current_step_bucket:
            self._finalize_current_step()
            self._current_step_bucket = bucket
            grid, mask = self._create_empty_grid()
            self._current_step = TimeStep(timestamp=bucket, grid=grid, filled_mask=mask)

    def ingest_reading(self, reading: ThermocoupleReading) -> None:
        coords = self.sensor_mapper.get_coordinates(reading.sensor_id)
        if coords is None:
            return
        row, col = coords
        if not (0 <= row < self.grid_height and 0 <= col < self.grid_width):
            return
        self._ensure_time_bucket(reading.timestamp)
        if self._current_step is None:
            return
        norm_temp = self._normalize_temperature(reading.temperature)
        self._current_step.grid[row, col] = norm_temp
        self._current_step.filled_mask[row, col] = True
        self.total_readings_processed += 1

    def ingest_batch(self, readings: List[ThermocoupleReading]) -> None:
        sorted_readings = sorted(readings, key=lambda r: r.timestamp)
        for reading in sorted_readings:
            self.ingest_reading(reading)

    def flush(self) -> None:
        self._finalize_current_step()

    def is_ready(self) -> bool:
        return len(self._time_steps) >= self.time_window

    def time_steps_filled(self) -> int:
        return len(self._time_steps)

    def get_tensor(self) -> Optional[np.ndarray]:
        if not self.is_ready():
            return None
        tensor = np.stack([step.grid for step in self._time_steps], axis=0)
        return tensor.astype(np.float32)

    def get_tensor_with_batch(self) -> Optional[np.ndarray]:
        tensor = self.get_tensor()
        if tensor is None:
            return None
        return tensor[np.newaxis, :, :, :]

    def get_shape(self) -> Tuple[int, int, int]:
        return (self.time_window, self.grid_height, self.grid_width)

    def reset(self) -> None:
        self._time_steps.clear()
        self._current_step = None
        self._current_step_bucket = None
        self.total_readings_processed = 0
