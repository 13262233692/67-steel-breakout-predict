import json
import os
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass


@dataclass
class SensorInfo:
    sensor_id: str
    row: int
    col: int
    physical_x_mm: float
    physical_y_mm: float
    description: str


class SensorMapper:
    def __init__(self, layout_path: str):
        self.layout_path = layout_path
        self.metadata: Dict[str, Any] = {}
        self.sensor_map: Dict[str, SensorInfo] = {}
        self._coord_to_sensor: Dict[Tuple[int, int], str] = {}
        self._load_and_generate()

    def _load_and_generate(self) -> None:
        with open(self.layout_path, "r", encoding="utf-8") as f:
            layout_data = json.load(f)
        self.metadata = layout_data["metadata"]
        self._generate_sensor_map()

    def _generate_sensor_map(self) -> None:
        grid_height = self.metadata["grid_height"]
        grid_width = self.metadata["grid_width"]
        row_spacing = self.metadata["row_spacing_mm"]
        col_spacing = self.metadata["col_spacing_mm"]
        id_pattern = self.metadata["sensor_id_pattern"]

        for row in range(grid_height):
            for col in range(grid_width):
                sensor_id = id_pattern.format(row=row, col=col)
                physical_x = col * col_spacing
                physical_y = row * row_spacing
                description = f"第{row}行第{col}列热电偶"
                info = SensorInfo(
                    sensor_id=sensor_id,
                    row=row,
                    col=col,
                    physical_x_mm=float(physical_x),
                    physical_y_mm=float(physical_y),
                    description=description,
                )
                self.sensor_map[sensor_id] = info
                self._coord_to_sensor[(row, col)] = sensor_id

    def get_grid_dimensions(self) -> Tuple[int, int]:
        return (self.metadata["grid_height"], self.metadata["grid_width"])

    def get_coordinates(self, sensor_id: str) -> Optional[Tuple[int, int]]:
        info = self.sensor_map.get(sensor_id)
        if info is None:
            return None
        return (info.row, info.col)

    def get_sensor_id(self, row: int, col: int) -> Optional[str]:
        return self._coord_to_sensor.get((row, col))

    def get_sensor_info(self, sensor_id: str) -> Optional[SensorInfo]:
        return self.sensor_map.get(sensor_id)

    def all_sensor_ids(self) -> List[str]:
        return list(self.sensor_map.keys())

    def total_sensors(self) -> int:
        return len(self.sensor_map)
