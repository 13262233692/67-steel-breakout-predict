from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Tuple
from datetime import datetime, timezone


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ThermocoupleReading(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    sensor_id: str = Field(..., description="热电偶传感器ID")
    temperature: float = Field(..., description="温度值（摄氏度）")
    timestamp: float = Field(..., description="Unix时间戳（秒）")


class ThermocoupleBatch(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    readings: List[ThermocoupleReading] = Field(..., description="一批热电偶读数")


class BreakoutPrediction(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    breakout_probability: float = Field(..., ge=0.0, le=1.0, description="漏钢概率")
    timestamp: datetime = Field(default_factory=_utcnow, description="预测时间")
    model_version: str = Field(default="1.0.0", description="模型版本")
    is_alert: bool = Field(default=False, description="是否触发告警（概率>0.7）")
    alert_threshold: float = Field(default=0.7, description="告警阈值")


class GridTensorInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    shape: List[int] = Field(..., description="3D张量形状 [Time, Height, Width]")
    time_steps_filled: int = Field(..., description="已填充的时间步数")
    is_ready: bool = Field(..., description="是否可以进行推理")


class SystemStatus(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    kafka_connected: bool = Field(default=False, description="Kafka连接状态")
    model_loaded: bool = Field(default=False, description="模型加载状态")
    grid_ready: bool = Field(default=False, description="时空网格就绪状态")
    latest_prediction: Optional[BreakoutPrediction] = Field(default=None, description="最新预测结果")
    total_readings_processed: int = Field(default=0, description="已处理的读数总数")


class AnomalySensor(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    sensor_id: str = Field(..., description="异常热电偶传感器ID")
    row: int = Field(..., description="结晶器铜板物理行坐标")
    col: int = Field(..., description="结晶器铜板物理列坐标")
    physical_x_mm: float = Field(..., description="结晶器宽度方向物理坐标(mm)")
    physical_y_mm: float = Field(..., description="结晶器拉坯方向物理坐标(mm)")
    attention_weight: float = Field(..., ge=0.0, le=1.0, description="Grad-CAM注意力权重")
    current_temperature: Optional[float] = Field(default=None, description="当前温度(℃)")
    is_highlight: bool = Field(default=False, description="是否为Top-K异常热点")


class GradCAMHeatmap(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    grid_height: int = Field(..., description="热力图高度(原始网格行)")
    grid_width: int = Field(..., description="热力图宽度(原始网格列)")
    heatmap_values: List[List[float]] = Field(
        ...,
        description="归一化[0,1]热力图权重，二维数组 [行][列]，与结晶器物理坐标完全对齐",
    )
    timestamp: datetime = Field(default_factory=_utcnow, description="Grad-CAM计算时间")


class BreakoutExplanation(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    prediction: BreakoutPrediction = Field(..., description="原始预测结果")
    heatmap: GradCAMHeatmap = Field(..., description="Grad-CAM热力图，与结晶器物理尺寸对齐")
    anomaly_sensors: List[AnomalySensor] = Field(
        ...,
        description="按注意力权重降序排列的异常热电偶列表",
    )
    top_k: int = Field(default=10, description="返回的Top-K异常热点数量")
    explanation_version: str = Field(default="1.0.0", description="解释算法版本")
    gradcam_enabled: bool = Field(default=True, description="Grad-CAM是否成功启用")
