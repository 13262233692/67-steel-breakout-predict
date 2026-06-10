from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
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
