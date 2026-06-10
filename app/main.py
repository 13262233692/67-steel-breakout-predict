import logging
import time
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import settings
from app.utils.sensor_mapping import SensorMapper
from app.services.spatiotemporal_grid import SpatiotemporalGrid
from app.services.kafka_consumer import ThermocoupleKafkaConsumer
from app.services.inference_engine import (
    BreakoutInferenceEngine,
    BatchSwitchReport,
    LifecycleState,
)
from app.schemas.prediction import (
    ThermocoupleReading,
    ThermocoupleBatch,
    BreakoutPrediction,
    GridTensorInfo,
    SystemStatus,
    BreakoutExplanation,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("steel-breakout-api")

sensor_mapper: Optional[SensorMapper] = None
spatiotemporal_grid: Optional[SpatiotemporalGrid] = None
kafka_consumer: Optional[ThermocoupleKafkaConsumer] = None
inference_engine: Optional[BreakoutInferenceEngine] = None


def _on_readings_callback(readings: List[ThermocoupleReading]) -> None:
    if spatiotemporal_grid is not None:
        spatiotemporal_grid.ingest_batch(readings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global sensor_mapper, spatiotemporal_grid, kafka_consumer, inference_engine

    logger.info("系统启动中...")

    sensor_mapper = SensorMapper(settings.sensor_layout_path)
    logger.info(
        "传感器映射已加载: %d 个传感器, 网格 %dx%d",
        sensor_mapper.total_sensors(),
        sensor_mapper.get_grid_dimensions()[0],
        sensor_mapper.get_grid_dimensions()[1],
    )

    spatiotemporal_grid = SpatiotemporalGrid(sensor_mapper)
    logger.info(
        "时空网格已初始化: 窗口=%d, 网格=%dx%d, 初始批次=%s",
        spatiotemporal_grid.time_window,
        spatiotemporal_grid.grid_height,
        spatiotemporal_grid.grid_width,
        spatiotemporal_grid.get_current_batch_id(),
    )

    inference_engine = BreakoutInferenceEngine(spatiotemporal_grid, sensor_mapper)
    inference_engine.load_model()

    kafka_consumer = ThermocoupleKafkaConsumer(
        sensor_mapper=sensor_mapper,
        on_readings_callback=_on_readings_callback,
    )
    await kafka_consumer.start()
    await inference_engine.start()

    logger.info("✅ 系统启动完成")
    yield

    logger.info("系统关闭中...")
    if kafka_consumer is not None:
        await kafka_consumer.stop()
    if inference_engine is not None:
        await inference_engine.stop()
    logger.info("系统已关闭")


app = FastAPI(
    title="炼钢厂漏钢预测 AI 引擎",
    description="基于CNN-LSTM混合模型的连铸结晶器粘结漏钢实时预测系统",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class BatchSwitchResponse(BaseModel):
    old_batch_id: str
    new_batch_id: str
    old_inference_count: int
    old_reading_count: int
    gc_triggered: bool
    completed_ts: float = Field(default_factory=time.time)


class EngineStatusResponse(BaseModel):
    model_loaded: bool
    lifecycle_state: str
    current_batch_id: Optional[str]
    batches_completed: int
    total_inferences: int
    inference_count_current_batch: int
    avg_inference_ms: float
    gc_count: int
    last_gc_ts: float
    cuda_mem_peak_mb: float
    latest_prediction: Optional[BreakoutPrediction]
    gradcam_enabled: bool
    gradcam_explanation_count: int
    gradcam_alert_threshold: float
    latest_explanation: Optional[BreakoutExplanation]


@app.get("/health", summary="健康检查")
async def health_check():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/api/v1/status", response_model=SystemStatus, summary="获取系统状态")
async def get_system_status():
    return SystemStatus(
        kafka_connected=kafka_consumer.status.connected if kafka_consumer else False,
        model_loaded=inference_engine.status.model_loaded if inference_engine else False,
        grid_ready=spatiotemporal_grid.is_ready() if spatiotemporal_grid else False,
        latest_prediction=(
            inference_engine.status.latest_prediction if inference_engine else None
        ),
        total_readings_processed=(
            spatiotemporal_grid.total_readings_processed if spatiotemporal_grid else 0
        ),
    )


@app.get("/api/v1/engine/status", response_model=EngineStatusResponse, summary="推理引擎详细状态")
async def get_engine_status():
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="推理引擎未初始化")
    s = inference_engine.status
    return EngineStatusResponse(
        model_loaded=s.model_loaded,
        lifecycle_state=s.lifecycle_state.value,
        current_batch_id=s.current_batch_id,
        batches_completed=s.batches_completed,
        total_inferences=s.total_inferences,
        inference_count_current_batch=s.inference_count_current_batch,
        avg_inference_ms=s.avg_inference_ms,
        gc_count=s.gc_count,
        last_gc_ts=s.last_gc_ts,
        cuda_mem_peak_mb=s.cuda_mem_peak_mb,
        latest_prediction=s.latest_prediction,
        gradcam_enabled=s.gradcam_enabled,
        gradcam_explanation_count=s.gradcam_explanation_count,
        gradcam_alert_threshold=s.gradcam_alert_threshold,
        latest_explanation=s.latest_explanation,
    )


@app.get(
    "/api/v1/prediction",
    response_model=Optional[BreakoutPrediction],
    summary="获取最新漏钢预测结果",
)
async def get_latest_prediction():
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="推理引擎未初始化")
    return inference_engine.status.latest_prediction


@app.get(
    "/api/v1/explain",
    response_model=Optional[BreakoutExplanation],
    summary="获取最新Grad-CAM可解释性热力图与异常传感器定位",
)
async def get_latest_explanation():
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="推理引擎未初始化")
    if not inference_engine.status.gradcam_enabled:
        raise HTTPException(status_code=503, detail="Grad-CAM解释模块未启用")
    return inference_engine.status.latest_explanation


@app.post(
    "/api/v1/explain/trigger",
    response_model=BreakoutExplanation,
    summary="手动触发Grad-CAM解释（无论是否告警）",
)
async def trigger_explanation(force: bool = True):
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="推理引擎未初始化")
    if not inference_engine.status.gradcam_enabled:
        raise HTTPException(status_code=503, detail="Grad-CAM解释模块未启用")
    explanation = inference_engine.explain_current(force=force)
    if explanation is None:
        raise HTTPException(
            status_code=400,
            detail="时空网格数据不足或Grad-CAM计算失败，请等待更多传感器数据",
        )
    return explanation


@app.post(
    "/api/v1/prediction/trigger",
    response_model=Optional[BreakoutPrediction],
    summary="手动触发一次推理",
)
async def trigger_inference():
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="推理引擎未初始化")
    result = inference_engine.run_inference()
    if result is None:
        raise HTTPException(
            status_code=400,
            detail="时空网格数据不足，请等待更多传感器数据",
        )
    return result


@app.get("/api/v1/grid", response_model=GridTensorInfo, summary="获取时空网格状态")
async def get_grid_info():
    if spatiotemporal_grid is None:
        raise HTTPException(status_code=503, detail="时空网格未初始化")
    return GridTensorInfo(
        shape=list(spatiotemporal_grid.get_shape()),
        time_steps_filled=spatiotemporal_grid.time_steps_filled(),
        is_ready=spatiotemporal_grid.is_ready(),
    )


@app.get(
    "/api/v1/batch/current",
    summary="获取当前生产批次ID及元数据",
)
async def get_current_batch():
    if spatiotemporal_grid is None:
        raise HTTPException(status_code=503, detail="时空网格未初始化")
    info = spatiotemporal_grid.get_batch_info()
    return {
        "batch_id": info.batch_id,
        "start_ts": info.start_ts,
        "end_ts": info.end_ts,
        "readings_count": info.readings_count,
        "inference_count": info.inference_count,
        "lifecycle_state": (
            inference_engine.status.lifecycle_state.value if inference_engine else "UNKNOWN"
        ),
    }


@app.post(
    "/api/v1/batch/switch",
    response_model=BatchSwitchResponse,
    summary="批次切换（换包/换批次）：清空时空滑窗+重置LSTM+强制GC",
)
async def switch_batch():
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="推理引擎未初始化")
    report: BatchSwitchReport = inference_engine.switch_batch()
    return BatchSwitchResponse(
        old_batch_id=report.old_batch_id,
        new_batch_id=report.new_batch_id,
        old_inference_count=report.old_inference_count,
        old_reading_count=report.old_reading_count,
        gc_triggered=report.gc_triggered,
    )


@app.post(
    "/api/v1/system/force_gc",
    summary="强制触发显存/内存垃圾回收（运维用）",
)
async def force_gc():
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="推理引擎未初始化")
    gc_count_before = inference_engine.status.gc_count
    inference_engine._aggressive_gc()
    return {
        "gc_count_before": gc_count_before,
        "gc_count_after": inference_engine.status.gc_count,
        "last_gc_ts": inference_engine.status.last_gc_ts,
        "cuda_mempeak_mb": inference_engine.status.cuda_mem_peak_mb,
        "triggered": True,
    }


@app.post("/api/v1/ingest", summary="手动摄入热电偶读数（调试用）")
async def ingest_readings(batch: ThermocoupleBatch):
    if spatiotemporal_grid is None:
        raise HTTPException(status_code=503, detail="时空网格未初始化")
    spatiotemporal_grid.ingest_batch(batch.readings)
    return {
        "ingested": len(batch.readings),
        "time_steps_filled": spatiotemporal_grid.time_steps_filled(),
        "is_ready": spatiotemporal_grid.is_ready(),
        "batch_id": spatiotemporal_grid.get_current_batch_id(),
    }
