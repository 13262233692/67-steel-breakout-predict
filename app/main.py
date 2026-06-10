import logging
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.utils.sensor_mapping import SensorMapper
from app.services.spatiotemporal_grid import SpatiotemporalGrid
from app.services.kafka_consumer import ThermocoupleKafkaConsumer
from app.services.inference_engine import BreakoutInferenceEngine
from app.schemas.prediction import (
    ThermocoupleReading,
    ThermocoupleBatch,
    BreakoutPrediction,
    GridTensorInfo,
    SystemStatus,
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
        "时空网格已初始化: 窗口=%d, 网格=%dx%d",
        spatiotemporal_grid.time_window,
        spatiotemporal_grid.grid_height,
        spatiotemporal_grid.grid_width,
    )

    inference_engine = BreakoutInferenceEngine(spatiotemporal_grid)
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
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", summary="健康检查")
async def health_check():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/api/v1/status", response_model=SystemStatus, summary="获取系统状态")
async def get_system_status():
    return SystemStatus(
        kafka_connected=kafka_consumer.status.connected if kafka_consumer else False,
        model_loaded=inference_engine.status.model_loaded if inference_engine else False,
        grid_ready=spatiotemporal_grid.is_ready() if spatiotemporal_grid else False,
        latest_prediction=inference_engine.status.latest_prediction if inference_engine else None,
        total_readings_processed=spatiotemporal_grid.total_readings_processed if spatiotemporal_grid else 0,
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


@app.post("/api/v1/ingest", summary="手动摄入热电偶读数（调试用）")
async def ingest_readings(batch: ThermocoupleBatch):
    if spatiotemporal_grid is None:
        raise HTTPException(status_code=503, detail="时空网格未初始化")
    spatiotemporal_grid.ingest_batch(batch.readings)
    return {
        "ingested": len(batch.readings),
        "time_steps_filled": spatiotemporal_grid.time_steps_filled(),
        "is_ready": spatiotemporal_grid.is_ready(),
    }
