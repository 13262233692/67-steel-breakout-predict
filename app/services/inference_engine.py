import asyncio
import time
import logging
import os
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone

import torch
import numpy as np

from app.config import settings
from app.models.cnn_lstm import CNNLSTMModel, load_model
from app.services.spatiotemporal_grid import SpatiotemporalGrid
from app.schemas.prediction import BreakoutPrediction

logger = logging.getLogger(__name__)


@dataclass
class InferenceStatus:
    model_loaded: bool = False
    running: bool = False
    total_inferences: int = 0
    avg_inference_ms: float = 0.0
    latest_prediction: Optional[BreakoutPrediction] = None


class BreakoutInferenceEngine:
    def __init__(
        self,
        grid: SpatiotemporalGrid,
        model_path: Optional[str] = None,
        interval: float = settings.inference_interval,
        alert_threshold: float = 0.7,
    ):
        self.grid = grid
        self.model_path = model_path or settings.model_path
        self.interval = interval
        self.alert_threshold = alert_threshold
        self.status = InferenceStatus()

        self._model: Optional[CNNLSTMModel] = None
        self._device: str = "cpu"
        self._task: Optional[asyncio.Task] = None
        self._inference_times: list[float] = []

    def load_model(self) -> bool:
        try:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            model_exists = os.path.exists(self.model_path)
            path_to_load = self.model_path if model_exists else None
            self._model = load_model(
                model_path=path_to_load,
                time_steps=self.grid.time_window,
                grid_height=self.grid.grid_height,
                grid_width=self.grid.grid_width,
                device=self._device,
            )
            self.status.model_loaded = True
            logger.info("模型加载成功，设备: %s", self._device)
            if not model_exists:
                logger.warning("模型文件不存在，使用随机初始化权重（仅用于测试）")
            return True
        except Exception as e:
            logger.error("模型加载失败: %s", e)
            self.status.model_loaded = False
            return False

    def run_inference(self) -> Optional[BreakoutPrediction]:
        if not self.status.model_loaded or self._model is None:
            return None
        tensor_np = self.grid.get_tensor_with_batch()
        if tensor_np is None:
            return None

        start = time.perf_counter()
        try:
            with torch.no_grad():
                tensor = torch.from_numpy(tensor_np).to(self._device)
                prob = self._model.predict_proba(tensor)
                prob_value = float(prob.cpu().numpy().squeeze())
        except Exception as e:
            logger.error("推理执行失败: %s", e)
            return None
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._inference_times.append(elapsed_ms)
            if len(self._inference_times) > 100:
                self._inference_times.pop(0)
            self.status.avg_inference_ms = float(np.mean(self._inference_times))

        prediction = BreakoutPrediction(
            breakout_probability=prob_value,
            timestamp=datetime.now(timezone.utc),
            is_alert=prob_value >= self.alert_threshold,
            alert_threshold=self.alert_threshold,
        )
        self.status.latest_prediction = prediction
        self.status.total_inferences += 1

        if prediction.is_alert:
            logger.warning(
                "⚠️ 漏钢告警！概率: %.4f, 时间: %s",
                prob_value,
                prediction.timestamp.isoformat(),
            )
        return prediction

    async def _inference_loop(self) -> None:
        self.status.running = True
        logger.info("推理引擎启动，间隔: %.1fs", self.interval)
        while True:
            try:
                self.run_inference()
            except Exception as e:
                logger.error("推理循环异常: %s", e)
            await asyncio.sleep(self.interval)

    async def start(self) -> None:
        if not self.status.model_loaded:
            self.load_model()
        if self._task is None:
            self._task = asyncio.create_task(self._inference_loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self.status.running = False
        logger.info("推理引擎已停止")
