import asyncio
import gc
import time
import logging
import os
import uuid
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

import torch
import numpy as np

from app.config import settings
from app.models.cnn_lstm import CNNLSTMModel, load_model
from app.services.spatiotemporal_grid import SpatiotemporalGrid, BatchInfo
from app.services.gradcam_explainer import GradCAMExplainer
from app.schemas.prediction import BreakoutPrediction, BreakoutExplanation
from app.utils.sensor_mapping import SensorMapper

logger = logging.getLogger(__name__)


class LifecycleState(str, Enum):
    IDLE = "IDLE"
    ACTIVE = "ACTIVE"
    BATCH_SWITCHING = "BATCH_SWITCHING"
    ERROR = "ERROR"


@dataclass
class InferenceStatus:
    model_loaded: bool = False
    running: bool = False
    total_inferences: int = 0
    avg_inference_ms: float = 0.0
    latest_prediction: Optional[BreakoutPrediction] = None
    lifecycle_state: LifecycleState = LifecycleState.IDLE
    current_batch_id: Optional[str] = None
    batches_completed: int = 0
    inference_count_current_batch: int = 0
    last_gc_ts: float = 0.0
    gc_count: int = 0
    cuda_mem_peak_mb: float = 0.0
    gradcam_enabled: bool = False
    gradcam_explanation_count: int = 0
    gradcam_alert_threshold: float = 0.85
    latest_explanation: Optional[BreakoutExplanation] = None


@dataclass
class BatchSwitchReport:
    old_batch_id: str
    new_batch_id: str
    old_inference_count: int
    old_reading_count: int
    gc_triggered: bool = True


class BreakoutInferenceEngine:
    _INFERENCES_PER_GC = 60
    _MAX_INFERENCE_TIMES = 200

    def __init__(
        self,
        grid: SpatiotemporalGrid,
        sensor_mapper: SensorMapper,
        model_path: Optional[str] = None,
        interval: float = settings.inference_interval,
        alert_threshold: float = 0.7,
        gradcam_alert_threshold: float = 0.85,
        gradcam_auto_explain: bool = True,
    ):
        self.grid = grid
        self.sensor_mapper = sensor_mapper
        self.model_path = model_path or settings.model_path
        self.interval = interval
        self.alert_threshold = alert_threshold
        self.gradcam_alert_threshold = gradcam_alert_threshold
        self.gradcam_auto_explain = gradcam_auto_explain
        self.status = InferenceStatus()
        self.status.gradcam_alert_threshold = gradcam_alert_threshold

        self._model: Optional[CNNLSTMModel] = None
        self._explainer: Optional[GradCAMExplainer] = None
        self._device: str = "cpu"
        self._cuda_device: Optional[torch.device] = None
        self._task: Optional[asyncio.Task] = None
        self._inference_times: list[float] = []
        self._inferences_since_gc: int = 0

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
            self._cuda_device = (
                next(self._model.parameters()).device
                if self._device == "cuda"
                else None
            )
            self._explainer = GradCAMExplainer(
                model=self._model,
                sensor_mapper=self.sensor_mapper,
                alert_threshold=self.gradcam_alert_threshold,
                top_k_sensors=10,
            )
            self.status.model_loaded = True
            self.status.gradcam_enabled = True
            self.status.lifecycle_state = LifecycleState.ACTIVE
            self.status.current_batch_id = self.grid.get_current_batch_id()
            logger.info("模型加载成功，设备: %s", self._device)
            if not model_exists:
                logger.warning("模型文件不存在，使用随机初始化权重（仅用于测试）")
            self._aggressive_gc()
            return True
        except Exception as e:
            logger.error("模型加载失败: %s", e)
            self.status.model_loaded = False
            self.status.gradcam_enabled = False
            self.status.lifecycle_state = LifecycleState.ERROR
            return False

    def _aggressive_gc(self) -> None:
        gc.collect()
        if self._cuda_device is not None:
            try:
                with torch.cuda.device(self._cuda_device):
                    torch.cuda.empty_cache()
                    peak = float(torch.cuda.max_memory_allocated(self._cuda_device)) / (1024 * 1024)
                    self.status.cuda_mem_peak_mb = max(self.status.cuda_mem_peak_mb, peak)
                    torch.cuda.reset_peak_memory_stats(self._cuda_device)
            except Exception:
                pass
        self.status.last_gc_ts = time.time()
        self.status.gc_count += 1

    def _maybe_gc(self) -> None:
        self._inferences_since_gc += 1
        if self._inferences_since_gc >= self._INFERENCES_PER_GC:
            self._aggressive_gc()
            self._inferences_since_gc = 0

    def run_inference(self) -> Optional[BreakoutPrediction]:
        if not self.status.model_loaded or self._model is None:
            return None
        if self.status.lifecycle_state == LifecycleState.BATCH_SWITCHING:
            return None

        tensor_np = self.grid.get_tensor_with_batch()
        if tensor_np is None:
            return None

        start = time.perf_counter()
        prob_value: Optional[float] = None
        try:
            device = self._cuda_device if self._cuda_device is not None else torch.device("cpu")
            input_tensor = torch.from_numpy(tensor_np).to(device)
            prob_value = self._model.predict_breakout(
                input_tensor,
                clear_intermediates=False,
            )
            del input_tensor
        except Exception as e:
            logger.error("推理执行失败: %s", e)
            return None
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._inference_times.append(elapsed_ms)
            if len(self._inference_times) > self._MAX_INFERENCE_TIMES:
                drop = len(self._inference_times) - self._MAX_INFERENCE_TIMES
                del self._inference_times[:drop]
            self.status.avg_inference_ms = float(np.mean(self._inference_times))
            self._maybe_gc()

        if prob_value is None:
            return None

        prediction = BreakoutPrediction(
            breakout_probability=float(prob_value),
            timestamp=datetime.now(timezone.utc),
            is_alert=bool(prob_value >= self.alert_threshold),
            alert_threshold=self.alert_threshold,
        )

        self.status.latest_prediction = prediction
        self.status.total_inferences += 1
        self.status.inference_count_current_batch += 1
        batch_info = self.grid.get_batch_info()
        batch_info.inference_count += 1

        if prediction.is_alert:
            logger.warning(
                "⚠️ 漏钢告警！批次=%s 概率=%.4f 时间=%s",
                self.status.current_batch_id,
                prob_value,
                prediction.timestamp.isoformat(),
            )
            if self.gradcam_auto_explain and prob_value >= self.gradcam_alert_threshold and self._explainer is not None:
                try:
                    tensor_4d = self.grid.get_tensor_with_batch()
                    if tensor_4d is not None:
                        device = self._cuda_device if self._cuda_device is not None else torch.device("cpu")
                        explain_tensor = torch.from_numpy(tensor_4d).to(device)
                        explanation = self._explainer.explain_from_grid(explain_tensor, force=False)
                        if explanation is not None:
                            self.status.latest_explanation = explanation
                            self.status.gradcam_explanation_count += 1
                            logger.info(
                                "🎯 Grad-CAM解释生成: 异常热点=%d个, Top权重=%.3f",
                                len(explanation.anomaly_sensors),
                                explanation.anomaly_sensors[0].attention_weight if explanation.anomaly_sensors else 0.0,
                            )
                        del explain_tensor
                except Exception as e:
                    logger.error("Grad-CAM解释失败: %s", e)
        return prediction

    def explain_current(self, force: bool = True) -> Optional[BreakoutExplanation]:
        if not self.status.model_loaded or self._model is None or self._explainer is None:
            return None
        tensor_np = self.grid.get_tensor_with_batch()
        if tensor_np is None:
            return None
        try:
            device = self._cuda_device if self._cuda_device is not None else torch.device("cpu")
            input_tensor = torch.from_numpy(tensor_np).to(device)
            explanation = self._explainer.explain_from_grid(input_tensor, force=force)
            del input_tensor
            if explanation is not None:
                self.status.latest_explanation = explanation
                if force:
                    self.status.gradcam_explanation_count += 1
            return explanation
        except Exception as e:
            logger.error("显式Grad-CAM解释失败: %s", e)
            return None

    def switch_batch(self) -> BatchSwitchReport:
        self.status.lifecycle_state = LifecycleState.BATCH_SWITCHING
        logger.info("🔄 开始批次切换...")

        old_batch_id = self.status.current_batch_id or "UNKNOWN"
        old_inferences = self.status.inference_count_current_batch
        old_readings = self.grid.get_batch_info().readings_count

        old_batch_info: BatchInfo = self.grid.batch_switch()
        del old_batch_info

        if self._model is not None:
            self._model.reset_batch_state()

        self._aggressive_gc()

        new_batch_id = self.grid.get_current_batch_id()
        self.status.current_batch_id = new_batch_id
        self.status.inference_count_current_batch = 0
        self.status.batches_completed += 1
        self.status.lifecycle_state = LifecycleState.ACTIVE

        report = BatchSwitchReport(
            old_batch_id=old_batch_id,
            new_batch_id=new_batch_id,
            old_inference_count=old_inferences,
            old_reading_count=old_readings,
        )
        logger.info(
            "✅ 批次切换完成: %s → %s (推理:%d 读数:%d)",
            report.old_batch_id,
            report.new_batch_id,
            report.old_inference_count,
            report.old_reading_count,
        )
        return report

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
        self._aggressive_gc()
        logger.info("推理引擎已停止")
