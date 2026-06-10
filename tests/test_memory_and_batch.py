import time
import pytest
import numpy as np
import torch

from app.config import settings
from app.utils.sensor_mapping import SensorMapper
from app.services.spatiotemporal_grid import SpatiotemporalGrid
from app.services.inference_engine import (
    BreakoutInferenceEngine,
    LifecycleState,
    BatchSwitchReport,
)
from app.models.cnn_lstm import CNNLSTMModel, load_model
from app.schemas.prediction import ThermocoupleReading


def _fill_grid(grid: SpatiotemporalGrid, time_steps: int, base_temp: float = 1200.0, base_ts: float = None):
    if base_ts is None:
        base_ts = 1_700_000_000.0
    base_ts = float(int(base_ts))
    for t in range(time_steps):
        step_ts = base_ts + float(t) * 5.0
        for row in range(grid.grid_height):
            for col in range(grid.grid_width):
                sensor_id = f"TC_R{row:02d}_C{col:02d}"
                temp = base_temp + np.random.uniform(-10, 10)
                reading = ThermocoupleReading(
                    sensor_id=sensor_id,
                    temperature=temp,
                    timestamp=step_ts,
                )
                grid.ingest_reading(reading)
    grid.flush()


class TestMemoryStripping:
    def test_predict_proba_returns_detached_tensor(self):
        model = CNNLSTMModel(time_steps=5, grid_height=8, grid_width=12)
        model.eval()

        x = torch.randn(2, 5, 8, 12)
        with torch.no_grad():
            probs = model.predict_proba(x)

        assert probs.requires_grad is False
        assert probs.is_leaf is True

    def test_predict_breakout_returns_pure_float_not_tensor(self):
        model = CNNLSTMModel(time_steps=5, grid_height=8, grid_width=12)
        model.eval()

        x = torch.randn(1, 5, 8, 12)
        result = model.predict_breakout(x, clear_intermediates=True)

        assert isinstance(result, float)
        assert not isinstance(result, torch.Tensor)
        assert 0.0 <= result <= 1.0

    def test_predict_breakout_no_tensor_leak_in_gc(self):
        import gc
        import weakref

        model = CNNLSTMModel(time_steps=5, grid_height=8, grid_width=12)
        model.eval()

        x = torch.randn(1, 5, 8, 12)
        x_ref = weakref.ref(x)

        result = model.predict_breakout(x, clear_intermediates=True)
        del x
        gc.collect()

        assert x_ref() is None, "输入Tensor应已被GC释放，存在泄漏"
        assert isinstance(result, float)

    def test_reset_batch_state_clears_everything(self):
        model = load_model(model_path=None, time_steps=5, grid_height=8, grid_width=12)
        for _ in range(5):
            x = torch.randn(2, 5, 8, 12)
            model.predict_breakout(x, clear_intermediates=False)

        model.reset_batch_state()

        dummy = torch.randn(1, 5, 8, 12)
        p = model.predict_breakout(dummy, clear_intermediates=True)
        assert 0.0 <= p <= 1.0


class TestBatchStateReset:
    def test_grid_batch_switch_clears_time_window(self):
        mapper = SensorMapper(settings.sensor_layout_path)
        grid = SpatiotemporalGrid(mapper, time_window=5)

        _fill_grid(grid, 8, base_temp=1500.0)
        assert grid.time_steps_filled() >= 5
        assert grid.is_ready()

        old_batch_id = grid.get_current_batch_id()
        old_readings = grid.get_batch_info().readings_count
        assert old_readings > 0

        tensor_before = grid.get_tensor()
        assert tensor_before is not None

        old_info = grid.batch_switch()

        assert grid.time_steps_filled() == 0
        assert grid.is_ready() is False
        assert grid.get_current_batch_id() != old_batch_id
        assert old_info.batch_id == old_batch_id
        assert old_info.end_ts is not None
        assert grid.get_batch_info().readings_count == 0

    def test_grid_batch_switch_tensor_cleared(self):
        mapper = SensorMapper(settings.sensor_layout_path)
        grid = SpatiotemporalGrid(mapper, time_window=5)

        _fill_grid(grid, 10, base_temp=1100.0)
        assert grid.get_tensor_with_batch() is not None

        grid.batch_switch()

        assert grid.get_tensor() is None
        assert grid.get_tensor_with_batch() is None

    def test_batch_switch_no_tail_contamination(self):
        mapper = SensorMapper(settings.sensor_layout_path)
        grid = SpatiotemporalGrid(mapper, time_window=5)

        _fill_grid(grid, 10, base_temp=1600.0, base_ts=1_700_000_000.0)
        grid.batch_switch()

        _fill_grid(grid, 3, base_temp=900.0, base_ts=1_800_000_000.0)
        assert grid.time_steps_filled() == 3
        assert grid.is_ready() is False

        _fill_grid(grid, 5, base_temp=900.0, base_ts=1_900_000_000.0)
        tensor = grid.get_tensor()
        assert tensor is not None
        for t in range(tensor.shape[0]):
            for r in range(tensor.shape[1]):
                for c in range(tensor.shape[2]):
                    assert tensor[t, r, c] < 0.3, "存在来自上一批次的高温数据拖尾污染"

    def test_inference_engine_batch_switch_lifecycle(self):
        mapper = SensorMapper(settings.sensor_layout_path)
        grid = SpatiotemporalGrid(mapper, time_window=5)
        engine = BreakoutInferenceEngine(grid, mapper, interval=0.01)
        engine.load_model()

        _fill_grid(grid, 10, base_temp=1200.0)
        for _ in range(3):
            engine.run_inference()

        old_batch_id = engine.status.current_batch_id
        assert engine.status.lifecycle_state == LifecycleState.ACTIVE
        assert engine.status.inference_count_current_batch > 0

        report: BatchSwitchReport = engine.switch_batch()

        assert report.old_batch_id == old_batch_id
        assert report.new_batch_id != old_batch_id
        assert engine.status.current_batch_id == report.new_batch_id
        assert engine.status.batches_completed == 1
        assert engine.status.inference_count_current_batch == 0
        assert engine.status.lifecycle_state == LifecycleState.ACTIVE

    def test_batch_switch_then_inference_no_tail(self):
        mapper = SensorMapper(settings.sensor_layout_path)
        grid = SpatiotemporalGrid(mapper, time_window=5)
        engine = BreakoutInferenceEngine(grid, mapper, interval=0.01)
        engine.load_model()

        _fill_grid(grid, 10, base_temp=1600.0)
        old_pred = engine.run_inference()
        assert old_pred is not None

        engine.switch_batch()
        pred_after_switch_empty = engine.run_inference()
        assert pred_after_switch_empty is None, "批次切换后滑窗清空，推理应返回None"

        _fill_grid(grid, 10, base_temp=900.0, base_ts=time.time() + 1000.0)
        new_pred = engine.run_inference()
        assert new_pred is not None
        assert engine.status.inference_count_current_batch >= 1


class TestGCSafety:
    def test_periodic_gc_triggered(self):
        mapper = SensorMapper(settings.sensor_layout_path)
        grid = SpatiotemporalGrid(mapper, time_window=5)
        engine = BreakoutInferenceEngine(grid, mapper, interval=0.01)
        engine._INFERENCES_PER_GC = 5
        engine.load_model()

        _fill_grid(grid, 10)
        gc_before = engine.status.gc_count

        for _ in range(8):
            engine.run_inference()

        assert engine.status.gc_count > gc_before

    def test_aggressive_gc_works(self):
        mapper = SensorMapper(settings.sensor_layout_path)
        grid = SpatiotemporalGrid(mapper, time_window=5)
        engine = BreakoutInferenceEngine(grid, mapper, interval=0.01)
        engine.load_model()

        gc_before = engine.status.gc_count
        engine._aggressive_gc()
        assert engine.status.gc_count == gc_before + 1
        assert engine.status.last_gc_ts > 0


@pytest.mark.asyncio
async def test_fastapi_batch_switch_endpoint():
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        batch1_resp = client.get("/api/v1/batch/current")
        assert batch1_resp.status_code == 200
        batch1_id = batch1_resp.json()["batch_id"]
        assert batch1_id.startswith("BATCH-")

        readings = []
        base_ts = 2_000_000_000.0
        for t in range(10):
            for row in range(16):
                for col in range(24):
                    readings.append({
                        "sensor_id": f"TC_R{row:02d}_C{col:02d}",
                        "temperature": 1200.0,
                        "timestamp": base_ts + t * 5.0,
                    })
        client.post("/api/v1/ingest", json={"readings": readings})

        switch_resp = client.post("/api/v1/batch/switch")
        assert switch_resp.status_code == 200
        switch_data = switch_resp.json()
        assert switch_data["old_batch_id"] == batch1_id
        assert switch_data["new_batch_id"] != batch1_id
        assert switch_data["gc_triggered"] is True
        assert switch_data["old_reading_count"] >= 10 * 16 * 24
        assert switch_data["old_inference_count"] >= 0

        batch2_resp = client.get("/api/v1/batch/current")
        assert batch2_resp.status_code == 200
        batch2_id = batch2_resp.json()["batch_id"]
        assert batch2_id == switch_data["new_batch_id"]
        assert batch2_resp.json()["readings_count"] == 0

        grid_resp = client.get("/api/v1/grid")
        assert grid_resp.status_code == 200
        grid_data = grid_resp.json()
        assert grid_data["time_steps_filled"] == 0
        assert grid_data["is_ready"] is False

        engine_resp = client.get("/api/v1/engine/status")
        assert engine_resp.status_code == 200
        engine_data = engine_resp.json()
        assert engine_data["current_batch_id"] == batch2_id
        assert engine_data["batches_completed"] == 1
        assert engine_data["lifecycle_state"] == "ACTIVE"


@pytest.mark.asyncio
async def test_fastapi_force_gc_endpoint():
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        gc_resp = client.post("/api/v1/system/force_gc")
        assert gc_resp.status_code == 200
        gc_data = gc_resp.json()
        assert gc_data["triggered"] is True
        assert gc_data["gc_count_after"] > gc_data["gc_count_before"]
        assert gc_data["last_gc_ts"] > 0
