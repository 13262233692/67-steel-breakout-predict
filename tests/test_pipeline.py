import time
import pytest
import numpy as np

from app.config import settings
from app.utils.sensor_mapping import SensorMapper
from app.services.spatiotemporal_grid import SpatiotemporalGrid
from app.services.inference_engine import BreakoutInferenceEngine
from app.models.cnn_lstm import CNNLSTMModel, load_model
from app.schemas.prediction import ThermocoupleReading


def test_sensor_mapper():
    mapper = SensorMapper(settings.sensor_layout_path)

    assert mapper.total_sensors() == 384

    h, w = mapper.get_grid_dimensions()
    assert h == 16
    assert w == 24

    coords = mapper.get_coordinates("TC_R00_C00")
    assert coords == (0, 0)

    coords = mapper.get_coordinates("TC_R05_C12")
    assert coords == (5, 12)

    coords = mapper.get_coordinates("TC_R15_C23")
    assert coords == (15, 23)

    assert mapper.get_coordinates("INVALID_ID") is None

    sensor_id = mapper.get_sensor_id(3, 7)
    assert sensor_id == "TC_R03_C07"

    info = mapper.get_sensor_info("TC_R02_C05")
    assert info is not None
    assert info.row == 2
    assert info.col == 5
    assert info.physical_x_mm == 250.0
    assert info.physical_y_mm == 100.0

    all_ids = mapper.all_sensor_ids()
    assert len(all_ids) == 384
    assert "TC_R00_C00" in all_ids
    assert "TC_R15_C23" in all_ids


def test_spatiotemporal_grid_basic():
    mapper = SensorMapper(settings.sensor_layout_path)
    grid = SpatiotemporalGrid(mapper, time_window=5)

    assert grid.get_shape() == (5, 16, 24)
    assert not grid.is_ready()
    assert grid.time_steps_filled() == 0

    base_ts = time.time()
    for t in range(7):
        for row in range(16):
            for col in range(24):
                sensor_id = f"TC_R{row:02d}_C{col:02d}"
                temp = 1200.0 + row * 2.0 + col * 1.0
                reading = ThermocoupleReading(
                    sensor_id=sensor_id,
                    temperature=temp,
                    timestamp=base_ts + t * 1.0,
                )
                grid.ingest_reading(reading)

    assert grid.time_steps_filled() >= 5
    assert grid.is_ready()

    tensor = grid.get_tensor()
    assert tensor is not None
    assert tensor.shape == (5, 16, 24)
    assert tensor.dtype == np.float32
    assert np.min(tensor) >= 0.0
    assert np.max(tensor) <= 1.0

    batch_tensor = grid.get_tensor_with_batch()
    assert batch_tensor is not None
    assert batch_tensor.shape == (1, 5, 16, 24)

    assert grid.total_readings_processed == 7 * 16 * 24


def test_spatiotemporal_grid_normalization():
    mapper = SensorMapper(settings.sensor_layout_path)
    grid = SpatiotemporalGrid(
        mapper, time_window=2, min_temp=800.0, max_temp=1600.0
    )

    base_ts = time.time()
    for t in range(3):
        sensor_id = "TC_R00_C00"
        reading = ThermocoupleReading(
            sensor_id=sensor_id, temperature=1200.0, timestamp=base_ts + t * 1.0
        )
        grid.ingest_reading(reading)

    tensor = grid.get_tensor()
    assert tensor is not None
    expected_norm = (1200.0 - 800.0) / (1600.0 - 800.0)
    assert abs(tensor[0, 0, 0] - expected_norm) < 0.01


def test_cnn_lstm_model_forward():
    model = CNNLSTMModel(time_steps=30, grid_height=16, grid_width=24)
    model.eval()

    batch_size = 2
    dummy_input = np.random.rand(batch_size, 30, 16, 24).astype(np.float32)

    import torch
    tensor = torch.from_numpy(dummy_input)

    with torch.no_grad():
        logits = model(tensor)
        assert logits.shape == (batch_size, 1)

        probs = model.predict_proba(tensor)
        assert probs.shape == (batch_size, 1)
        assert torch.all(probs >= 0.0)
        assert torch.all(probs <= 1.0)


def test_cnn_lstm_load_model():
    model = load_model(
        model_path=None, time_steps=5, grid_height=8, grid_width=12
    )
    assert model is not None
    assert model.time_steps == 5
    assert model.grid_height == 8
    assert model.grid_width == 12

    import torch
    dummy = torch.randn(1, 5, 8, 12)
    with torch.no_grad():
        out = model.predict_proba(dummy)
    assert out.shape == (1, 1)


def test_inference_engine_full_pipeline():
    mapper = SensorMapper(settings.sensor_layout_path)
    grid = SpatiotemporalGrid(mapper, time_window=5)

    engine = BreakoutInferenceEngine(grid, interval=0.1)
    engine.load_model()
    assert engine.status.model_loaded is True

    base_ts = time.time()
    for t in range(10):
        for row in range(16):
            for col in range(24):
                sensor_id = f"TC_R{row:02d}_C{col:02d}"
                temp = 1100.0 + np.random.uniform(-30, 30)
                reading = ThermocoupleReading(
                    sensor_id=sensor_id,
                    temperature=temp,
                    timestamp=base_ts + t * 1.0,
                )
                grid.ingest_reading(reading)

    assert grid.is_ready()

    result = engine.run_inference()
    assert result is not None
    assert 0.0 <= result.breakout_probability <= 1.0
    assert result.is_alert == (result.breakout_probability >= 0.7)
    assert engine.status.total_inferences == 1
    assert engine.status.latest_prediction is not None


@pytest.mark.asyncio
async def test_fastapi_endpoints():
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "timestamp" in data

        response = client.get("/api/v1/status")
        assert response.status_code == 200
        data = response.json()
        assert "kafka_connected" in data
        assert "model_loaded" in data
        assert "grid_ready" in data

        response = client.get("/api/v1/grid")
        assert response.status_code == 200
        data = response.json()
        assert data["shape"] == [30, 16, 24]
        assert "time_steps_filled" in data
        assert "is_ready" in data

        readings = []
        base_ts = time.time()
        for row in range(16):
            for col in range(24):
                readings.append({
                    "sensor_id": f"TC_R{row:02d}_C{col:02d}",
                    "temperature": 1200.0,
                    "timestamp": base_ts,
                })
        response = client.post("/api/v1/ingest", json={"readings": readings})
        assert response.status_code == 200
        data = response.json()
        assert data["ingested"] == 384
