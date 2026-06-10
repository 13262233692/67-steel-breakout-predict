import time
import pytest
import numpy as np
import torch

from app.config import settings
from app.utils.sensor_mapping import SensorMapper
from app.services.spatiotemporal_grid import SpatiotemporalGrid
from app.services.inference_engine import BreakoutInferenceEngine
from app.services.gradcam_explainer import GradCAMExplainer
from app.models.cnn_lstm import CNNLSTMModel, load_model
from app.schemas.prediction import (
    ThermocoupleReading,
    BreakoutExplanation,
    GradCAMHeatmap,
    AnomalySensor,
)


def _fill_grid(
    grid: SpatiotemporalGrid,
    time_steps: int,
    base_temp: float = 1200.0,
    base_ts: float = None,
    anomaly_region: tuple[int, int, int, int] | None = None,
    anomaly_temp: float = 1450.0,
):
    if base_ts is None:
        base_ts = 1_700_000_000.0
    base_ts = float(int(base_ts))
    for t in range(time_steps):
        step_ts = base_ts + float(t) * 5.0
        for row in range(grid.grid_height):
            for col in range(grid.grid_width):
                sensor_id = f"TC_R{row:02d}_C{col:02d}"
                temp = base_temp + np.random.uniform(-10, 10)
                if anomaly_region is not None:
                    r1, r2, c1, c2 = anomaly_region
                    if r1 <= row <= r2 and c1 <= col <= c2:
                        temp = anomaly_temp + np.random.uniform(-5, 5)
                reading = ThermocoupleReading(
                    sensor_id=sensor_id,
                    temperature=temp,
                    timestamp=step_ts,
                )
                grid.ingest_reading(reading)
    grid.flush()


class TestGradCAMModelCore:
    def test_gradcam_enable_disable(self):
        model = CNNLSTMModel(time_steps=5, grid_height=8, grid_width=12)
        model.eval()

        assert model._gradcam_enabled is False
        model.enable_gradcam()
        assert model._gradcam_enabled is True
        assert len(model._gradcam_hooks) == 2

        model.enable_gradcam()
        assert len(model._gradcam_hooks) == 2

        model.disable_gradcam()
        assert model._gradcam_enabled is False
        assert len(model._gradcam_hooks) == 0
        assert model._activations is None
        assert model._gradients is None

    def test_compute_gradcam_returns_correct_shape(self):
        time_steps = 5
        H, W = 8, 12
        model = CNNLSTMModel(time_steps=time_steps, grid_height=H, grid_width=W)
        model.eval()
        model.enable_gradcam()

        x = torch.randn(1, time_steps, H, W)
        prob, cam = model.compute_gradcam(x, target_class=0, upsample_to_input=True)

        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0
        assert isinstance(cam, np.ndarray)
        assert cam.shape == (H, W)
        assert cam.dtype == np.float32
        assert np.all(cam >= 0.0) and np.all(cam <= 1.0)
        cam_max = cam.max()
        assert cam_max >= 0.0

    def test_compute_gradcam_no_upsample(self):
        time_steps = 5
        H, W = 16, 24
        model = CNNLSTMModel(time_steps=time_steps, grid_height=H, grid_width=W)
        model.eval()
        model.enable_gradcam()

        x = torch.randn(1, time_steps, H, W)
        prob, cam = model.compute_gradcam(x, target_class=0, upsample_to_input=False)

        assert cam.shape == (H // 4, W // 4)

    def test_gradcam_memory_cleanup(self):
        import gc
        import weakref

        time_steps = 5
        H, W = 8, 12
        model = CNNLSTMModel(time_steps=time_steps, grid_height=H, grid_width=W)
        model.eval()
        model.enable_gradcam()

        x = torch.randn(1, time_steps, H, W)
        x_ref = weakref.ref(x)

        prob, cam = model.compute_gradcam(x, target_class=0, upsample_to_input=True)

        del x
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        assert x_ref() is None
        assert isinstance(prob, float)
        assert isinstance(cam, np.ndarray)

    def test_gradcam_with_batch_size_gt_1_fails_gracefully(self):
        time_steps = 5
        H, W = 8, 12
        model = CNNLSTMModel(time_steps=time_steps, grid_height=H, grid_width=W)
        model.eval()
        model.enable_gradcam()

        x = torch.randn(2, time_steps, H, W)
        prob, cam = model.compute_gradcam(x, target_class=0, upsample_to_input=True)

        assert isinstance(prob, float)
        assert cam.shape == (H, W)


class TestGradCAMExplainer:
    def test_explainer_initialization(self):
        sensor_mapper = SensorMapper(settings.sensor_layout_path)
        model = CNNLSTMModel(
            time_steps=5,
            grid_height=sensor_mapper.get_grid_dimensions()[0],
            grid_width=sensor_mapper.get_grid_dimensions()[1],
        )
        model.eval()

        explainer = GradCAMExplainer(
            model=model,
            sensor_mapper=sensor_mapper,
            alert_threshold=0.85,
            top_k_sensors=10,
        )

        assert explainer.alert_threshold == 0.85
        assert explainer.top_k_sensors == 10
        assert explainer._gradcam_enabled is True
        assert model._gradcam_enabled is True

    def test_explainer_heatmap_to_sensor_mapping(self):
        sensor_mapper = SensorMapper(settings.sensor_layout_path)
        H, W = sensor_mapper.get_grid_dimensions()
        model = CNNLSTMModel(time_steps=5, grid_height=H, grid_width=W)
        model.eval()

        explainer = GradCAMExplainer(model, sensor_mapper, top_k_sensors=5)

        heatmap = np.zeros((H, W), dtype=np.float32)
        heatmap[3:6, 8:12] = 1.0

        current_grid = np.ones((H, W), dtype=np.float32) * 0.5
        sensors = explainer._map_heatmap_to_sensors(heatmap, current_grid)

        assert len(sensors) == H * W
        assert sensors[0].attention_weight == 1.0
        assert sensors[0].is_highlight is True
        assert sensors[0].row >= 3 and sensors[0].row <= 5
        assert sensors[0].col >= 8 and sensors[0].col <= 11
        assert sensors[0].current_temperature is not None
        assert 800 <= sensors[0].current_temperature <= 1600
        assert sensors[-1].attention_weight == 0.0
        assert sensors[-1].is_highlight is False

        highlight_count = sum(1 for s in sensors if s.is_highlight)
        assert highlight_count == 5

        first_sensor = sensors[0]
        info = sensor_mapper.get_sensor_info(first_sensor.sensor_id)
        assert info is not None
        assert first_sensor.physical_x_mm == info.physical_x_mm
        assert first_sensor.physical_y_mm == info.physical_y_mm

    def test_explainer_heatmap_serialization(self):
        sensor_mapper = SensorMapper(settings.sensor_layout_path)
        H, W = sensor_mapper.get_grid_dimensions()
        model = CNNLSTMModel(time_steps=5, grid_height=H, grid_width=W)
        model.eval()

        explainer = GradCAMExplainer(model, sensor_mapper)

        heatmap = np.random.rand(H, W).astype(np.float32)
        as_list = explainer._heatmap_to_list(heatmap)

        assert isinstance(as_list, list)
        assert len(as_list) == H
        assert len(as_list[0]) == W
        assert isinstance(as_list[0][0], float)

    def test_explainer_explain_returns_none_for_low_prob(self):
        sensor_mapper = SensorMapper(settings.sensor_layout_path)
        H, W = sensor_mapper.get_grid_dimensions()
        time_steps = 5
        model = CNNLSTMModel(time_steps=time_steps, grid_height=H, grid_width=W)
        model.eval()

        explainer = GradCAMExplainer(model, sensor_mapper, alert_threshold=0.999)

        x = torch.randn(1, time_steps, H, W)
        explanation = explainer.explain(x, force=False)

        assert explanation is None

    def test_explainer_explain_force_returns_explanation(self):
        sensor_mapper = SensorMapper(settings.sensor_layout_path)
        H, W = sensor_mapper.get_grid_dimensions()
        time_steps = 5
        model = CNNLSTMModel(time_steps=time_steps, grid_height=H, grid_width=W)
        model.eval()

        explainer = GradCAMExplainer(model, sensor_mapper, alert_threshold=0.85)

        x = torch.randn(1, time_steps, H, W)
        explanation = explainer.explain(x, force=True)

        assert explanation is not None
        assert isinstance(explanation, BreakoutExplanation)
        assert isinstance(explanation.heatmap, GradCAMHeatmap)
        assert explanation.heatmap.grid_height == H
        assert explanation.heatmap.grid_width == W
        assert len(explanation.heatmap.heatmap_values) == H
        assert len(explanation.heatmap.heatmap_values[0]) == W
        assert len(explanation.anomaly_sensors) == H * W
        assert explanation.gradcam_enabled is True
        assert 0.0 <= explanation.prediction.breakout_probability <= 1.0


class TestGradCAMInferenceEngine:
    def test_inference_engine_with_gradcam_initialization(self):
        sensor_mapper = SensorMapper(settings.sensor_layout_path)
        grid = SpatiotemporalGrid(sensor_mapper)
        engine = BreakoutInferenceEngine(
            grid,
            sensor_mapper,
            gradcam_alert_threshold=0.85,
            gradcam_auto_explain=True,
        )
        engine.load_model()

        assert engine.status.gradcam_enabled is True
        assert engine.status.gradcam_alert_threshold == 0.85
        assert engine.status.gradcam_explanation_count == 0
        assert engine._explainer is not None

    def test_explain_current_returns_explanation(self):
        sensor_mapper = SensorMapper(settings.sensor_layout_path)
        grid = SpatiotemporalGrid(sensor_mapper, time_window=5)
        engine = BreakoutInferenceEngine(grid, sensor_mapper)
        engine.load_model()

        _fill_grid(grid, time_steps=5, base_temp=1200.0, base_ts=1_700_000_000.0)

        explanation = engine.explain_current(force=True)

        assert explanation is not None
        assert isinstance(explanation, BreakoutExplanation)
        assert engine.status.gradcam_explanation_count == 1
        assert engine.status.latest_explanation is not None

    def test_explain_current_returns_none_when_grid_empty(self):
        sensor_mapper = SensorMapper(settings.sensor_layout_path)
        grid = SpatiotemporalGrid(sensor_mapper, time_window=5)
        engine = BreakoutInferenceEngine(grid, sensor_mapper)
        engine.load_model()

        explanation = engine.explain_current(force=True)
        assert explanation is None

    def test_gradcam_heatmap_values_in_valid_range(self):
        sensor_mapper = SensorMapper(settings.sensor_layout_path)
        H, W = sensor_mapper.get_grid_dimensions()
        grid = SpatiotemporalGrid(sensor_mapper, time_window=5)
        engine = BreakoutInferenceEngine(grid, sensor_mapper)
        engine.load_model()

        _fill_grid(grid, time_steps=5, base_temp=1200.0, base_ts=1_700_000_000.0)

        explanation = engine.explain_current(force=True)
        assert explanation is not None

        for row in explanation.heatmap.heatmap_values:
            for val in row:
                assert 0.0 <= val <= 1.0

    def test_anomaly_sensors_have_valid_physical_coordinates(self):
        sensor_mapper = SensorMapper(settings.sensor_layout_path)
        grid = SpatiotemporalGrid(sensor_mapper, time_window=5)
        engine = BreakoutInferenceEngine(grid, sensor_mapper)
        engine.load_model()

        _fill_grid(grid, time_steps=5, base_temp=1200.0, base_ts=1_700_000_000.0)

        explanation = engine.explain_current(force=True)
        assert explanation is not None

        max_x = (sensor_mapper.get_grid_dimensions()[1] - 1) * sensor_mapper.metadata["col_spacing_mm"]
        max_y = (sensor_mapper.get_grid_dimensions()[0] - 1) * sensor_mapper.metadata["row_spacing_mm"]

        for sensor in explanation.anomaly_sensors:
            assert 0.0 <= sensor.physical_x_mm <= max_x
            assert 0.0 <= sensor.physical_y_mm <= max_y
            assert sensor.sensor_id.startswith("TC_R")
            assert "_C" in sensor.sensor_id
            assert 0 <= sensor.row < sensor_mapper.get_grid_dimensions()[0]
            assert 0 <= sensor.col < sensor_mapper.get_grid_dimensions()[1]

    def test_top_k_anomaly_sensors_marked_correctly(self):
        sensor_mapper = SensorMapper(settings.sensor_layout_path)
        grid = SpatiotemporalGrid(sensor_mapper, time_window=5)
        engine = BreakoutInferenceEngine(grid, sensor_mapper)
        engine.load_model()

        _fill_grid(grid, time_steps=5, base_temp=1200.0, base_ts=1_700_000_000.0)

        explanation = engine.explain_current(force=True)
        assert explanation is not None

        highlight_count = sum(1 for s in explanation.anomaly_sensors if s.is_highlight)
        assert highlight_count == 10

        for i, sensor in enumerate(explanation.anomaly_sensors):
            if i < 10:
                assert sensor.is_highlight is True
            else:
                assert sensor.is_highlight is False

        for i in range(len(explanation.anomaly_sensors) - 1):
            assert explanation.anomaly_sensors[i].attention_weight >= \
                   explanation.anomaly_sensors[i + 1].attention_weight


class TestGradCAMFastAPI:
    def test_explain_endpoint_returns_valid_response(self):
        from fastapi.testclient import TestClient
        from app.main import app
        import time as _time

        with TestClient(app) as client:
            base_ts = _time.time() + 100.0
            for i in range(40):
                readings = []
                for row in range(settings.grid_height):
                    for col in range(settings.grid_width):
                        sensor_id = f"TC_R{row:02d}_C{col:02d}"
                        temp = 1200.0 + np.random.uniform(-10, 10)
                        if 5 <= row <= 9 and 10 <= col <= 15:
                            temp = 1450.0 + np.random.uniform(-5, 5)
                        readings.append(
                            {
                                "sensor_id": sensor_id,
                                "temperature": temp,
                                "timestamp": base_ts + i * 5.0,
                            }
                        )
                client.post(
                    "/api/v1/ingest",
                    json={"readings": readings},
                )

            _time.sleep(0.5)
            for _retry in range(10):
                grid_resp = client.get("/api/v1/grid")
                assert grid_resp.status_code == 200
                grid_data = grid_resp.json()
                if grid_data["is_ready"]:
                    break
                _time.sleep(0.2)

            assert grid_data["is_ready"] is True, f"Grid not ready: {grid_data}"

            response = client.post("/api/v1/explain/trigger", params={"force": True})
            assert response.status_code == 200, f"Explain failed: {response.status_code} {response.text}"
            data = response.json()

            assert "prediction" in data
            assert "heatmap" in data
            assert "anomaly_sensors" in data
            assert data["gradcam_enabled"] is True

            assert "heatmap_values" in data["heatmap"]
            assert len(data["heatmap"]["heatmap_values"]) == settings.grid_height
            assert len(data["heatmap"]["heatmap_values"][0]) == settings.grid_width

            assert len(data["anomaly_sensors"]) == settings.grid_height * settings.grid_width

            highlight_count = sum(1 for s in data["anomaly_sensors"] if s["is_highlight"])
            assert highlight_count == 10

            for sensor in data["anomaly_sensors"]:
                assert "sensor_id" in sensor
                assert "row" in sensor
                assert "col" in sensor
                assert "physical_x_mm" in sensor
                assert "physical_y_mm" in sensor
                assert "attention_weight" in sensor
                assert 0.0 <= sensor["attention_weight"] <= 1.0

    def test_get_latest_explanation_endpoint(self):
        from fastapi.testclient import TestClient
        from app.main import app

        with TestClient(app) as client:
            response = client.get("/api/v1/explain")
            assert response.status_code == 200

    def test_engine_status_includes_gradcam_fields(self):
        from fastapi.testclient import TestClient
        from app.main import app

        with TestClient(app) as client:
            response = client.get("/api/v1/engine/status")
            assert response.status_code == 200
            data = response.json()

            assert "gradcam_enabled" in data
            assert "gradcam_explanation_count" in data
            assert "gradcam_alert_threshold" in data
            assert "latest_explanation" in data
