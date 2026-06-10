from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="STEEL_",
        protected_namespaces=("settings_",),
    )

    kafka_brokers: list[str] = ["localhost:9092"]
    kafka_topic: str = "steel_thermocouple_temperatures"
    kafka_group_id: str = "breakout_prediction_group"

    sensor_layout_path: str = "data/sensor_layout.json"

    grid_height: int = 16
    grid_width: int = 24
    time_window: int = 30

    model_path: str = "data/cnn_lstm_breakout_model.pt"

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    inference_interval: float = 1.0

    min_temp: float = 800.0
    max_temp: float = 1600.0


settings = Settings()
