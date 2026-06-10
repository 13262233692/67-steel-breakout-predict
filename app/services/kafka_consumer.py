import asyncio
import json
import time
import random
import logging
from typing import Callable, Optional, List
from dataclasses import dataclass

from app.config import settings
from app.schemas.prediction import ThermocoupleReading
from app.utils.sensor_mapping import SensorMapper

logger = logging.getLogger(__name__)


@dataclass
class KafkaConsumerStatus:
    connected: bool = False
    running: bool = False
    messages_consumed: int = 0


class ThermocoupleKafkaConsumer:
    def __init__(
        self,
        sensor_mapper: SensorMapper,
        on_readings_callback: Callable[[List[ThermocoupleReading]], None],
        brokers: Optional[List[str]] = None,
        topic: Optional[str] = None,
        group_id: Optional[str] = None,
    ):
        self.sensor_mapper = sensor_mapper
        self.callback = on_readings_callback
        self.brokers = brokers or settings.kafka_brokers
        self.topic = topic or settings.kafka_topic
        self.group_id = group_id or settings.kafka_group_id
        self.status = KafkaConsumerStatus()
        self._consumer = None
        self._task: Optional[asyncio.Task] = None
        self._simulate_mode: bool = False

    async def _connect_kafka(self) -> bool:
        try:
            from aiokafka import AIOKafkaConsumer
            self._consumer = AIOKafkaConsumer(
                self.topic,
                bootstrap_servers=",".join(self.brokers),
                group_id=self.group_id,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            )
            await self._consumer.start()
            self.status.connected = True
            logger.info("Kafka消费者已连接到 %s, topic: %s", self.brokers, self.topic)
            return True
        except Exception as e:
            logger.warning("Kafka连接失败，切换到模拟数据模式: %s", e)
            self._simulate_mode = True
            self.status.connected = True
            return True

    def _parse_message(self, message_value) -> Optional[List[ThermocoupleReading]]:
        try:
            if isinstance(message_value, dict):
                if "readings" in message_value:
                    readings = [ThermocoupleReading(**r) for r in message_value["readings"]]
                else:
                    readings = [ThermocoupleReading(**message_value)]
            elif isinstance(message_value, list):
                readings = [ThermocoupleReading(**r) for r in message_value]
            else:
                return None
            return readings
        except Exception as e:
            logger.warning("解析Kafka消息失败: %s", e)
            return None

    async def _consume_loop(self) -> None:
        self.status.running = True
        try:
            if not self._simulate_mode and self._consumer is not None:
                async for msg in self._consumer:
                    readings = self._parse_message(msg.value)
                    if readings:
                        self.callback(readings)
                        self.status.messages_consumed += len(readings)
            else:
                await self._simulate_data_loop()
        except asyncio.CancelledError:
            logger.info("Kafka消费者任务被取消")
        except Exception as e:
            logger.error("Kafka消费循环异常: %s", e)
        finally:
            self.status.running = False

    def _generate_simulated_reading(self, sensor_id: str, ts: float, anomaly: bool = False) -> ThermocoupleReading:
        base_temp = 1200.0
        if anomaly:
            temp = base_temp + random.uniform(50.0, 150.0)
        else:
            temp = base_temp + random.uniform(-20.0, 20.0)
        return ThermocoupleReading(sensor_id=sensor_id, temperature=temp, timestamp=ts)

    async def _simulate_data_loop(self) -> None:
        logger.info("启动模拟数据模式")
        all_sensor_ids = self.sensor_mapper.all_sensor_ids()
        step = 0
        while True:
            ts = time.time()
            readings = []
            anomaly_center_row = random.randint(5, 10) if step % 100 < 20 else -1
            anomaly_center_col = random.randint(8, 16) if step % 100 < 20 else -1
            for sensor_id in all_sensor_ids:
                coords = self.sensor_mapper.get_coordinates(sensor_id)
                is_anomaly = False
                if coords and anomaly_center_row >= 0 and anomaly_center_col >= 0:
                    row, col = coords
                    distance = abs(row - anomaly_center_row) + abs(col - anomaly_center_col)
                    if distance <= 3:
                        is_anomaly = True
                readings.append(self._generate_simulated_reading(sensor_id, ts, is_anomaly))
            self.callback(readings)
            self.status.messages_consumed += len(readings)
            step += 1
            await asyncio.sleep(0.5)

    async def start(self) -> None:
        await self._connect_kafka()
        self._task = asyncio.create_task(self._consume_loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._consumer is not None:
            try:
                await self._consumer.stop()
            except Exception:
                pass
            self._consumer = None
        self.status.connected = False
        self.status.running = False
        logger.info("Kafka消费者已停止")
