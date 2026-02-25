import os
import json
import warnings
from kafka import KafkaConsumer, KafkaProducer
from collections import defaultdict

warnings.filterwarnings("ignore")

# адрес Kafka брокера
KAFKA_SERVER = "localhost:9092"

# читаем сырые данные из того же топика что и consumer_ml
TOPIC_IN  = "stock_raw"

# пишем обработанную статистику в новый топик
TOPIC_OUT = "stock_processed"


class DataProcessorConsumer:
    # Consumer оформлен как класс — это требование преподавателя
    # класс удобнее функций: хранит состояние (статистику) внутри себя

    def __init__(self):
        self.consumer = self.create_consumer()
        self.producer = self.create_producer()

        # словари для накопления статистики по каждому тикеру
        # defaultdict автоматически создаёт пустой список для нового ключа
        self.prices  = defaultdict(list)
        self.volumes = defaultdict(list)

        # счётчик обработанных сообщений
        self.processed_count = 0

    def create_consumer(self):
        consumer = KafkaConsumer(
            TOPIC_IN,
            bootstrap_servers=KAFKA_SERVER,
            value_deserializer=lambda x: json.loads(x.decode("utf-8")),
            auto_offset_reset="earliest",
            # отдельный group_id — не конкурирует с consumer_ml
            # каждая группа читает топик независимо
            group_id="processor_consumer_group"
        )
        return consumer

    def create_producer(self):
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_SERVER,
            value_serializer=lambda x: json.dumps(x).encode("utf-8")
        )
        return producer

    def process_message(self, data):
        ticker = data.get("ticker")
        if not ticker:
            return None

        # накапливаем цены и объёмы по каждому тикеру
        price  = data.get("Adj Close")
        volume = data.get("Volume")

        if price is not None:
            self.prices[ticker].append(price)
        if volume is not None:
            self.volumes[ticker].append(volume)

        # считаем статистику только если накопили хотя бы 5 значений
        if len(self.prices[ticker]) < 5:
            return None

        prices_list  = self.prices[ticker]
        volumes_list = self.volumes[ticker]

        # средняя цена за все накопленные данные
        avg_price = sum(prices_list) / len(prices_list)

        # минимальная и максимальная цена
        min_price = min(prices_list)
        max_price = max(prices_list)

        # средний объём
        avg_volume = sum(volumes_list) / len(volumes_list) if volumes_list else 0

        # тренд — сравниваем последнюю цену со средней
        # > 1.0 = цена выше среднего = восходящий тренд
        trend = prices_list[-1] / avg_price if avg_price > 0 else 1.0

        # формируем обработанное сообщение
        result = {
            "ticker":      ticker,
            "date":        data.get("Date"),
            "avg_price":   round(avg_price, 6),
            "min_price":   round(min_price, 6),
            "max_price":   round(max_price, 6),
            "avg_volume":  round(avg_volume, 2),
            "trend":       round(trend, 4),
            # интерпретация тренда
            "trend_label": "UP" if trend > 1.0 else "DOWN",
            "data_points": len(prices_list)
        }
        return result

    def run(self):
        print(f"DataProcessorConsumer запущен")
        print(f"Читаем из топика: '{TOPIC_IN}'")
        print(f"Результаты пишем в топик: '{TOPIC_OUT}'")
        print("Нажми Ctrl+C чтобы остановить\n")

        for message in self.consumer:
            data = message.value

            result = self.process_message(data)

            if result is None:
                continue

            # отправляем статистику в топик stock_processed
            self.producer.send(TOPIC_OUT, value=result)
            self.processed_count += 1

            if self.processed_count % 100 == 0:
                print(f"Обработано: {self.processed_count} | "
                      f"Тикер: {result['ticker']} | "
                      f"Средняя цена: {result['avg_price']} | "
                      f"Тренд: {result['trend_label']}")


if __name__ == "__main__":
    processor = DataProcessorConsumer()
    processor.run()