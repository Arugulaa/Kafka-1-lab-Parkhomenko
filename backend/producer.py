import os
import time
import json
import pandas as pd
from kafka import KafkaProducer

# абсолютный путь к папке проекта
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset_combined.csv")

# название топика в Kafka
# топик — это как очередь сообщений с именем
# Producer пишет в топик, Consumer читает из топика
TOPIC = "stock_raw"

# адрес Kafka брокера
KAFKA_SERVER = "localhost:9092"

# задержка между отправкой сообщений (в секундах)
# имитируем реальный поток данных
SLEEP_TIME = 0.01


def create_producer():
    # создаём объект Producer
    # value_serializer — функция которая конвертирует Python dict в bytes
    # Kafka работает только с байтами, не с Python объектами
    # json.dumps конвертирует dict в строку, encode конвертирует строку в bytes
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_SERVER,
        value_serializer=lambda x: json.dumps(x).encode("utf-8")
    )
    return producer


def send_data(producer, df):
    print(f"Начинаем отправку {len(df)} сообщений в топик '{TOPIC}'...")
    print(f"Задержка между сообщениями: {SLEEP_TIME} сек")
    print("Нажми Ctrl+C чтобы остановить\n")

    sent_count = 0

    for index, row in df.iterrows():
        # конвертируем строку датафрейма в словарь
        # это будет одно сообщение в Kafka
        message = row.to_dict()

        # конвертируем Timestamp в строку
        # json не умеет сериализовать pandas Timestamp
        message["Date"] = str(message["Date"])

        # отправляем сообщение в топик
        # send() — асинхронная отправка (не ждёт подтверждения)
        producer.send(TOPIC, value=message)

        sent_count += 1

        # каждые 100 сообщений печатаем прогресс
        if sent_count % 100 == 0:
            print(f"Отправлено: {sent_count} сообщений | "
                  f"Последний тикер: {message.get('ticker')} | "
                  f"Дата: {message.get('Date')}")

        # имитируем реальный поток данных
        time.sleep(SLEEP_TIME)

    # flush — ждём пока все сообщения действительно отправятся
    # без flush некоторые сообщения могут не дойти до Kafka
    producer.flush()
    print(f"\nГотово! Отправлено {sent_count} сообщений")


if __name__ == "__main__":
    # читаем датасет
    print(f"Читаем датасет: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    print(f"Загружено строк: {len(df)}")
    print(f"Компаний: {df['ticker'].nunique()}")

    # для теста берём только первые 1000 строк
    # когда убедимся что всё работает — уберём это ограничение
    df_test = df.head(50000)
    print(f"Отправляем первые {len(df_test)} строк для теста\n")

    # создаём Producer и отправляем данные
    producer = create_producer()
    send_data(producer, df_test)