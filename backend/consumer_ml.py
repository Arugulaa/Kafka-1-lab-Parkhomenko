import os
import json
import joblib
import numpy as np
import pandas as pd
from kafka import KafkaConsumer, KafkaProducer
import warnings
warnings.filterwarnings("ignore")

# абсолютный путь к папке проекта
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

# топик из которого читаем сырые данные
TOPIC_IN  = "stock_raw"

# топик в который пишем результаты предсказаний
TOPIC_OUT = "stock_predictions"

# адрес Kafka брокера
KAFKA_SERVER = "localhost:9092"

# признаки — должны совпадать с теми что использовались при обучении
FEATURES = ["Adj Close", "Volume", "volatility",
            "MA5_ratio", "MA20_ratio", "MA_cross",
            "price_change", "day_of_week", "month"]


def load_model_and_scaler():
    # загружаем обученную модель и скалер из файлов
    print(f"Загружаем модель из: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    print(f"Загружаем скалер из: {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)

    print("Модель и скалер загружены успешно\n")
    return model, scaler


def create_consumer():
    # создаём Consumer который читает из топика stock_raw
    # value_deserializer — обратная операция к сериализатору в Producer
    # конвертируем bytes -> строку -> Python dict
    consumer = KafkaConsumer(
        TOPIC_IN,
        bootstrap_servers=KAFKA_SERVER,
        value_deserializer=lambda x: json.loads(x.decode("utf-8")),
        # auto_offset_reset — откуда начинать читать
        # earliest = с самого первого сообщения в топике
        auto_offset_reset="earliest",
        # group_id — идентификатор группы Consumer'ов
        # Kafka помнит какие сообщения эта группа уже читала
        group_id="ml_consumer_group"
    )
    return consumer


def create_producer():
    # Producer для отправки результатов предсказаний
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_SERVER,
        value_serializer=lambda x: json.dumps(x).encode("utf-8")
    )
    return producer


def predict(message, model, scaler):
    # проверяем есть ли все нужные признаки в сообщении
    for feature in FEATURES:
        if feature not in message:
            return None

    # извлекаем значения признаков в правильном порядке
    # важно что порядок совпадает с порядком при обучении
    feature_values = [message[feature] for feature in FEATURES]

    # конвертируем в numpy array нужной формы
    # reshape(1, -1) = одна строка, количество колонок автоматически
    # X = np.array(feature_values).reshape(1, -1)
    X = pd.DataFrame([feature_values], columns=FEATURES)
    # делаем предсказание
    # predict возвращает класс (0 или 1)
    # predict_proba возвращает вероятности для каждого класса
    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][prediction])

    return prediction, probability


def run_consumer():
    model, scaler = load_model_and_scaler()
    consumer = create_consumer()
    producer = create_producer()

    print(f"Consumer запущен. Читаем из топика '{TOPIC_IN}'...", flush=True)
    print(f"Результаты отправляем в топик '{TOPIC_OUT}'", flush=True)
    print("Нажми Ctrl+C чтобы остановить\n", flush=True)

    processed_count = 0

    # бесконечный цикл — Consumer всегда ждёт новых сообщений
    for message in consumer:
        # message.value — это Python dict (уже десериализован)
        data = message.value

        # делаем предсказание
        result = predict(data, model, scaler)

        if result is None:
            continue

        prediction, probability = result
        processed_count += 1

        # формируем результат для отправки в топик predictions
        output = {
            "ticker":      data.get("ticker"),
            "date":        data.get("Date"),
            "adj_close":   data.get("Adj Close"),
            "volume":      data.get("Volume"),
            "prediction":  prediction,
            # prediction=1 значит модель думает что цена вырастет
            "signal":      "UP" if prediction == 1 else "DOWN",
            "probability": round(probability, 4)
        }

        # отправляем результат в топик stock_predictions
        producer.send(TOPIC_OUT, value=output)

        # каждые 100 сообщений печатаем прогресс
        if processed_count % 100 == 0:
            print(f"Обработано: {processed_count} | "
                  f"Тикер: {output['ticker']} | "
                  f"Дата: {output['date']} | "
                  f"Сигнал: {output['signal']} | "
                  f"Вероятность: {output['probability']}", flush=True)


if __name__ == "__main__":
    run_consumer()