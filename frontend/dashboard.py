import os
import json
import warnings
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from kafka import KafkaConsumer
from collections import deque
import uuid

warnings.filterwarnings("ignore")

# адрес Kafka брокера
KAFKA_SERVER = "localhost:9092"

# топик из которого читаем предсказания
TOPIC = "stock_predictions"

# максимальное количество точек на графике
# deque автоматически удаляет старые данные когда достигает maxlen
MAX_POINTS = 200

# настройка страницы Streamlit
st.set_page_config(
    page_title="Stock ML Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Market ML Dashboard")
st.caption("Предсказания направления цены акций в реальном времени")


# def create_consumer():
#     # подключаемся к Kafka и читаем топик stock_predictions
#     consumer = KafkaConsumer(
#         TOPIC,
#         bootstrap_servers=KAFKA_SERVER,
#         value_deserializer=lambda x: json.loads(x.decode("utf-8")),
#         auto_offset_reset="earliest",
#         # group_id="dashboard_group",
#         # consumer_timeout_ms — сколько миллисекунд ждать новых сообщений
#         # после этого времени цикл заканчивается и страница обновляется
#         group_id=f"dashboard_{uuid.uuid4()}",
#         consumer_timeout_ms=2000
#     )
#     return consumer
def create_consumer():
    consumer = KafkaConsumer(
        bootstrap_servers=KAFKA_SERVER,
        value_deserializer=lambda x: json.loads(x.decode("utf-8")),
        # без group_id — Kafka не запоминает позицию
        # каждый раз читаем всё с начала
        auto_offset_reset="earliest",
        consumer_timeout_ms=3000
    )
    # подписываемся на топик отдельно
    consumer.assign([__import__('kafka').TopicPartition(TOPIC, 0)])
    # явно перемещаемся в начало топика
    consumer.seek_to_beginning()
    return consumer

def load_messages():
    # читаем все доступные сообщения из Kafka
    messages = []
    try:
        consumer = create_consumer()
        for message in consumer:
            messages.append(message.value)
            # читаем максимум 5000 сообщений
            # иначе браузер зависает при большом объёме
            if len(messages) >= 5000:
                break
        consumer.close()
    except Exception as e:
        st.error(f"Ошибка подключения к Kafka: {e}")
    return messages


# загружаем сообщения из Kafka
messages = load_messages()

if not messages:
    st.warning("Нет данных. Запусти Producer и Consumer:")
    st.code("python backend/producer.py")
    st.code("python backend/consumer_ml.py")
    st.stop()

# конвертируем список сообщений в DataFrame
df = pd.DataFrame(messages)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

st.success(f"Загружено {len(df)} предсказаний из Kafka")

# метрики вверху страницы
col1, col2, col3, col4 = st.columns(4)

total = len(df)
up_count   = (df["signal"] == "UP").sum()
down_count = (df["signal"] == "DOWN").sum()
avg_prob   = df["probability"].mean()

col1.metric("Всего предсказаний", total)
col2.metric("Сигналов UP 📈",   up_count,   f"{up_count/total*100:.1f}%")
col3.metric("Сигналов DOWN 📉", down_count, f"{down_count/total*100:.1f}%")
col4.metric("Средняя уверенность", f"{avg_prob:.3f}")

st.divider()

# график 1: цена и сигналы для выбранной компании
st.subheader("График 1: Цена акции и сигналы модели")

# выбор компании через выпадающий список
tickers = sorted(df["ticker"].unique())
selected_ticker = st.selectbox("Выбери компанию:", tickers)

# фильтруем данные по выбранной компании
df_ticker = df[df["ticker"] == selected_ticker].copy()

# создаём график с помощью plotly
fig1 = go.Figure()

# линия цены
fig1.add_trace(go.Scatter(
    x=df_ticker["date"],
    y=df_ticker["adj_close"],
    mode="lines",
    name="Adj Close",
    line=dict(color="steelblue", width=1)
))

# зелёные точки — сигнал UP
df_up = df_ticker[df_ticker["signal"] == "UP"]
fig1.add_trace(go.Scatter(
    x=df_up["date"],
    y=df_up["adj_close"],
    mode="markers",
    name="UP (вырастет)",
    marker=dict(color="green", size=6, symbol="triangle-up")
))

# красные точки — сигнал DOWN
df_down = df_ticker[df_ticker["signal"] == "DOWN"]
fig1.add_trace(go.Scatter(
    x=df_down["date"],
    y=df_down["adj_close"],
    mode="markers",
    name="DOWN (упадёт)",
    marker=dict(color="red", size=6, symbol="triangle-down")
))

fig1.update_layout(
    title=f"{selected_ticker} — цена и сигналы модели",
    xaxis_title="Дата",
    yaxis_title="Adj Close",
    height=400
)
st.plotly_chart(fig1, use_container_width=True)

st.divider()

# график 2 и 3 рядом
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("График 2: Распределение сигналов")

    signal_counts = df["signal"].value_counts()
    fig2 = go.Figure(go.Pie(
        labels=signal_counts.index,
        values=signal_counts.values,
        marker_colors=["green", "red"],
        hole=0.3
    ))
    fig2.update_layout(height=350)
    st.plotly_chart(fig2, use_container_width=True)

with col_right:
    st.subheader("График 3: Уверенность модели")

    fig3 = go.Figure(go.Histogram(
        x=df["probability"],
        nbinsx=30,
        marker_color="steelblue"
    ))
    fig3.update_layout(
        xaxis_title="Вероятность",
        yaxis_title="Количество",
        height=350
    )
    st.plotly_chart(fig3, use_container_width=True)

st.divider()

# график 4: сигналы по компаниям
st.subheader("График 4: Количество сигналов UP по компаниям (топ-10)")

up_by_ticker = df[df["signal"] == "UP"].groupby("ticker").size().sort_values(ascending=True).tail(10)

fig4 = go.Figure(go.Bar(
    x=up_by_ticker.values,
    y=up_by_ticker.index,
    orientation="h",
    marker_color="green"
))
fig4.update_layout(
    xaxis_title="Количество сигналов UP",
    yaxis_title="Компания",
    height=400
)
st.plotly_chart(fig4, use_container_width=True)

st.divider()

# таблица последних предсказаний
st.subheader("Последние 20 предсказаний")

df_display = df.tail(20)[["date", "ticker", "adj_close", "signal", "probability"]].copy()
df_display = df_display.sort_values("date", ascending=False)
df_display.columns = ["Дата", "Тикер", "Цена", "Сигнал", "Вероятность"]

st.dataframe(df_display, use_container_width=True)

# кнопка обновления
st.divider()
if st.button("🔄 Обновить данные"):
    st.rerun()