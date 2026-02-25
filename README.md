# Kafka-1-lab-Parkhomenko
Kafka + ML Pipeline for Stock Market
# Kafka ML Pipeline — Stock Market

Система потоковой обработки данных фондового рынка с ML предсказаниями.

## Стек
- Apache Kafka (Docker, bitnami/kafka)
- Python 3.13
- scikit-learn (Random Forest)
- Streamlit

## Структура
- backend/producer.py — отправка данных в Kafka
- backend/consumer_ml.py — ML предсказания
- backend/consumer_processor.py — статистика
- frontend/dashboard.py — Streamlit Dashboard
- notebooks/ — EDA и предобработка данных

## Запуск

### 1. Запустить Kafka
docker-compose up -d

### 2. Обучить модель
python backend/train_model.py

### 3. Запустить сервисы
python backend/consumer_ml.py
python backend/consumer_processor.py
python backend/producer.py
streamlit run frontend/dashboard.py

## Датасет
Stock Market Dataset с Kaggle:
https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset
```

---

## Шаг 4: Установи Git

Проверь установлен ли Git:
```
git --version
```

Если не установлен — скачай с git-scm.com и установи.

---

## Шаг 5: Загрузи проект

Выполни команды по одной в терминале из папки проекта:
```
git init
```
```
git add .
```
```
git commit -m "Kafka ML Pipeline: Producer, Consumer, Dashboard"
```
```
git branch -M main
```
```
git remote add origin https://github.com/ТУТ_ТВОЙ_ЛОГИН/kafka_ml_proj.git
```
```
git push -u origin main
