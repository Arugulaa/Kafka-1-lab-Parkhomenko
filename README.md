# Репозиторий Kafka-1-lab-Parkhomenko

Технологии работы с большими массивами данных-2026 - 1 Лабораторная работа
Kafka + ML Pipeline для датасета Stock Market (Система потоковой обработки данных фондового рынка с ML предсказаниями).
Выполнил: Пархоменко Николай, 25МАГ-ИАД

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
(Обученная модель лежит в /models/model.pkl)

### 3. Запустить сервисы
python backend/consumer_ml.py
python backend/consumer_processor.py
python backend/producer.py
streamlit run frontend/dashboard.py



## Датасет
Stock Market Dataset с Kaggle:
https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset

В связи с большим объемом исходных данных, в данной работе используется уменьшенная выборка:
https://drive.google.com/file/d/1nCBkx6k1L6aa1-f_UkPJ64YEI-6f6ZBO/view?usp=drive_link

## Демонстрация работы

<img width="1280" height="424" alt="image" src="https://github.com/user-attachments/assets/3718d2c2-1fcb-4bd8-9f16-ede722c43c40" />

<img width="1280" height="459" alt="image" src="https://github.com/user-attachments/assets/7fdbb308-55c2-4d63-8634-bbd8334c85d5" />

<img width="1280" height="380" alt="image" src="https://github.com/user-attachments/assets/6a6bd14d-23fa-4168-8e16-6eb256ffdf91" />

<img width="1280" height="444" alt="image" src="https://github.com/user-attachments/assets/a88ccd4d-c9dd-4efc-862b-0ec4c0e746cb" />

<img width="1280" height="500" alt="image" src="https://github.com/user-attachments/assets/e0c84ee9-80a7-4690-aeaa-1dfd32d3b74b" />





