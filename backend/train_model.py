# df_recent = df_all[df_all["Date"] >= "2010-01-01"].copy()
#
# print(f"Строк после фильтрации: {len(df_recent)}")
#
# # делим 80/20 по времени
# split_date = df_recent["Date"].quantile(0.8)
# # quantile не работает с датами напрямую — используем iloc
# split_idx = int(len(df_recent) * 0.8)
# df_sorted = df_recent.sort_values("Date")
#
# train = df_sorted.iloc[:split_idx]
# test  = df_sorted.iloc[split_idx:]
#
# # print(f"Train: {len(train)} строк ({train['Date'].min().date()} — {train['Date'].max().date()})")
# # print(f"Test:  {len(test)} строк ({test['Date'].min().date()} — {test['Date'].max().date()})")
#
#
#
# model = RandomForestClassifier(
#     n_estimators=100,
#     max_depth=10,        # ограничиваем глубину дерева
#     min_samples_leaf=50, # минимум 50 примеров в листе
#     random_state=42,
#     n_jobs=-1
# )
#
# model.fit(X_train, y_train)
#
# y_pred = model.predict(X_test)
#
# # считаем метрики
# accuracy = accuracy_score(y_test, y_pred)
# # print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
#
# # classification_report — подробный отчёт по каждому классу
# # print("\nПодробный отчёт:")
# # print(classification_report(y_test, y_pred,
# #       target_names=["0 (упадёт)", "1 (вырастет)"]))
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# читаем готовый датасет который сохранил EDA ноутбук
# все признаки уже созданы и нормализованы
# определяем абсолютный путь к папке проекта
# __file__ — это путь к текущему файлу (train_model.py)
# dirname — берём папку где лежит файл (backend/)
# dirname ещё раз — поднимаемся на уровень выше (kafka_ml_proj/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# теперь пути всегда правильные независимо откуда запускаешь
DATA_PATH  = os.path.join(BASE_DIR, "data", "dataset_combined.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")

df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])

print(f"Загружено строк: {len(df)}")
print(f"Колонки: {list(df.columns)}")

# берём только данные с 2010 года
# данные до 2010 содержат устаревшие паттерны рынка
df_recent = df[df["Date"] >= "2010-01-01"].copy()
print(f"Строк после фильтрации (с 2010 года): {len(df_recent)}")

# признаки для модели
# Date и ticker не используем — они не числовые
# target не используем — это то что предсказываем
features = ["Adj Close", "Volume", "volatility",
            "MA5_ratio", "MA20_ratio", "MA_cross",
            "price_change", "day_of_week", "month"]

# разбивка по времени — 80% train, 20% test
# ВАЖНО: сортируем по дате перед разбивкой
# нельзя обучать на данных 2015 года и проверять на 2010
df_sorted = df_recent.sort_values("Date")
split_idx = int(len(df_sorted) * 0.8)

train = df_sorted.iloc[:split_idx]
test  = df_sorted.iloc[split_idx:]

print(f"Train: {len(train)} строк ({train['Date'].min().date()} — {train['Date'].max().date()})")
print(f"Test:  {len(test)} строк  ({test['Date'].min().date()} — {test['Date'].max().date()})")

# X — входные признаки, y — целевой признак
X_train = train[features]
y_train = train["target"]

X_test = test[features]
y_test = test["target"]

print(f"\nX_train: {X_train.shape}")
print(f"X_test:  {X_test.shape}")

# обучаем модель Random Forest
# n_estimators=100 — количество деревьев
# max_depth=10 — ограничиваем глубину чтобы избежать переобучения
# min_samples_leaf=50 — минимум 50 примеров в каждом листе
# n_jobs=-1 — использовать все ядра процессора
print("\nНачинаем обучение...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=50,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
print("Обучение завершено!")

# предсказываем на тестовых данных
y_pred = model.predict(X_test)

# считаем метрики
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nПодробный отчёт:")
print(classification_report(y_test, y_pred,
      target_names=["0 (упадёт)", "1 (вырастет)"]))

# сохраняем модель
os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_PATH)
# print("Модель сохранена в models/model.pkl")
print(f"Размер файла: {os.path.getsize(MODEL_PATH) // 1024} KB")