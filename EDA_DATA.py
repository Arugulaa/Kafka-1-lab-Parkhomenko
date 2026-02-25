#%%

#%% md
# #  Загрузка и предварительная обработка данных
#  Был взят датасет Stocks Market
# 
#%% md
#  ## Загрузка датасета и создание персонализированного датасета
# Из-за большого объёма набора данных может потребоваться больше ресурсов для его обработки.
# Выборочно возьмём отдельные файлы из всей совокупности для лабораторной работы.
# 
#%%
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# путь к папке stocks
STOCKS_PATH = r"C:\Users\nikpa\OneDrive\Документы\Николя Учёба\Магистратура ВШЭ\1 курс\JupyterProject\data\stocks"

# находим все CSV файлы в папке
all_files = glob.glob(os.path.join(STOCKS_PATH, "*.csv"))

print(f"Найдено файлов: {len(all_files)}")

for f in all_files[:5]:
    print(os.path.basename(f))

#%%
# сортируем файлы по размеру — от большего к меньшему
# большой файл = больше исторических данных = больше строк
all_files_sorted = sorted(
    all_files,
    key=lambda f: os.path.getsize(f),
    reverse=True
)

# берём первые 100 файлов
files_to_use = all_files_sorted[:100]

print("Топ-10 крупнейших файлов:")
for f in files_to_use[:10]:
    size_kb = os.path.getsize(f) // 1024
    print(f"{os.path.basename(f):20} {size_kb} KB")

#%%
# объединяем все файлы в одну таблицу
dataframes = []

for filepath in files_to_use:
    df = pd.read_csv(filepath)
    ticker = os.path.basename(filepath).replace(".csv", "")
    df["ticker"] = ticker
    dataframes.append(df)

df_all = pd.concat(dataframes, ignore_index=True)

print(f"Итого строк: {len(df_all)}")
print(f"Итого компаний: {df_all['ticker'].nunique()}")
print(f"Колонки: {list(df_all.columns)}")
df_all.head()

#%% md
# Информация по признакам
#  * Date — Дата торгов
#  * Open — Цена открытия
#  * High — Максимальная цена за день
#  * Low — Минимальная цена за день
#  * Close — Цена закрытия (скорректированная на сплиты)
#  * Adj Close — Скорректированная цена закрытия (учитывает сплиты и дивиденды)
#  * Volume — Объём торгов
# * ticker — Тикер компании (добавлен при объединении файлов)
# 
#%% md
#  **Шаг 1: Проверка пропусков**
# 
#%%
print(f"Пропуски до удаления:")
print(df_all.isnull().sum())

df_all = df_all.dropna()
print(f"\nСтрок после удаления пропусков: {len(df_all)}")

#%% md
# **Шаг 2: Проверка дубликатов**
# 
#%%
print(f"Дубликатов: {df_all.duplicated().sum()}")
df_all = df_all.drop_duplicates()
print(f"Строк после удаления дубликатов: {len(df_all)}")

#%% md
# **Шаг 3: Обработка типов данных**
# 
#%%
# Date — конвертируем из строки в дату
df_all["Date"] = pd.to_datetime(df_all["Date"])
print(f"Тип Date: {df_all['Date'].dtype}")

# Volume — проверяем есть ли дробные значения
has_fractional = (df_all["Volume"] % 1 != 0).sum()
print(f"Строк с дробным Volume: {has_fractional}")

# дробных нет — делаем int
df_all["Volume"] = df_all["Volume"].astype(int)

df_all.info()

#%% md
# **Шаг 4: Проверка выбросов**
# Удаляем только физически невозможные значения.
#  Статистические выбросы (IQR) — только смотрим, не удаляем.
#  Резкий рост Volume это реальное рыночное событие, а не ошибка данных.
# 
#%%
print("-- Шаг 4.1: Отрицательные и нулевые значения --")
price_cols = ["Open", "High", "Low", "Close", "Adj Close"]
for col in price_cols:
    count = (df_all[col] <= 0).sum()
    print(f"  {col} <= 0: {count} строк")

count_vol = (df_all["Volume"] <= 0).sum()
print(f"  Volume <= 0: {count_vol} строк")

# удаляем строки с некорректными ценами
for col in price_cols:
    df_all = df_all[df_all[col] > 0]
df_all = df_all[df_all["Volume"] > 0]
print(f"  Строк после очистки: {len(df_all)}\n")

print("-- Шаг 4.2: Логические противоречия --")
invalid_lh = (df_all["Low"] > df_all["High"]).sum()
print(f"  Low > High: {invalid_lh} строк")
df_all = df_all[df_all["Low"] <= df_all["High"]]
print(f"  Строк после очистки: {len(df_all)}\n")

print("-- Шаг 4.3: Open и Close вне диапазона [Low, High] --")
invalid_open = ((df_all["Open"] < df_all["Low"]) |
                (df_all["Open"] > df_all["High"])).sum()
invalid_close = ((df_all["Close"] < df_all["Low"]) |
                 (df_all["Close"] > df_all["High"])).sum()
print(f"  Open вне [Low, High]: {invalid_open} строк")
print(f"  Close вне [Low, High]: {invalid_close} строк")

df_all = df_all[
    (df_all["Open"] >= df_all["Low"]) &
    (df_all["Open"] <= df_all["High"])
]
df_all = df_all[
    (df_all["Close"] >= df_all["Low"]) &
    (df_all["Close"] <= df_all["High"])
]
print(f"  Строк после очистки: {len(df_all)}\n")

print("-- Шаг 4.4: Статистические выбросы (IQR) — только смотрим --")
for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
    Q1 = df_all[col].quantile(0.25)
    Q3 = df_all[col].quantile(0.75)
    IQR = Q3 - Q1
    count = ((df_all[col] < Q1 - 1.5*IQR) |
             (df_all[col] > Q3 + 1.5*IQR)).sum()
    print(f"  Выбросов {col}: {count} ({count/len(df_all)*100:.1f}%) — не удаляем")

print(f"\nИтого строк после всех проверок: {len(df_all)}")

#%% md
# **Шаг 5: Создание целевого признака target**
#  Задача — бинарная классификация:
#  * 1 -> цена Adj Close завтра ВЫШЕ чем сегодня
#  * 0 -> цена Adj Close завтра НИЖЕ чем сегодня
# 
#%%
# сортируем по компании и дате — ОБЯЗАТЕЛЬНО перед shift
df_all = df_all.sort_values(["ticker", "Date"]).reset_index(drop=True)

# shift(-1) берёт значение следующей строки (следующий день)
# groupby гарантирует что сдвиг происходит внутри одной компании
df_all["target"] = (
    df_all.groupby("ticker")["Adj Close"].shift(-1) > df_all["Adj Close"]
).astype("Int64")

# удаляем последнюю строку каждой компании — у неё нет следующего дня
df_all = df_all.dropna(subset=["target"])
df_all["target"] = df_all["target"].astype(int)

print("Распределение целевого признака:")
print(df_all["target"].value_counts())
print(f"\nПроцентное соотношение:")
print(df_all["target"].value_counts(normalize=True).mul(100).round(2))
print(f"\nВсего строк: {len(df_all)}")

#%% md
# **Шаг 6: Feature Engineering**
# 
#  Создаём новые признаки из существующих.
# 
#  ВАЖНО: делаем это ДО нормализации — на оригинальных данных.
# 
#  Иначе скользящие средние будут коррелировать 1.00 с Adj Close.
# 
#%%
# volatility — создаём ПЕРВЫМ пока High и Low ещё существуют
# разница между максимумом и минимумом дня = нервозность рынка
# df_all["volatility"] = df_all["High"] - df_all["Low"]
df_all["volatility"] = (df_all["High"] - df_all["Low"]) / df_all["Low"]
# сначала считаем сырые скользящие средние
ma5_raw = df_all.groupby("ticker")["Adj Close"].transform(
    lambda x: x.rolling(window=5).mean()
)
ma20_raw = df_all.groupby("ticker")["Adj Close"].transform(
    lambda x: x.rolling(window=20).mean()
)

# MA5_ratio — отношение текущей цены к средней за 5 дней
# > 1.0 = цена выше краткосрочного тренда (бычий сигнал)
# < 1.0 = цена ниже краткосрочного тренда (медвежий сигнал)
df_all["MA5_ratio"] = df_all["Adj Close"] / ma5_raw

# MA20_ratio — отношение текущей цены к средней за 20 дней
df_all["MA20_ratio"] = df_all["Adj Close"] / ma20_raw

# MA_cross — разница между краткосрочным и долгосрочным трендом
# > 0 = краткосрочный тренд сильнее долгосрочного (бычий сигнал)
# < 0 = долгосрочный тренд сильнее (медвежий сигнал)
df_all["MA_cross"] = ma5_raw - ma20_raw

# price_change — процентное изменение цены за день
# pct_change() = (сегодня - вчера) / вчера
df_all["price_change"] = df_all.groupby("ticker")["Adj Close"].transform(
    lambda x: x.pct_change()
)

# день недели: 0=понедельник, 4=пятница
df_all["day_of_week"] = df_all["Date"].dt.dayofweek

# месяц: 1=январь, 12=декабрь
df_all["month"] = df_all["Date"].dt.month

# удаляем ненужные колонки
# Open, High, Low — дублируют Adj Close (корреляция 0.99-1.00)
# Close — заменён на Adj Close (более честная цена)
df_all = df_all.drop(columns=["Open", "High", "Low", "Close"])

# удаляем строки с NaN (первые 19 строк каждой компании не имеют MA20)
df_all = df_all.dropna()

print(f"Строк после Feature Engineering: {len(df_all)}")
print(f"Колонки: {list(df_all.columns)}")

#%% md
# **Шаг 7: EDA — анализ данных**
# 
#  Проводим EDA ДО нормализации — на оригинальных данных.
# 
#  Так графики показывают реальные значения, а не числа от 0 до 1.
# 
#%% md
# **График 1: Распределение целевого признака**
# 
#%%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

target_counts = df_all["target"].value_counts()
target_labels = ["0 (упадёт)", "1 (вырастет)"]

axes[0].bar(target_labels, target_counts.values, color=["#e74c3c", "#2ecc71"])
axes[0].set_title("Баланс классов (количество)")
axes[0].set_ylabel("Количество строк")

for i, v in enumerate(target_counts.values):
    axes[0].text(i, v + 1000, str(v), ha="center", fontsize=11)

target_pct = df_all["target"].value_counts(normalize=True) * 100
axes[1].pie(
    target_pct.values,
    labels=target_labels,
    autopct="%1.1f%%",
    colors=["#e74c3c", "#2ecc71"]
)
axes[1].set_title("Баланс классов (проценты)")

plt.suptitle("Распределение целевого признака (target)", fontsize=14)
plt.tight_layout()
plt.show()

#%% md
#  **График 2: Распределение Adj Close и Volume**
# 
#%%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# гистограмма Adj Close — на оригинальных данных видны реальные цены
axes[0].hist(df_all["Adj Close"], bins=150, color="steelblue", edgecolor="black")
axes[0].set_title("Распределение цен (Adj Close)")
axes[0].set_xlabel("Цена ($)")
axes[0].set_ylabel("Количество строк")

# гистограмма Volume
axes[1].hist(df_all["Volume"], bins=150, color="coral", edgecolor="black")
axes[1].set_title("Распределение объёма торгов (Volume)")
axes[1].set_xlabel("Объём (кол-во акций)")
axes[1].set_ylabel("Количество строк")

plt.suptitle("График 2: Распределение признаков (оригинальные данные)", fontsize=14)
plt.tight_layout()
plt.show()

#%% md
#  **График 3: Количество записей по годам**
# 
#%%
df_all["year"] = df_all["Date"].dt.year
records_per_year = df_all.groupby("year").size()

plt.figure(figsize=(14, 5))
plt.bar(records_per_year.index, records_per_year.values, color="steelblue")
plt.title("График 3: Количество торговых записей по годам")
plt.xlabel("Год")
plt.ylabel("Количество записей")
plt.axvline(x=2008, color="red", linestyle="--", label="Кризис 2008")
plt.axvline(x=2020, color="orange", linestyle="--", label="COVID 2020")
plt.legend()
plt.tight_layout()
plt.show()

df_all = df_all.drop(columns=["year"])

#%% md
#  **График 4: Корреляционная матрица**
# 
#%%
numeric_cols = ["Adj Close", "Volume", "volatility", "MA5_ratio",
                "MA20_ratio", "MA_cross", "price_change",
                "day_of_week", "month", "target"]

correlation_matrix = df_all[numeric_cols].corr()

plt.figure(figsize=(12, 9))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True
)
plt.title("График 4: Корреляционная матрица признаков")
plt.tight_layout()
plt.show()

print("Корреляция каждого признака с target (по убыванию):")
print(correlation_matrix["target"].drop("target").sort_values(
    key=abs, ascending=False
))

#%% md
#  **График 5: Временной ряд одной компании**
# 
#%%
top_ticker = df_all["ticker"].value_counts().index[0]
company_data = df_all[df_all["ticker"] == top_ticker].copy()

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

axes[0].plot(company_data["Date"], company_data["Adj Close"],
             color="steelblue", linewidth=0.8)
axes[0].set_title(f"{top_ticker} — История цены закрытия")
axes[0].set_xlabel("Дата")
axes[0].set_ylabel("Adj Close ($)")
axes[0].axvline(pd.Timestamp("2008-09-15"), color="red",
                linestyle="--", alpha=0.7, label="Кризис 2008")
axes[0].axvline(pd.Timestamp("2020-03-01"), color="orange",
                linestyle="--", alpha=0.7, label="COVID 2020")
axes[0].legend()

axes[1].bar(company_data["Date"], company_data["Volume"],
            color="steelblue", alpha=0.7, width=1)
axes[1].set_title(f"{top_ticker} — Объём торгов")
axes[1].set_xlabel("Дата")
axes[1].set_ylabel("Volume")

plt.suptitle(f"График 5: Временной ряд компании {top_ticker}", fontsize=14)
plt.tight_layout()
plt.show()

#%% md
#  **Шаг 8: Нормализация**
# 
# ВАЖНО: нормализуем ПОСЛЕ EDA — чтобы графики показывали реальные значения.
# 
#  Нормализуем только признаки для модели. Date, ticker, target — не трогаем.
# 
#%%
from sklearn.preprocessing import MinMaxScaler
import joblib

# признаки которые нормализуем
features_to_scale = ["Adj Close", "Volume", "volatility",
                     "MA5_ratio", "MA20_ratio", "MA_cross",
                     "price_change"]

# создаём нормализатор
scaler = MinMaxScaler()

# fit_transform — изучает данные и применяет нормализацию
df_all[features_to_scale] = scaler.fit_transform(df_all[features_to_scale])

print("Минимальные значения после нормализации:")
print(df_all[features_to_scale].min())
print("\nМаксимальные значения после нормализации:")
print(df_all[features_to_scale].max())

#%% md
#  Сохраняем скалер — он понадобится Consumer'у чтобы нормализовывать
#  новые данные из Kafka теми же параметрами что и при обучении модели.
# 
#%%
joblib.dump(scaler, "models/scaler.pkl")
print(f"Скалер сохранён. Размер: {os.path.getsize('models/scaler.pkl')} байт")

#%% md
#  **Шаг 9: Сохранение датасета**
# 
#%%
df_all.to_csv("data/dataset_combined.csv", index=False)

print(f"Датасет сохранён в data/dataset_combined.csv")
print(f"Итого строк: {len(df_all)}")
print(f"Итого колонок: {len(df_all.columns)}")
print(f"Колонки: {list(df_all.columns)}")
#%% md
# # Обучение модели
#%%
# # признаки для модели — всё кроме Date, ticker, target
# features = ["Adj Close", "Volume", "volatility",
#             "MA5_ratio", "MA20_ratio", "MA_cross",
#             "price_change", "day_of_week", "month"]
#
# # целевой признак
# target = "target"
#
# # делим по времени — train до 2018, test после
# # это правильно для временных рядов
# train = df_all[df_all["Date"] < "2018-01-01"]
# test  = df_all[df_all["Date"] >= "2018-01-01"]
#
# print(f"Train: {len(train)} строк ({train['Date'].min().date()} — {train['Date'].max().date()})")
# print(f"Test:  {len(test)} строк ({test['Date'].min().date()} — {test['Date'].max().date()})")
#
# # X — входные данные (признаки)
# # y — правильные ответы (целевой признак)
# X_train = train[features]
# y_train = train[target]
#
# X_test = test[features]
# y_test = test[target]
#
# print(f"\nX_train: {X_train.shape}")
# print(f"X_test:  {X_test.shape}")

# берём только данные с 2010 года
# рынок с 2010 более однородный чем с 1962
df_recent = df_all[df_all["Date"] >= "2010-01-01"].copy()

print(f"Строк после фильтрации: {len(df_recent)}")

# делим 80/20 по времени
split_date = df_recent["Date"].quantile(0.8)
# quantile не работает с датами напрямую — используем iloc
split_idx = int(len(df_recent) * 0.8)
df_sorted = df_recent.sort_values("Date")

train = df_sorted.iloc[:split_idx]
test  = df_sorted.iloc[split_idx:]

print(f"Train: {len(train)} строк ({train['Date'].min().date()} — {train['Date'].max().date()})")
print(f"Test:  {len(test)} строк ({test['Date'].min().date()} — {test['Date'].max().date()})")
#%%
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report
# import joblib
# import pandas as pd
# # создаём модель Random Forest
# # n_estimators=100 — количество деревьев в лесу
# # random_state=42 — фиксируем случайность для воспроизводимости
# # n_jobs=-1 — использовать все ядра процессора для скорости
# model = RandomForestClassifier(
#     n_estimators=100,
#     random_state=42,
#     n_jobs=-1
# )
#
# print("Начинаем обучение...")
#
# # fit — обучение модели на тренировочных данных
# # модель ищет закономерности в X_train которые объясняют y_train
# model.fit(X_train, y_train)
#
# print("Обучение завершено!")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,        # ограничиваем глубину дерева
    min_samples_leaf=50, # минимум 50 примеров в листе
    random_state=42,
    n_jobs=-1
)

print("Начинаем обучение...")
model.fit(X_train, y_train)
print("Обучение завершено!")
#%%
# предсказываем на тестовых данных
# модель никогда не видела эти данные во время обучения
y_pred = model.predict(X_test)

# считаем метрики
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# classification_report — подробный отчёт по каждому классу
print("\nПодробный отчёт:")
print(classification_report(y_test, y_pred,
      target_names=["0 (упадёт)", "1 (вырастет)"]))



#%%

#%%
