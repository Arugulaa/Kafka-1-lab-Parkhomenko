#%%

#%% md
# 
#%% md
# # # Загрузка и предварительная обработка данных
# # Датасет: Stock Market Dataset (Kaggle)
# 
#%% md
# # ## Загрузка датасета
# # Из-за большого объёма берём топ-100 файлов по размеру.
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

all_files = glob.glob(os.path.join(STOCKS_PATH, "*.csv"))
print(f"Найдено файлов: {len(all_files)}")

# сортируем по размеру — большой файл = больше исторических данных
all_files_sorted = sorted(all_files, key=lambda f: os.path.getsize(f), reverse=True)
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
# # ## Информация по признакам
# # * Date — Дата торгов
# # * Open — Цена открытия
# # * High — Максимальная цена за день
# # * Low — Минимальная цена за день
# # * Close — Цена закрытия (скорректированная на сплиты)
# # * Adj Close — Скорректированная цена закрытия (учитывает сплиты и дивиденды)
# # * Volume — Объём торгов
# # * ticker — Тикер компании (добавлен при объединении файлов)
# 
#%% md
# # ## Шаг 1: Проверка пропусков
# 
#%%
print("Пропуски до удаления:")
print(df_all.isnull().sum())

df_all = df_all.dropna()
print(f"\nСтрок после удаления пропусков: {len(df_all)}")

#%% md
# # ## Шаг 2: Проверка дубликатов
# 
#%%
print(f"Дубликатов: {df_all.duplicated().sum()}")
df_all = df_all.drop_duplicates()
print(f"Строк после удаления дубликатов: {len(df_all)}")

#%% md
# # ## Шаг 3: Обработка типов данных
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
# # ## Шаг 4: Проверка выбросов
# # Удаляем только физически невозможные значения.
# # Статистические выбросы (IQR) — только смотрим, не удаляем:
# # резкий рост Volume это реальное рыночное событие, а не ошибка.
# 
#%%
print("-- Шаг 4.1: Отрицательные и нулевые значения --")
price_cols = ["Open", "High", "Low", "Close", "Adj Close"]
for col in price_cols:
    count = (df_all[col] <= 0).sum()
    print(f"  {col} <= 0: {count} строк")

count_vol = (df_all["Volume"] <= 0).sum()
print(f"  Volume <= 0: {count_vol} строк")

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
    count = ((df_all[col] < Q1 - 1.5 * IQR) |
             (df_all[col] > Q3 + 1.5 * IQR)).sum()
    print(f"  Выбросов {col}: {count} ({count / len(df_all) * 100:.1f}%) — не удаляем")

print(f"\nИтого строк после всех проверок: {len(df_all)}")

#%% md
# # ## Шаг 5: Создание целевого признака target
# # Задача — бинарная классификация:
# # * 1 -> цена Adj Close завтра ВЫШЕ чем сегодня
# # * 0 -> цена Adj Close завтра НИЖЕ или равна сегодняшней
# 
#%%
# сортируем по компании и дате — ОБЯЗАТЕЛЬНО перед shift
df_all = df_all.sort_values(["ticker", "Date"]).reset_index(drop=True)

# shift(-1) берёт значение следующей строки (следующий день)
# groupby гарантирует сдвиг внутри одной компании
df_all["target"] = (
    df_all.groupby("ticker")["Adj Close"].shift(-1) > df_all["Adj Close"]
).astype("Int64")

# последняя строка каждой компании получает NaN — удаляем
df_all = df_all.dropna(subset=["target"])
df_all["target"] = df_all["target"].astype(int)

print("Распределение целевого признака:")
print(df_all["target"].value_counts())
print(f"\nПроцентное соотношение:")
print(df_all["target"].value_counts(normalize=True).mul(100).round(2))
print(f"\nВсего строк: {len(df_all)}")

#%% md
# # ## Шаг 6: Feature Engineering
# # Создаём новые признаки ДО нормализации — на оригинальных данных.
# #
# # Почему ДО нормализации:
# # MA от уже нормализованных данных коррелирует 1.00 с Adj Close
# # и не несёт новой информации для модели.
# #
# # Что добавляем и зачем:
# # * volatility     — нервозность рынка внутри дня
# # * MA5_ratio      — отклонение цены от 5-дневного тренда
# # * MA20_ratio     — отклонение цены от 20-дневного тренда
# # * MA_cross       — пересечение краткосрочного и долгосрочного трендов
# # * price_change   — импульс за 1 день
# # * return_5d      — импульс за 5 дней (неделя)
# # * return_20d     — импульс за 20 дней (месяц)
# # * volume_ratio   — объём относительно нормы (повышенный интерес?)
# # * std_20d        — волатильность за 20 дней (нервозность периода)
# # * day_of_week    — день недели (рынок ведёт себя по-разному)
# # * month          — месяц (сезонные паттерны)
# 
#%%
# ВАЖНО: берём только данные с 2000 года
# рынок 1962 года сильно отличается от современного
# старые данные добавляют шум, а не полезные паттерны
df_all = df_all[df_all["Date"] >= "2000-01-01"].copy()
df_all = df_all.sort_values(["ticker", "Date"]).reset_index(drop=True)
print(f"Строк после фильтрации (с 2000 года): {len(df_all)}")

#%%
# volatility — создаём ПЕРВЫМ пока High и Low ещё существуют
# процентная волатильность: на сколько % колебалась цена за день
# используем процент а не абсолютную разницу
# иначе дорогие акции (Google 2000$) будут иметь искусственно высокую volatility
df_all["volatility"] = (df_all["High"] - df_all["Low"]) / df_all["Low"]

# считаем сырые скользящие средние — нужны для ratio и cross
ma5_raw = df_all.groupby("ticker")["Adj Close"].transform(
    lambda x: x.rolling(window=5).mean()
)
ma20_raw = df_all.groupby("ticker")["Adj Close"].transform(
    lambda x: x.rolling(window=20).mean()
)

# MA5_ratio: > 1.0 = цена выше краткосрочного тренда (бычий сигнал)
df_all["MA5_ratio"] = df_all["Adj Close"] / ma5_raw

# MA20_ratio: > 1.0 = цена выше долгосрочного тренда (бычий сигнал)
df_all["MA20_ratio"] = df_all["Adj Close"] / ma20_raw

# MA_cross: > 0 = краткосрочный тренд сильнее долгосрочного
df_all["MA_cross"] = ma5_raw - ma20_raw

# price_change: изменение цены за 1 день
df_all["price_change"] = df_all.groupby("ticker")["Adj Close"].transform(
    lambda x: x.pct_change()
)

# return_5d: изменение цены за 5 дней (моментум недели)
# показывает силу тренда за последние 5 торговых дней
df_all["return_5d"] = df_all.groupby("ticker")["Adj Close"].transform(
    lambda x: x.pct_change(periods=5)
)

# return_20d: изменение цены за 20 дней (моментум месяца)
df_all["return_20d"] = df_all.groupby("ticker")["Adj Close"].transform(
    lambda x: x.pct_change(periods=20)
)

# volume_ratio: объём сегодня относительно среднего за 20 дней
# > 1.0 = сегодня торгуют активнее чем обычно (повышенный интерес)
vol_ma20 = df_all.groupby("ticker")["Volume"].transform(
    lambda x: x.rolling(window=20).mean()
)
df_all["volume_ratio"] = df_all["Volume"] / vol_ma20

# std_20d: стандартное отклонение цены за 20 дней
# высокое std = рынок нервничает = меньше предсказуемости
df_all["std_20d"] = df_all.groupby("ticker")["Adj Close"].transform(
    lambda x: x.rolling(window=20).std()
)

# день недели: 0=понедельник, 4=пятница
df_all["day_of_week"] = df_all["Date"].dt.dayofweek

# месяц: 1=январь, 12=декабрь
df_all["month"] = df_all["Date"].dt.month

# удаляем ненужные колонки
# Open, High, Low — дублируют Adj Close (корреляция 0.99-1.00)
# Close — заменён на Adj Close (более честная цена без дивидендных артефактов)
df_all = df_all.drop(columns=["Open", "High", "Low", "Close"])

# удаляем NaN — появляются из-за rolling(window=20) и pct_change(periods=20)
df_all = df_all.dropna()

print(f"Строк после Feature Engineering: {len(df_all)}")
print(f"Колонки: {list(df_all.columns)}")

#%% md
# # ## Шаг 7: EDA — анализ данных
# # Проводим EDA ДО нормализации — графики показывают реальные значения.
# 
#%% md
# # ### График 1: Распределение целевого признака
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
axes[1].pie(target_pct.values, labels=target_labels,
            autopct="%1.1f%%", colors=["#e74c3c", "#2ecc71"])
axes[1].set_title("Баланс классов (проценты)")

plt.suptitle("График 1: Распределение целевого признака (target)", fontsize=14)
plt.tight_layout()
plt.show()

#%% md
# # ### График 2: Распределение Adj Close и Volume
# 
#%%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df_all["Adj Close"], bins=50, color="steelblue", edgecolor="black")
axes[0].set_title("Распределение цен (Adj Close)")
axes[0].set_xlabel("Цена ($)")
axes[0].set_ylabel("Количество строк")

axes[1].hist(df_all["Volume"], bins=50, color="coral", edgecolor="black")
axes[1].set_title("Распределение объёма торгов (Volume)")
axes[1].set_xlabel("Объём (кол-во акций)")
axes[1].set_ylabel("Количество строк")

plt.suptitle("График 2: Распределение признаков (оригинальные данные)", fontsize=14)
plt.tight_layout()
plt.show()

#%% md
# # ### График 3: Количество записей по годам
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
# # ### График 4: Корреляционная матрица
# # Проверяем что новые признаки не дублируют друг друга.
# 
#%%
numeric_cols = ["Adj Close", "Volume", "volatility", "MA5_ratio", "MA20_ratio",
                "MA_cross", "price_change", "return_5d", "return_20d",
                "volume_ratio", "std_20d", "day_of_week", "month", "target"]

correlation_matrix = df_all[numeric_cols].corr()

plt.figure(figsize=(14, 11))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, square=True)
plt.title("График 4: Корреляционная матрица признаков")
plt.tight_layout()
plt.show()

print("Корреляция каждого признака с target (по убыванию):")
print(correlation_matrix["target"].drop("target").sort_values(
    key=abs, ascending=False
))

#%% md
# # ### График 5: Временной ряд одной компании
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
            color="coral", alpha=0.7, width=1)
axes[1].set_title(f"{top_ticker} — Объём торгов")
axes[1].set_xlabel("Дата")
axes[1].set_ylabel("Volume")

plt.suptitle(f"График 5: Временной ряд компании {top_ticker}", fontsize=14)
plt.tight_layout()
plt.show()

#%% md
# # ## Шаг 8: Нормализация
# # Нормализуем ПОСЛЕ EDA — чтобы графики показывали реальные значения.
# # Random Forest не требует нормализации, но она нужна для:
# # 1. Logistic Regression (сравниваем несколько моделей)
# # 2. Consumer в Kafka — должен нормализовать новые данные теми же параметрами
# 
#%%
from sklearn.preprocessing import MinMaxScaler
import joblib

# нормализуем все числовые признаки кроме целевого и временных
features_to_scale = ["Adj Close", "Volume", "volatility", "MA5_ratio",
                     "MA20_ratio", "MA_cross", "price_change",
                     "return_5d", "return_20d", "volume_ratio", "std_20d"]

scaler = MinMaxScaler()
df_all[features_to_scale] = scaler.fit_transform(df_all[features_to_scale])

print("Минимальные значения после нормализации:")
print(df_all[features_to_scale].min())
print("\nМаксимальные значения после нормализации:")
print(df_all[features_to_scale].max())

# сохраняем скалер — Consumer будет использовать те же параметры нормализации
joblib.dump(scaler, "models/scaler.pkl")
print(f"\nСкалер сохранён. Размер: {os.path.getsize('models/scaler.pkl')} байт")

#%% md
# # ## Шаг 9: Сохранение датасета
# 
#%%
df_all.to_csv("data/dataset_combined.csv", index=False)
print(f"Датасет сохранён: {len(df_all)} строк, {len(df_all.columns)} колонок")
print(f"Колонки: {list(df_all.columns)}")

#%% md
# # ## Шаг 10: Обучение и сравнение моделей
# #
# # Сравниваем три модели:
# # 1. Random Forest — ансамбль деревьев, не чувствителен к масштабу
# # 2. Gradient Boosting — деревья строятся последовательно, каждое исправляет ошибки предыдущего
# # 3. Logistic Regression — простая линейная модель, хороший baseline
# #
# # Разбивка по времени (а не случайная):
# # нельзя обучать на данных 2015 года и проверять на 2010 — модель будет "знать будущее"
# 
#%%
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# список признаков для модели
features = ["Adj Close", "Volume", "volatility", "MA5_ratio", "MA20_ratio",
            "MA_cross", "price_change", "return_5d", "return_20d",
            "volume_ratio", "std_20d", "day_of_week", "month"]

# разбивка 80/20 по времени
df_sorted = df_all.sort_values("Date")
split_idx = int(len(df_sorted) * 0.8)

train = df_sorted.iloc[:split_idx]
test  = df_sorted.iloc[split_idx:]

print(f"Train: {len(train)} строк ({train['Date'].min().date()} — {train['Date'].max().date()})")
print(f"Test:  {len(test)} строк  ({test['Date'].min().date()} — {test['Date'].max().date()})")

X_train = train[features]
y_train = train["target"]
X_test  = test[features]
y_test  = test["target"]

print(f"\nX_train: {X_train.shape}")
print(f"X_test:  {X_test.shape}")
print(f"\nNaN в X_train: {X_train.isnull().sum().sum()}")
print(f"NaN в X_test:  {X_test.isnull().sum().sum()}")

#%%
# модель 1: Random Forest
# max_depth=8 — ограничиваем глубину чтобы избежать переобучения
# min_samples_leaf=50 — в каждом листе минимум 50 примеров
print("Обучаем Random Forest...")
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    min_samples_leaf=50,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
acc_rf = accuracy_score(y_test, rf.predict(X_test))
print(f"Random Forest: {acc_rf * 100:.2f}%")

#%%
# модель 2: Gradient Boosting
# строит деревья последовательно — каждое исправляет ошибки предыдущего
# обычно точнее чем Random Forest на табличных данных
# learning_rate=0.1 — шаг обучения (меньше = медленнее но точнее)
print("Обучаем Gradient Boosting...")
gb = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)
gb.fit(X_train, y_train)
acc_gb = accuracy_score(y_test, gb.predict(X_test))
print(f"Gradient Boosting: {acc_gb * 100:.2f}%")

#%%
# модель 3: Logistic Regression
# простая линейная модель — хороший baseline для сравнения
# требует нормализованных данных (у нас уже нормализовано)
print("Обучаем Logistic Regression...")
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train, y_train)
acc_lr = accuracy_score(y_test, lr.predict(X_test))
print(f"Logistic Regression: {acc_lr * 100:.2f}%")

#%%
# сравниваем результаты
print("\n=== Сравнение моделей ===")
results = {
    "Random Forest":      acc_rf,
    "Gradient Boosting":  acc_gb,
    "Logistic Regression": acc_lr
}
for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"  {name:25} {acc * 100:.2f}%")

# выбираем лучшую модель
best_name = max(results, key=results.get)
best_acc  = results[best_name]
print(f"\nЛучшая модель: {best_name} ({best_acc * 100:.2f}%)")

# сохраняем лучшую модель
model_map = {
    "Random Forest": rf,
    "Gradient Boosting": gb,
    "Logistic Regression": lr
}
best_model = model_map[best_name]
joblib.dump(best_model, "models/model.pkl")
print("Лучшая модель сохранена в models/model.pkl")

#%%
# подробный отчёт по лучшей модели
y_pred = best_model.predict(X_test)
print(f"\nПодробный отчёт ({best_name}):")
print(classification_report(y_test, y_pred,
      target_names=["0 (упадёт)", "1 (вырастет)"]))

#%% md
# # ### График 6: Важность признаков
# 
#%%
# важность признаков доступна только у деревьев
# для Logistic Regression используем коэффициенты
if best_name in ["Random Forest", "Gradient Boosting"]:
    importance = best_model.feature_importances_
else:
    importance = abs(best_model.coef_[0])

feature_importance = pd.DataFrame({
    "признак": features,
    "важность": importance
}).sort_values("важность", ascending=True)

plt.figure(figsize=(10, 7))
plt.barh(feature_importance["признак"],
         feature_importance["важность"],
         color="steelblue")
plt.title(f"График 6: Важность признаков ({best_name})")
plt.xlabel("Важность")
plt.tight_layout()
plt.show()

print("Топ-5 самых важных признаков:")
print(feature_importance.sort_values("важность", ascending=False).head())

#%% md
# # ### График 7: Сравнение точности моделей
# 
#%%
model_names = list(results.keys())
model_accs  = [v * 100 for v in results.values()]

plt.figure(figsize=(8, 5))
bars = plt.bar(model_names, model_accs, color=["steelblue", "coral", "green"])
plt.title("График 7: Сравнение точности моделей")
plt.ylabel("Accuracy (%)")
plt.ylim(45, 70)

# линия случайного угадывания (50%)
plt.axhline(y=50, color="red", linestyle="--", label="Случайное угадывание (50%)")
plt.legend()

# подписываем значения на столбцах
for bar, acc in zip(bars, model_accs):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.3,
             f"{acc:.2f}%", ha="center", fontsize=11)

plt.tight_layout()
plt.show()