# Proekt
Проект прогнозирования цены Dogecoin
1. Введение
В данном проекте была реализована модель машинного обучения для прогнозирования цены Dogecoin на основе исторических данных. Основная цель заключалась в создании механизма, позволяющего обновлять модель по запросу, а также выдавать предсказание цены через через 10 минут.
2. Получение и обработка данных
Для сбора данных использовался yfinance, позволяющий получать котировки Dogecoin в реальном времени. Исторические данные загружаются за последние 7 дней с шагом в 1 минуту.
Код загрузки данных:
import yfinance as yf
import pandas as pd

def get_doge_data():
    doge = yf.Ticker("DOGE-USD")
    df = doge.history(period="7d", interval="1m")  # Загружаем данные за неделю с шагом 1 минута
    df.reset_index(inplace=True)

    # Добавляем временные признаки
    df["hour"] = df["Datetime"].dt.hour
    df["minute"] = df["Datetime"].dt.minute
    df["dayofweek"] = df["Datetime"].dt.dayofweek

    # Добавляем лаговые признаки
    df["Close_lag1"] = df["Close"].shift(1)
    df["Close_lag5"] = df["Close"].shift(5)
    df["Close_lag10"] = df["Close"].shift(10)

    return df.dropna()  # Убираем строки с пропущенными значениями


Преобразования данных включают:
Добавление часа, минуты и дня недели, чтобы учесть временные закономерности.
Введение лаговых признаков (Close_lag1, Close_lag5, Close_lag10), чтобы модель могла анализировать тренды.

3. Обучение модели
Для прогнозирования цены была выбрана модель RandomForestRegressor. Она использует исторические данные и временные признаки для предсказания будущей цены.
Код обучения модели:
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Выбираем признаки и целевую переменную
features = ["Open", "High", "Low", "Volume", "hour", "minute", "dayofweek", "Close_lag1", "Close_lag5", "Close_lag10"]
target = "Close"

X = data[features]
y = data[target]

# Разделяем на тренировочную и тестовую выборку
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Инициализируем модель
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

Основные моменты обучения:
Данные разделяются на тренировочную (80%) и тестовую (20%) выборку.
Используется 200 деревьев решений (n_estimators=200) для улучшения качества предсказания.


4. Прогнозирование цен
После загрузки новых данных модель дообучается и выполняет предсказание цены через 10 минут
Код прогнозирования:
import numpy as np

def predict_future_prices():
    global model, X_train, y_train

    # Загружаем новые данные
    new_data = get_doge_data()
    new_data.set_index('Datetime', inplace=True)

    # Обновляем обучающий набор
    combined_data = pd.concat([data, new_data]).drop_duplicates()
    X_train, y_train = combined_data[features], combined_data[target]

    # Дообучаем модель
    model.fit(X_train, y_train)

    # Прогнозируем будущие цены
    last_row = pd.DataFrame([new_data.iloc[-1][features]], columns=features)
    future_times = [10, 60, 360]  # Минуты
    prediction_results = []

    for future_time in future_times:
        future_index = new_data.index[-1] + pd.Timedelta(minutes=future_time)

        # Обновляем временные признаки
        last_row["hour"] = future_index.hour
        last_row["minute"] = future_index.minute
        last_row["dayofweek"] = future_index.dayofweek

        predicted_price = model.predict(last_row)[0]
        prediction_results.append((future_index, predicted_price))
        print(f"Через {future_time} минут ({future_index}): {predicted_price:.5f} USD")
    
    # Строим график
    plot_predictions(new_data, prediction_results)


5. Визуализация результатов
Для лучшего понимания результатов предсказаний строится график:
Синяя линия – фактическая цена Dogecoin.
Красные крестики – предсказанные значения на будущие временные метки.
Код построения графика:
import matplotlib.pyplot as plt

def plot_predictions(data, prediction_results):
    plt.figure(figsize=(12, 6))
    
    # График фактических цен
    plt.plot(data.index[-500:], data["Close"].tail(500), label="Фактическая цена", color="blue")

    # Прогнозируемые значения
    times, prices = zip(*prediction_results)
    plt.scatter(times, prices, color="red", label="Прогноз", marker="x", s=100)

    plt.xlabel("Дата и время")
    plt.ylabel("Цена Dogecoin (USD)")
    plt.title("Прогноз цены Dogecoin")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.show()

Преимущества визуализации:
Позволяет сравнивать прогнозируемые цены с реальными.
Удобно анализировать тренды и потенциальные ошибки модели.
6. Выводы и перспективы
Проект показывает, что RandomForestRegressor можно использовать для прогнозирования краткосрочных цен Dogecoin, но есть несколько направлений для улучшения:
Улучшение модели: попробовать LSTM или XGBoost, которые лучше работают с временными рядами.
Дополнительные признаки: учитывать скользящие средние, RSI, объем торгов.
API-интеграция: автоматизировать обновление модели и запросы к yfinance через FastAPI или Flask.
Чат-бот: Интеграция чат-бота Телеграмма для получения прогнозов цены в реальном времени
