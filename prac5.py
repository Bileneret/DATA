import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

# Вимикаємо зайві попередження від statsmodels
warnings.filterwarnings('ignore')

# 1. Генерація історичних даних попиту (щоденні продажі за 2 роки)
np.random.seed(42)
dates = pd.date_range(start='2022-01-01', periods=730, freq='D')

# Моделюємо попит: базова лінія + зростаючий тренд + тижнева сезонність + шум
base_demand = 100
trend = np.linspace(0, 150, len(dates))  # Попит поступово зростає
seasonality = 30 * np.sin(2 * np.pi * dates.dayofweek / 7)  # Піки в певні дні тижня
noise = np.random.normal(0, 15, len(dates))  # Випадкові коливання

demand = base_demand + trend + seasonality + noise
df = pd.DataFrame({'demand': demand}, index=dates)

# 2. Розбиття на тренувальну та тестову вибірки (80% / 20%)
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

print("Навчаємо модель ARIMA... (це може зайняти кілька секунд)")

# 3. Навчання моделі ARIMA
# Параметри (p, d, q) = (7, 1, 1). p=7 дозволяє моделі "зрозуміти" тижневу сезонність (7 днів)
model = ARIMA(train['demand'], order=(7, 1, 1))
fitted_model = model.fit()

# 4. Прогнозування на період тестової вибірки
forecast = fitted_model.forecast(steps=len(test))
forecast.index = test.index # Вирівнюємо індекси дат

# 5. Розрахунок метрик точності
mae = mean_absolute_error(test['demand'], forecast)
rmse = np.sqrt(mean_squared_error(test['demand'], forecast))
# Розрахунок MAPE (Mean Absolute Percentage Error)
mape = np.mean(np.abs((test['demand'] - forecast) / test['demand'])) * 100

print("\n--- Результати прогнозування (ARIMA) ---")
print(f"MAE (Середня абсолютна похибка): {mae:.2f} одиниць товару")
print(f"RMSE (Середньоквадратична похибка): {rmse:.2f} одиниць товару")
print(f"MAPE (Середня відносна похибка): {mape:.2f}%")

# 6. Візуалізація результатів
plt.figure(figsize=(14, 6))

# Будуємо графіки
plt.plot(train.index, train['demand'], label='Історичні дані (Тренування)', color='#1f77b4', alpha=0.8)
plt.plot(test.index, test['demand'], label='Фактичний попит (Тест)', color='#2ca02c', linewidth=2)
plt.plot(test.index, forecast, label='Прогноз ARIMA', color='#d62728', linestyle='--', linewidth=2)

plt.title('Прогнозування попиту на товар за допомогою моделі ARIMA', fontsize=14, fontweight='bold')
plt.xlabel('Дата')
plt.ylabel('Попит (кількість проданих товарів)')
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()