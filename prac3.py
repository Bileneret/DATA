import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Генерація даних енергоспоживання з трендом, сезонністю та шумом
dates = pd.date_range(start='2020-01-01', periods=60, freq='ME')
np.random.seed(42)

trend_component = np.linspace(1000, 1500, len(dates)) 
seasonal_component = 200 * np.sin(2 * np.pi * dates.month / 12) 
noise_component = np.random.normal(0, 50, len(dates)) 

consumption = trend_component + seasonal_component + noise_component
df = pd.DataFrame({'consumption': consumption}, index=dates)

# Розкладання на складові (тренд, сезонність, залишок)
decomposition = seasonal_decompose(df['consumption'], model='additive')

# Візуалізація графіків оригінальних даних, тренду, сезонності та залишку
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

# Оригінальні дані (Споживання)
axes[0].plot(df.index, df['consumption'], label='Оригінальні дані', color='#1f77b4', linewidth=2)
axes[0].set_title('Декомпозиція часового ряду енергоспоживання', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Споживання')

# Тренд 
axes[1].plot(decomposition.trend.index, decomposition.trend, label='Тренд', color='#ff7f0e', linewidth=2)
axes[1].set_ylabel('Тренд')

# Сезонність 
axes[2].plot(decomposition.seasonal.index, decomposition.seasonal, label='Сезонність', color='#2ca02c', linewidth=2)
axes[2].set_ylabel('Сезонність')

# Залишок (Шум)
axes[3].scatter(decomposition.resid.index, decomposition.resid, label='Залишок (Шум)', color='#d62728', s=30)
axes[3].axhline(0, color='black', linestyle='--', linewidth=1)
axes[3].set_ylabel('Залишок')
axes[3].set_xlabel('Дата')

# Налаштування сітки та легенди для кожного графіка
for ax in axes:
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper left')

plt.tight_layout()
plt.show()

# Оцінка точності моделі (Тренд + Сезонність)
reconstructed = decomposition.trend + decomposition.seasonal
valid_data = pd.concat([df['consumption'], reconstructed.rename('forecast')], axis=1).dropna()

# Розрахунок MAE та RMSE
actual = valid_data['consumption']
forecast = valid_data['forecast']

# Розрахунок середнього споживання для відносної похибки
mae = mean_absolute_error(actual, forecast)
rmse = np.sqrt(mean_squared_error(actual, forecast))
mean_consumption = actual.mean()

print("--- Варіант 14 ---")
print("Оцінка точності моделі (Тренд + Сезонність):")
print(f"Середнє споживання: {mean_consumption:.2f} кВт*год")
print(f"MAE (Середня абсолютна похибка): {mae:.2f} кВт*год")
print(f"RMSE (Середньоквадратична похибка): {rmse:.2f} кВт*год")
print(f"Відносна похибка (MAE / Mean): {(mae / mean_consumption) * 100:.2f}%")