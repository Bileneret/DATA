import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

# 1. Генерація даних (годинні дані за 14 днів)
np.random.seed(42)
hours = np.arange(14 * 24)

# Моделюємо сонячну радіацію (схід о 6:00, захід о 18:00)
# Використовуємо синусоїду для імітації піку вдень
radiation = 800 * np.sin(np.pi * (hours - 6) / 12) 
radiation = np.clip(radiation, 0, None) # Вночі радіація дорівнює 0

# Додаємо трохи "хмарності" (шум)
radiation -= np.clip(np.random.normal(0, 100, len(hours)), 0, None)
radiation = np.clip(radiation, 0, None)

# Моделюємо генерацію енергії. 
# ККД панелі хай буде ~15%, плюс невеликий випадковий шум
generation = radiation * 0.15 + np.random.normal(0, 5, len(hours))
generation = np.clip(generation, 0, None) # Генерація не може бути від'ємною

df = pd.DataFrame({'radiation': radiation, 'generation': generation})

# 2. Аналіз через Лінійну регресію та кореляцію Пірсона
X = df[['radiation']]
y = df['generation']

reg_model = LinearRegression().fit(X, y)
r_squared = reg_model.score(X, y)
corr_pearson, _ = pearsonr(df['radiation'], df['generation'])

# 3. Крос-кореляція (Time Series Cross-Correlation)
# Нормалізуємо дані для коректної побудови графіка
rad_norm = (df['radiation'] - df['radiation'].mean()) / df['radiation'].std()
gen_norm = (df['generation'] - df['generation'].mean()) / df['generation'].std()

# 4. Візуалізація
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Графік 1: Лінійна регресія
axes[0].scatter(df['radiation'], df['generation'], alpha=0.5, label='Фактичні дані')
axes[0].plot(df['radiation'], reg_model.predict(X), color='red', linewidth=2, label=f'Регресія (R²={r_squared:.2f})')
axes[0].set_title('Лінійна регресія: Радіація vs Генерація', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Сонячна радіація (Вт/м²)')
axes[0].set_ylabel('Генерація (кВт*год)')
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.7)

# Графік 2: Крос-кореляція (Корелограма)
# maxlags=24 означає, що ми шукаємо кореляцію зі зсувом до 24 годин вперед і назад
axes[1].xcorr(rad_norm, gen_norm, usevlines=True, maxlags=24, normed=True, lw=2, color='blue')
axes[1].set_title('Крос-кореляція (Cross-Correlation)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Лаг (зсув у часі, години)')
axes[1].set_ylabel('Коефіцієнт кореляції')
axes[1].grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# 5. Вивід результатів у консоль
print("--- Результати аналізу (Варіант 14) ---")
print(f"Коефіцієнт Пірсона (звичайна кореляція): {corr_pearson:.4f}")
print(f"Коефіцієнт детермінації (R²) лінійної регресії: {r_squared:.4f}")
print("\nВисновок для звіту:")
print("1. Лінійна регресія підтверджує сильну пряму залежність (R² близький до 1).")
print("2. Графік крос-кореляції показує, що максимальна кореляція знаходиться рівно на лагу 0.")
print("   Це означає, що генерація відбувається миттєво з появою сонця, без затримок у часі.")
print("   Також на крос-кореляції видно піки кожні 24 години, що підтверджує добову циклічність.")