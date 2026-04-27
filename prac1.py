import pandas as pd
import numpy as np

# Генерація великого набору даних
num_rows = 100_000_000
data = {
    'id': np.arange(num_rows, dtype='int64'),               # int64 за замовчуванням
    'price': np.random.uniform(10.0, 10000.0, num_rows),    # float64 за замовчуванням
    'quantity': np.random.randint(1, 100, num_rows),        # int64
    'discount': np.random.uniform(0.0, 0.5, num_rows)       # float64
}

df = pd.DataFrame(data)

# Розрахунок пам'яті до оптимізації
mem_before = df.memory_usage(deep=True).sum() / (1024**2)

# Фіксуємо середнє значення price у точності float64
mean_64 = df['price'].mean()

# Перетворення числових колонок у float32 та int32
df_optimized = df.copy()
df_optimized['id'] = df_optimized['id'].astype('int32')
df_optimized['price'] = df_optimized['price'].astype('float32')
df_optimized['quantity'] = df_optimized['quantity'].astype('int32')
df_optimized['discount'] = df_optimized['discount'].astype('float32')

# Розрахунок пам'яті після оптимізації
mem_after = df_optimized.memory_usage(deep=True).sum() / (1024**2)

# Фіксуємо середнє значення price у точності float32
mean_32 = df_optimized['price'].mean()

# Розрахунок похибки
error = abs(mean_64 - mean_32)

print(f"--- Варіант 14 ---")
print(f"Обсяг пам'яті (float64/int64): {mem_before:.2f} MB")
print(f"Обсяг пам'яті (float32/int32): {mem_after:.2f} MB")
print(f"Економія пам'яті: {((mem_before - mem_after) / mem_before) * 100:.1f}%")
print(f"\nСереднє price (float64): {mean_64}")
print(f"Середнє price (float32): {mean_32}")
print(f"Похибка середнього: {error:.10f}")