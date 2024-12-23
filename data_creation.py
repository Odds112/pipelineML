import os
import numpy as np
import pandas as pd

os.makedirs("train", exist_ok=True)
os.makedirs("test", exist_ok=True)

# Функция для генерации синтетических данных о температуре
def generate_temperature_data(days, anomaly_rate=0.05, noise_std=1.0, seed=None):
    if seed is not None:
        np.random.seed(seed)

    x = np.arange(days)
    base_temperature = 20 + 10 * np.sin(2 * np.pi * x / 365)

    noise = np.random.normal(0, noise_std, size=days)
    temperature = base_temperature + noise

    num_anomalies = int(days * anomaly_rate)
    anomaly_indices = np.random.choice(days, num_anomalies, replace=False)
    temperature[anomaly_indices] += np.random.uniform(-15, 15, size=num_anomalies)

    return pd.DataFrame({"Day": x, "Temperature": temperature})

# Генерация данных для обучающей выборки
for i in range(5):
    data = generate_temperature_data(days=365, anomaly_rate=0.05, noise_std=1.5, seed=i)
    data.to_csv(f"train/train_data_{i}.csv", index=False)

# Генерация данных для тестовой выборки
for i in range(3):
    data = generate_temperature_data(days=365, anomaly_rate=0.1, noise_std=2.0, seed=i+100)
    data.to_csv(f"test/test_data_{i}.csv", index=False)

print("Данные успешно сгенерированы и сохранены в папки 'train' и 'test'.")