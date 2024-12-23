import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Функция для загрузки данных из директории
def load_data_from_directory(directory):
    data_frames = []
    for file_name in os.listdir(directory):
        if file_name.endswith(".csv"):
            file_path = os.path.join(directory, file_name)
            df = pd.read_csv(file_path)
            data_frames.append(df)
    return pd.concat(data_frames, ignore_index=True)

# Функция для предобработки данных с использованием StandardScaler
def preprocess_data(data, features):
    scaler = StandardScaler()
    data[features] = scaler.fit_transform(data[features])
    return data, scaler

# Основной блок скрипта
if __name__ == "__main__":
    # Пути к директориям с данными
    train_dir = "train"
    test_dir = "test"

    # Загрузка данных
    print("Загрузка обучающих данных...")
    train_data = load_data_from_directory(train_dir)

    print("Загрузка тестовых данных...")
    test_data = load_data_from_directory(test_dir)

    # Предобработка данных
    print("Выполнение предобработки данных...")
    features_to_scale = ["Temperature"]

    train_data, scaler = preprocess_data(train_data, features_to_scale)
    test_data[features_to_scale] = scaler.transform(test_data[features_to_scale])

    # Сохранение предобработанных данных
    train_data.to_csv("preprocessed_train_data.csv", index=False)
    test_data.to_csv("preprocessed_test_data.csv", index=False)

    print("Предобработка завершена. Данные сохранены в 'preprocessed_train_data.csv' и 'preprocessed_test_data.csv'.")