import pandas as pd
from sklearn.metrics import mean_squared_error
import joblib

# Функция для загрузки предобработанных данных
def load_preprocessed_data(file_path):
    return pd.read_csv(file_path)

# Основной блок скрипта
if __name__ == "__main__":
    # Загрузка тестовых данных
    print("Загрузка предобработанных тестовых данных...")
    test_data = load_preprocessed_data("preprocessed_test_data.csv")

    # Определение признаков и целевой переменной
    X_test = test_data[["Day"]]
    y_test = test_data["Temperature"]

    # Загрузка модели
    print("Загрузка обученной модели...")
    model = joblib.load("temperature_model.pkl")

    # Прогнозирование
    print("Прогнозирование на тестовых данных...")
    y_pred = model.predict(X_test)

    # Оценка модели
    print("Оценка модели на тестовых данных...")
    mse = mean_squared_error(y_test, y_pred)
    print(f"Среднеквадратичная ошибка (MSE) на тестовых данных: {mse}")

    # Сохранение результатов
    print("Сохранение результатов прогнозирования...")
    test_data["Predicted_Temperature"] = y_pred
    test_data.to_csv("test_results.csv", index=False)

    print("Результаты успешно сохранены в 'test_results.csv'.")