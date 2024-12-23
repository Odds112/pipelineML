import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Функция для загрузки предобработанных данных
def load_preprocessed_data(file_path):
    return pd.read_csv(file_path)


if __name__ == "__main__":
    # Загрузка данных
    print("Загрузка предобработанных обучающих данных...")
    train_data = load_preprocessed_data("preprocessed_train_data.csv")

    # Определение признаков и целевой переменной
    X = train_data[["Day"]]
    y = train_data["Temperature"]

    # Разделение данных на обучающую и валидационную выборки
    print("Разделение данных на обучающую и валидационную выборки...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Создание и обучение модели
    print("Создание и обучение модели RandomForestRegressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Оценка модели
    print("Оценка модели на валидационной выборке...")
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    print(f"Среднеквадратичная ошибка (MSE) на валидационной выборке: {mse}")

    # Сохранение обученной модели
    print("Сохранение модели...")
    joblib.dump(model, "temperature_model.pkl")

    print("Модель успешно создана, обучена и сохранена в 'temperature_model.pkl'.")