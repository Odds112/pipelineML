#!/bin/bash

# Step 0: Установка зависимостей
echo "Установка зависимостей..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Ошибка при установке зависимостей."
    exit 1
fi

# Step 1: Генерация данных
echo "Генерация данных..."
python3 data_creation.py
if [ $? -ne 0 ]; then
    echo "Ошибка при генерации данных."
    exit 1
fi

# Step 2: Предобработка данных
echo "Предобработка данных..."
python3 model_preprocessing.py
if [ $? -ne 0 ]; then
    echo "Ошибка при предобработке данных."
    exit 1
fi

# Step 3: Обучение модели
echo "Обучение модели..."
python3 model_preparation.py
if [ $? -ne 0 ]; then
    echo "Ошибка при обучении модели."
    exit 1
fi

# Step 4: Тестирование модели
echo "Тестирование модели..."
output=$(python3 model_testing.py 2>&1)
if [ $? -ne 0 ]; then
    echo "Ошибка при тестировании модели."
    exit 1
fi

# Извлечение метрики MSE из вывода
mse=$(echo "$output" | grep "Среднеквадратичная ошибка (MSE)" | awk '{print $6}')
if [ -z "$mse" ]; then
    echo "Ошибка: не удалось извлечь метрику."
    exit 1
fi

# Вывод метрики
echo "Model test MSE is: $mse"
