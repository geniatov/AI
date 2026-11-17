import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Установка сида
tf.random.set_seed(303)
tf.keras.utils.set_random_seed(303)

# 1. Загрузка и подготовка данных MNIST
# MNIST (рукописные цифры) используется как простая модель для демонстрации CNN
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Преобразование данных: (N, 28, 28) -> (N, 28, 28, 1) для CNN
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')

# Нормализация: пиксели 0-255 -> 0.0-1.0
X_train /= 255
X_test /= 255

# One-Hot Encoding (10 классов: цифры от 0 до 9)
y_train_cat = to_categorical(y_train, num_classes=10)
y_test_cat = to_categorical(y_test, num_classes=10)

print(f"Обучающие данные: {X_train.shape}, Метки: {y_train_cat.shape}")
print(f"Тестовые данные: {X_test.shape}, Метки: {y_test_cat.shape}")

# 2. Построение CNN
model_cnn_bio = Sequential([
    # Блок 1
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),

    # Блок 2
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    # Классификационная часть
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # 10 классов
])

model_cnn_bio.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
model_cnn_bio.summary()

# 3. Обучение модели
print("\nОбучение CNN для биометрической идентификации")
history = model_cnn_bio.fit(
    X_train, y_train_cat,
    epochs=10,
    batch_size=64,
    validation_data=(X_test, y_test_cat),
    verbose=1
)

# 4. Тестирование
print("\nОценка на тестовом наборе")
test_loss, test_acc = model_cnn_bio.evaluate(X_test, y_test_cat, verbose=0)
print(f"Точность на тестовом наборе: {test_acc:.4f}")
print(f"Потери на тестовом наборе: {test_loss:.4f}")

# 5. Демонстрация работы (проверка нескольких образцов)
test_count = 5
predictions = model_cnn_bio.predict(X_test[:test_count], verbose=0)
print(f"\nПроверка первых {test_count} тестовых образцов")
for i in range(test_count):
    predicted_class = np.argmax(predictions[i])
    true_class = y_test[i]
    print(f"Образец {i + 1}: Истинный класс={true_class}, "
          f"Распознано={predicted_class} "
          f"({'CORRECT' if predicted_class == true_class else 'INCORRECT'})")
