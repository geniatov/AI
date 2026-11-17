import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Установка сида
np.random.seed(202)
tf.random.set_seed(202)
tf.keras.utils.set_random_seed(202)

# 1. Генерация реалистичных данных
N_SAMPLES = 600
N_FEATURES = 5

# Генерируем данные с перекрытием
X_normal = np.random.randn(N_SAMPLES // 2, N_FEATURES) * 1.5 + 2
X_malicious = np.random.randn(N_SAMPLES // 2, N_FEATURES) * 1.8 + 4
X_malicious[:, 1] = np.abs(X_malicious[:, 1]) * 1.5  # Нелинейность

# Добавляем выбросы
outliers_normal = np.random.randn(10, N_FEATURES) * 3 + 5
outliers_malicious = np.random.randn(10, N_FEATURES) * 3 + 1

X_normal = np.vstack([X_normal, outliers_normal])
X_malicious = np.vstack([X_malicious, outliers_malicious])

# Добавляем шум
X_normal += np.random.normal(0, 0.3, X_normal.shape)
X_malicious += np.random.normal(0, 0.4, X_malicious.shape)

X = np.vstack([X_normal, X_malicious]).astype(np.float32)
y = np.array([0] * len(X_normal) + [1] * len(X_malicious))

# Перемешиваем данные
shuffle_idx = np.random.permutation(len(X))
X = X[shuffle_idx]
y = y[shuffle_idx]

# 2. Масштабирование данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=202, stratify=y
)

print(f"Всего образцов: {len(X)} (Normal: {sum(y == 0)}, Malicious: {sum(y == 1)})")
print(f"Обучающая выборка: {len(y_train)}, тестовая: {len(y_test)}")

# 3. Однослойный персептрон (эквивалент логистической регрессии)
model_single = Sequential([
    Dense(1, activation='sigmoid', input_shape=(N_FEATURES,))
])
model_single.compile(optimizer='adam', loss='binary_crossentropy',
                     metrics=['accuracy'])

print("\n--- Обучение Однослойной модели ---")
model_single.fit(X_train, y_train, epochs=50, verbose=0)
loss, acc_single = model_single.evaluate(X_test, y_test, verbose=0)
print(f"Точность однослойной модели: {acc_single:.4f}")

# 4. Многослойный персептрон (MLP)
model_mlp = Sequential([
    Dense(32, activation='relu', input_shape=(N_FEATURES,)),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
model_mlp.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])

print("\n--- Обучение Многослойной модели (MLP) ---")
history_mlp = model_mlp.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    verbose=0
)
loss_mlp, acc_mlp = model_mlp.evaluate(X_test, y_test, verbose=0)
print(f"Точность MLP модели: {acc_mlp:.4f}")

# 5. Сравнение и отчет
y_pred_single = (model_single.predict(X_test, verbose=0) > 0.5).astype("int32")
y_pred_mlp = (model_mlp.predict(X_test, verbose=0) > 0.5).astype("int32")

print("\n--- Отчет однослойной модели ---")
print(classification_report(y_test, y_pred_single,
                            target_names=['Normal', 'Malicious']))
print("Матрица ошибок:")
print(confusion_matrix(y_test, y_pred_single))

print("\n--- Отчет MLP ---")
print(classification_report(y_test, y_pred_mlp,
                            target_names=['Normal', 'Malicious']))
print("Матрица ошибок:")
print(confusion_matrix(y_test, y_pred_mlp))

print("\n--- Сравнение моделей ---")
print(f"Однослойный персептрон: {acc_single:.4f}")
print(f"Многослойный персептрон: {acc_mlp:.4f}")
print(f"Прирост точности: {(acc_mlp - acc_single) * 100:+.2f} п.п.")
