import random
import string
from collections import Counter

import numpy as np

# 1. ШИФР ЦЕЗАРЯ (БАЗОВАЯ ФУНКЦИЯ)
ALPHABET = string.ascii_lowercase
ALPHABET_SIZE = len(ALPHABET)


def caesar_cipher(text, key, mode='encrypt'):
    result = ""
    shift = key if mode == 'encrypt' else -key
    for char in text.lower():
        if char in ALPHABET:
            new_index = (ALPHABET.index(char) + shift) % ALPHABET_SIZE
            result += ALPHABET[new_index]
        else:
            result += char
    return result


# 2. ФУНКЦИЯ ПРИСПОСОБЛЕННОСТИ (FITNESS)
# Используем стандартные частоты английского языка
# (Источник: Wikipedia/стандартные частоты)
ENGLISH_FREQS = {
    'e': 12.70, 't': 9.06, 'a': 8.17, 'o': 7.51, 'i': 6.97, 'n': 6.75,
    's': 6.33, 'h': 6.09, 'r': 5.99, 'd': 4.25, 'l': 4.03, 'c': 2.78,
    'u': 2.76, 'm': 2.41, 'w': 2.36, 'f': 2.23, 'g': 2.02, 'y': 1.97,
    'p': 1.93, 'b': 1.29, 'v': 0.98, 'k': 0.77, 'j': 0.15, 'x': 0.15,
    'q': 0.10, 'z': 0.07
}


def calculate_fitness(ciphertext, key, expected_freqs):
    # Декодируем текст с текущим ключом
    decoded_text = caesar_cipher(ciphertext, key, 'decrypt')

    # Подсчет частот только букв, попавших в наш алфавит
    text_letters = [char for char in decoded_text if char in ALPHABET]
    if not text_letters:
        return 0

    total_letters = len(text_letters)
    actual_counts = Counter(text_letters)

    fitness_score = 0
    # Считаем сумму абсолютных разностей между ожидаемой и фактической частотой
    for char, expected_freq in expected_freqs.items():
        actual_freq = (actual_counts.get(char, 0) / total_letters) * 100
        fitness_score += abs(actual_freq - expected_freq)

    # Чем меньше общая ошибка, тем выше приспособленность (инвертируем для ГА)
    return 10000.0 / (fitness_score + 1e-6)  # Константа против деления на ноль


# 3. Основные компоненты ГА
POPULATION_SIZE = 100
MAX_GENERATIONS = 100
MUTATION_RATE = 0.1


def initialize_population():
    # Каждая особь — это ключ (целое число от 1 до 25)
    return [random.randint(1, ALPHABET_SIZE - 1) for _ in range(POPULATION_SIZE)]


def crossover(parent1, parent2):
    # Одноточечный кроссовер для простого ключа
    return parent2, parent1


def mutate(key):
    if random.random() < MUTATION_RATE:
        # Мутация: сдвиг ключа на ±1 в пределах [1, 25]
        return (key - 1 + random.choice([-1, 1])) % (ALPHABET_SIZE - 1) + 1
    return key


# 4. ЗАПУСК АТАКИ
random.seed(404)
np.random.seed(404)

# Исходный текст и его шифрование (ключ случайный, но воспроизводимый)
PLAINTEXT = "london is the capital of great britain"
SECRET_KEY = random.randint(1, 25)
ciphertext = caesar_cipher(PLAINTEXT, SECRET_KEY, 'encrypt')
print(f"Исходный текст (Англ): {PLAINTEXT}")
print(f"Зашифрованный текст: {ciphertext}")

population = initialize_population()
best_key = None
best_fitness = -float('inf')

for generation in range(MAX_GENERATIONS):

    # 1. Расчет приспособленности
    fitnesses = [calculate_fitness(ciphertext, key, ENGLISH_FREQS) for key in population]

    # Обновление лучшего решения
    current_best_fitness = max(fitnesses)
    if current_best_fitness > best_fitness:
        best_fitness = current_best_fitness
        best_key = population[np.argmax(fitnesses)]

    # 2. Селекция (отбор лучших 50% для скрещивания)
    sorted_indices = np.argsort(fitnesses)[::-1]  # Сортируем по убыванию fitness
    elite_count = POPULATION_SIZE // 2
    parents_indices = sorted_indices[:elite_count]

    new_population = []
    # Сохраняем лучшего
    new_population.append(best_key)

    # 3. Скрещивание и мутация
    for i in range(POPULATION_SIZE - 1):
        p1_idx = random.choice(parents_indices)
        p2_idx = random.choice(parents_indices)

        parent1 = population[p1_idx]
        parent2 = population[p2_idx]

        child_key, _ = crossover(parent1, parent2)
        mutated_child = mutate(child_key)

        # Ключ должен быть в диапазоне [1, 25]
        new_population.append(mutated_child)

    population = new_population

    if generation % 20 == 0:
        print(f"Gen {generation}: Best Fitness={best_fitness:.2f}, Key={best_key}")

# 5. Финальный результат
print("\nВЗЛОМ ЗАВЕРШЕН")
print(f"Найденный ключ: {best_key}")
print(f"Расшифрованный текст: {caesar_cipher(ciphertext, best_key, 'decrypt')}")
print(f"Истинный ключ: {SECRET_KEY}")
print(f"Ключ найден верно: {'да' if best_key == SECRET_KEY else 'нет'}")
