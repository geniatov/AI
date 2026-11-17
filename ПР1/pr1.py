"""Практическое занятие № 1.
Задача фильтрации спама методом байесовского классификатора.
"""

import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


# Предобработка текста
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)  # Удаляем числа
    text = ' '.join(text.split())
    return text


# Расширенные данные с большим количеством примеров
data = {
    'text': [
        # СПАМ сообщения (1)
        "Поздравляем, вы выиграли крупный приз! Перейдите по ссылке.",
        "Срочно обновите пароль для вашего аккаунта.",
        "Уникальное предложение, только для вас! Скидка 90%",
        "Ваш банковский счет заблокирован, кликните здесь.",
        "Срочное уведомление о безопасности вашей учетной записи.",
        "Получите деньги немедленно, нажмите!",
        "Вы выиграли миллион! Заберите приз сейчас!",
        "Только сегодня! Супер предложение!",
        "Ваша учетная запись будет удалена. Подтвердите email.",
        "Заработай быстро! Переходи по ссылке!",

        # HAM сообщения (0)
        "Встреча переносится на завтра в 10 утра.",
        "Проект отчета готов, жду вашего мнения.",
        "Напоминаю о необходимости сдачи курсовой работы.",
        "Прошу подтвердить получение документации.",
        "Добрый день, пожалуйста, посмотрите приложенный файл.",
        "Как дела? Давай пообедаем на этой неделе.",
        "Завтра собрание в 14:00 в конференц-зале.",
        "Отправь мне отчет до конца дня.",
        "Спасибо за помощь вчера!",
        "Когда будет готов новый дизайн?",
        "Не забудь оплатить интернет до пятницы.",
        "Пришлите пожалуйста контакты того человека."
    ],
    'label': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # 10 спам
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 12 не спам
}
df = pd.DataFrame(data)

# Применяем предобработку
df['text_clean'] = df['text'].apply(preprocess_text)

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(
    df['text_clean'], df['label'], test_size=0.3, random_state=42,
    stratify=df['label']
)

print("\n")
print("-" * 4, "РАСПРЕДЕЛЕНИЕ ПО ВЫБОРКАМ", "-" * 4)
print(f"Обучающая выборка - всего: {len(y_train)}")
print(f"  HAM: {sum(y_train == 0)} ({sum(y_train == 0) / len(y_train) * 100:.1f}%)")
print(f"  SPAM: {sum(y_train == 1)} ({sum(y_train == 1) / len(y_train) * 100:.1f}%)")

print(f"\nТестовая выборка - всего: {len(y_test)}")
print(f"  HAM: {sum(y_test == 0)} ({sum(y_test == 0) / len(y_test) * 100:.1f}%)")
print(f"  SPAM: {sum(y_test == 1)} ({sum(y_test == 1) / len(y_test) * 100:.1f}%)")

# Создаем пайплайн с TF-IDF и Naive Bayes
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        token_pattern=r'(?u)\b\w+\b',
        min_df=1,
        max_df=0.8,
        ngram_range=(1, 2),  # Используем униграммы и биграммы
        sublinear_tf=True
    )),
    ('nb', MultinomialNB(alpha=0.5))
])

# Обучение
pipeline.fit(X_train, y_train)

# Предсказание
y_pred = pipeline.predict(X_test)

print("\n")
print("-" * 4, "РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ", "-" * 4)
print("Матрица ошибок:")
print(confusion_matrix(y_test, y_pred))
print("\nОтчет о классификации:")
print(classification_report(y_test, y_pred, target_names=['HAM', 'SPAM']))

# Тестирование на новых сообщениях
print("-" * 4, "ТЕСТИРОВАНИЕ НА НОВЫХ СООБЩЕНИЯХ", "-" * 4)

new_messages = [
    "Ваш заказ успешно обработан и будет доставлен завтра.",
    "Секретная информация доступна здесь!! Переходи быстро!",
]

new_messages_clean = [preprocess_text(msg) for msg in new_messages]
new_pred = pipeline.predict(new_messages_clean)
new_proba = pipeline.predict_proba(new_messages_clean)

print("Результаты:")
for msg, pred, proba in zip(new_messages, new_pred, new_proba):
    label = "SPAM" if pred == 1 else "HAM"
    confidence = proba[pred] * 100
    print(f"\nСообщение: '{msg}'")
    print(f"  Предсказано: {label}")
    print(f"  Уверенность: {confidence:.1f}%")
    print(f"  Вероятности: HAM={proba[0]:.3f}, SPAM={proba[1]:.3f}")

# Ручная проверка граничных случаев
print("\n")
print("-" * 4, "ПРОВЕРКА ГРАНИЧНЫХ СЛУЧАЕВ", "-" * 4)

test_cases = [
    "срочно обнови пароль",  # Должен быть спам
    "встреча завтра",  # Должен быть ham
    "выиграй приз",  # Должен быть спам
    "спасибо за помощь"  # Должен быть ham
]

test_clean = [preprocess_text(tc) for tc in test_cases]
test_pred = pipeline.predict(test_clean)

for tc, pred in zip(test_cases, test_pred):
    print(f"'{tc}' -> {'SPAM' if pred == 1 else 'HAM'}")
