Track: B

# Telegram Sentiment AI Agent

AI-агент для анализа тональности текста, который использует классическую ML-модель как инструмент и отправляет результат в Telegram.

---

## Описание

Проект демонстрирует работу AI-агента, который:

1. принимает текст от пользователя (через n8n Chat)
2. вызывает локальный ML-инструмент (`predict.py`)
3. определяет тональность текста (positive / negative)
4. использует LLM для формирования ответа
5. отправляет результат в Telegram

---

## Инструменты агента

Агент использует два инструмента:

### 1. ML Tool (`predict.py`)
- Загружает обученную модель
- Возвращает:
  
  ```json
  {"label": "positive", "confidence": 0.73}
  ```

### 2. LLM (OpenRouter / OpenAI)
- Интерпретирует результат модели
- Формирует ответ пользователю

---

## Стек

- Python
- scikit-learn
- joblib
- n8n
- OpenRouter / OpenAI API
- Telegram Bot API

---

## Модель

Используется классический pipeline:

- `TfidfVectorizer`
- `LogisticRegression`

Модель обучена на небольшом датасете из 300 текстов:
- 150 positive
- 150 negative

Accuracy: 0.85
---

## Данные

Файл:

data/sentiment_sample.csv

Датасет синтетический, создан вручную для демонстрации.

---

## Запуск проекта

### 1. Клонировать репозиторий

git clone <your-repo-url>  
cd telegram_sentiment-agent

### 2. Установить зависимости и запустить проект

python run.py

Команда автоматически:
- установит зависимости
- обучит ML-модель
- запустит n8n

### 3. Открыть n8n

http://localhost:5678

### 4. Импортировать workflow

- Нажмите Import
- Выберите файл:

n8n/workflow.json

### 5. Настроить credentials

Добавьте:
- OpenRouter / OpenAI API key
- Telegram Bot Token

### 6. Запустить workflow

- Откройте workflow
- Нажмите Execute
- Введите текст в Chat

### 7. Получить результат

Агент:
- определит тональность текста
- сформирует ответ
- отправит сообщение в Telegram

---