# Vector Search RAG

Система векторного поиска с RAG для поиска информации в документах.

## Быстрый запуск

```bash
# Запуск через Docker (рекомендуется)
docker-compose up -d

# Доступ
# Веб-интерфейс: http://localhost:8000
# ChromaDB: http://localhost:8001
```

## Стек

- **Backend**: FastAPI, ChromaDB, Sentence Transformers
- **Frontend**: HTML5/CSS3/JavaScript
- **DevOps**: Docker, Docker Compose

## Функциональность

- Векторный поиск по документам
- Загрузка .txt файлов через веб
- Чат-интерфейс для взаимодействия
- Docker-контейнеризация

## Структура

```
vector-search-rag/
├── app/                 # Приложение
│   ├── main.py         # FastAPI эндпоинты
│   ├── templates/       # HTML шаблоны
│   └── static/          # CSS/JS файлы
├── data/               # Данные
├── docker-compose.yml   # Docker конфигурация
├── requirements.txt    # Зависимости
└── README.md
```

## Команды

```bash
# Управление контейнерами
docker-compose up -d    # Запустить
docker-compose down     # Остановить
docker-compose logs -f  # Логи
docker-compose restart  # Перезапустить

# Полная пересборка
docker-compose down -v && docker-compose up --build -d
```