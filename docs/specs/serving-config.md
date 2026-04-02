## Serving / Config

## Состав сервинга

Текущий runtime состоит из двух процессов:

- Streamlit UI/симулятор (`pizzeria_sim.py`);
- FastAPI-сервис агента (`app_run.py`).

UI обращается к backend по HTTP, по умолчанию на `http://localhost:8001/analyze`.

## Backend

### Приложение

- Framework: FastAPI
- Entry point: `app_run.py`
- Health endpoint: `GET /health`
- Основной endpoint: `POST /analyze`
- Endpoint подтверждения действий: `POST /action`

### Сетевые параметры

- `LLM_SERVICE_HOST`, по умолчанию `0.0.0.0`
- `LLM_SERVICE_PORT`, по умолчанию `8001`

### Таймауты

- полный анализ `/analyze`: 90 секунд;
- timeout одного LLM вызова: 120 секунд;

## Конфигурация модели

Конфигурация читается через Dynaconf из `config.json`.

Используемые поля:
- `agent.model_name`
- `agent.base_url`
- `agent.api_key`
- `agent.temperature`
- `tool_timeout_sec`
- `tool_retry_count`
- `max_tool_calls`
- `agent.system_prompt`

LLM-клиент создается как `ChatOpenAI(...)`, то есть backend ожидает OpenAI-compatible API.

## Секреты

Текущее состояние MVP:
- `api_key` хранится в `config.json`.

Это допустимо только для локального прототипа

## Версия модели

Текущая конфигурация в репозитории использует:
- `openai/gpt-oss-120b`
- base URL `https://integrate.api.nvidia.com/v1`

Архитектурно сервис не привязан к конкретному вендору, пока API совместим с OpenAI-форматом.

## Запуск

Минимальный путь запуска:

1. Поднять FastAPI backend - запустить `python app_run.py`
2. Поднять Streamlit UI - запустить `streamlit run pizzeria_sim.py`
3. Убедиться, что UI смотрит на корректный `LLM_SERVICE_URL`

## Dependency на данные

Для корректной работы backend ожидает наличие:
- `data/pre_data.xlsx`
- `data/weather.xlsx`
- `data/schedule.json`
- `data/c_schedule.json`
- `data/stores.xlsx`

Для корректной работы UI ожидает наличие:
- `pizzeria.db` или импортируемого `data.xlsx`

