# AI Project Framework Advisor

Учебный LangChain-агент, который анализирует требования ИИ-проекта, сравнивает фреймворки и рекомендует подходящий вариант.

`Python 3.12+` `LangChain` `81 tests passed` `Educational project`

## О проекте

Агент принимает текстовое описание ИИ-проекта и помогает выбрать фреймворк для его построения. Он:

- рассматривает пять фреймворков: LangChain, LlamaIndex, Haystack, Semantic Kernel и CrewAI;
- использует локальную проверяемую базу характеристик фреймворков, а не догадки модели;
- вызывает собственные инструменты для получения фактов;
- возвращает структурированную рекомендацию (Pydantic-схема), а не произвольный текст;
- поддерживает краткосрочную память диалога в рамках запущенного процесса.

Проект создан как дополнительная часть домашнего задания по фреймворкам управления памятью и контекстом.

## Возможности

- рекомендации по выбору фреймворка под описанный проект;
- явное сравнение нескольких вариантов (поле `comparison` в ответе);
- получение отдельного профиля одного фреймворка;
- два собственных инструмента:
  - `get_framework_profile` — профиль одного фреймворка;
  - `compare_frameworks` — сравнение от двух до пяти фреймворков по выбранным критериям;
- структурированный ответ через Pydantic (`AdvisorResponse`);
- краткосрочная память через `InMemorySaver` и `thread_id`;
- команды CLI `/new`, `/exit`, `/quit`;
- обработка ошибок структурированного ответа без вывода traceback пользователю;
- автоматические тесты, не выполняющие реальных API-вызовов.

## Поддерживаемые фреймворки

| Фреймворк | Основной фокус |
| --- | --- |
| LangChain | Агенты, инструменты и многошаговые сценарии |
| LlamaIndex | Документы, собственные данные и RAG |
| Haystack | Компонентные поисковые и RAG-конвейеры |
| Semantic Kernel | ИИ-функции, плагины и Microsoft-экосистема |
| CrewAI | Команды специализированных агентов и управляемые процессы |

Ни один из вариантов не является универсально лучшим — выбор зависит от конкретного проекта.

## Архитектура

```text
ai-project-framework-advisor/
├── src/advisor/
│   ├── framework_data.py
│   ├── tools.py
│   ├── schema.py
│   ├── agent.py
│   └── cli.py
├── tests/
├── docs/evidence/
├── .env.example
├── pyproject.toml
└── README.md
```

- `framework_data.py` — локальный словарь с профилями пяти фреймворков, критериями сравнения и их подписями. Единый источник данных для инструментов.
- `tools.py` — два собственных инструмента (`get_framework_profile`, `compare_frameworks`), нормализация названий фреймворков и критериев.
- `schema.py` — Pydantic-модели `AdvisorResponse` и `FrameworkComparison`, описывающие структуру финального ответа агента.
- `agent.py` — сборка агента через `create_agent`: модель, инструменты, системный промпт, `response_format`, `checkpointer`.
- `cli.py` — консольный интерфейс: цикл диалога, форматирование ответа, обработка команд и ошибок.

## Как работает агент

1. Пользователь описывает проект в консоли.
2. Агент определяет разумных кандидатов среди поддерживаемых фреймворков.
3. Агент вызывает `get_framework_profile` (один фреймворк) или `compare_frameworks` (несколько вариантов).
4. Инструменты получают данные только из `framework_data.py`, не выдумывая характеристики.
5. Модель формирует ответ в виде `AdvisorResponse`.
6. CLI выводит рекомендацию, сравнение вариантов (если есть), причины, подходящие компоненты и риски.
7. `InMemorySaver` сохраняет состояние диалога по `thread_id` в рамках текущего процесса.

`InMemorySaver` не является долгосрочной памятью и не сохраняет историю после завершения программы.

## Структурированный ответ

Упрощённый пример `AdvisorResponse`:

```json
{
  "recommended_framework": "llamaindex",
  "comparison": [
    {
      "framework": "langchain",
      "strengths": ["..."],
      "limitations": ["..."],
      "fit_for_project": "..."
    }
  ],
  "reasoning": ["..."],
  "suitable_components": ["..."],
  "risks": ["..."]
}
```

## Установка

```powershell
git clone https://github.com/eliv1982/ai-project-framework-advisor.git
cd ai-project-framework-advisor
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Настройка переменных окружения

```powershell
Copy-Item .env.example .env
```

Содержимое `.env`:

```dotenv
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4.1-mini
```

Не публикуйте `.env`. Не добавляйте API-ключ в README, скриншоты или коммиты. Файл `.env` уже исключён через `.gitignore`.

## Запуск

```powershell
framework-advisor
```

Пример запроса в диалоге:

```text
Сравни LangChain, LlamaIndex и Haystack для создания юридического RAG-ассистента. Сравни их по поддержке RAG, порогу входа и ограничениям, а затем порекомендуй один вариант.
```

Команды интерфейса:

- `/new` — начать новый диалог;
- `/exit` или `/quit` — завершить программу.

## Тестирование

```powershell
python -m pytest -q
python -m pip check
```

Текущий результат:

```text
81 passed
No broken requirements found.
```

## Демонстрация

![Рекомендация для юридического RAG-ассистента](docs/evidence/01_legal_rag_recommendation.png)

Рекомендация для юридического RAG-ассистента.

![Продолжение диалога с учетом предыдущего контекста](docs/evidence/02_followup_framework_comparison.png)

Продолжение диалога с учетом предыдущего контекста.

![Новый диалог и рекомендация CrewAI для многоагентного сценария](docs/evidence/03_new_dialog_multiagent_recommendation.png)

Новый диалог и рекомендация CrewAI для многоагентного сценария.

![Явное сравнение LangChain, LlamaIndex и Haystack](docs/evidence/04_agent_framework_comparison.png)

Явное сравнение LangChain, LlamaIndex и Haystack.

![Отдельный профиль Semantic Kernel](docs/evidence/05_agent_framework_profile.png)

Отдельный профиль Semantic Kernel.

![Результаты автоматических тестов и проверки зависимостей](docs/evidence/06_test_suite.png)

Результаты автоматических тестов и проверки зависимостей.

## Учебные ограничения

- характеристики фреймворков хранятся в локальном словаре и не обновляются автоматически;
- `InMemorySaver` работает только в рамках процесса;
- CLI рассчитан на одного локального пользователя;
- нет постоянной базы истории диалогов;
- нет повторных попыток при сетевых ошибках;
- нет LangSmith или другой производственной наблюдаемости;
- рекомендации агента требуют человеческой проверки перед архитектурным решением.

## Технологии

- Python 3.12+
- LangChain
- LangGraph
- langchain-openai
- Pydantic 2
- python-dotenv
- pytest
- Hatchling
