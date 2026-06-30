"""Сборка ИИ-агента-консультанта по выбору фреймворка.

Этот модуль только конструирует агента через create_agent (модель,
инструменты, системный промпт, формат ответа и checkpointer) и не
выполняет никаких вызовов модели и не имеет побочных эффектов.
"""

from __future__ import annotations

import os
from typing import Any

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from advisor.schema import AdvisorResponse
from advisor.tools import compare_frameworks, get_framework_profile

SYSTEM_PROMPT = """Ты — агент-консультант, который помогает выбрать фреймворк для построения ИИ-проекта на основе LLM.

Поддерживаемые варианты (используй только эти канонические ключи):
- langchain
- llamaindex
- haystack
- semantic_kernel
- crewai

Источник фактов: используй только сведения, полученные через доступные инструменты get_framework_profile и compare_frameworks. Не придумывай характеристики фреймворков, которых нет в данных, возвращённых инструментами.

Использование инструментов:
- вызывай get_framework_profile, когда пользователь спрашивает об одном конкретном фреймворке;
- вызывай compare_frameworks, когда нужно выбрать фреймворк для проекта или сравнить несколько вариантов;
- для итоговой рекомендации по проекту сначала определи как минимум двух разумных кандидатов и сравни их по релевантным критериям через compare_frameworks;
- не вызывай инструменты повторно без необходимости — если нужные данные уже получены в этом диалоге, используй их повторно.

Критерии выбора: учитывай задачу проекта, необходимость RAG, поддержку агентов, потребность в многоагентной системе, важность явного и предсказуемого конвейера, экосистему интеграций, порог входа команды и известные ограничения фреймворков.

Формат финального ответа должен соответствовать схеме AdvisorResponse:
- recommended_framework — ровно один канонический ключ из списка выше;
- reasoning — конкретные причины выбора, основанные на данных инструментов;
- suitable_components — реальные компоненты выбранного фреймворка (из его профиля), которые подходят проекту;
- risks — реальные ограничения и компромиссы рекомендации.

Язык: формулируй reasoning, suitable_components и risks на русском языке, за исключением официальных названий компонентов и фреймворков (например, "LangGraph", "Crews", "Workflows").

Недостаточные данные: если пользователь описал проект не полностью, не выдумывай отсутствующие требования. Дай осторожную предварительную рекомендацию на основе того, что уже известно, и явно отметь неопределённость в reasoning или risks.

Не утверждай заранее, что какой-то один фреймворк универсально лучше остальных вне контекста конкретного проекта."""


def build_advisor_agent(
    *,
    api_key: str | None = None,
    model_name: str | None = None,
    checkpointer: Any | None = None,
) -> Any:
    """Собирает агента-консультанта через create_agent.

    api_key и model_name берутся из переданных аргументов, а при их
    отсутствии (None) — из переменных окружения OPENAI_API_KEY и
    OPENAI_MODEL соответственно. Не выполняет invoke и не обращается
    к сети — только конструирует объект агента.
    """
    resolved_api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
    resolved_model_name = (
        model_name if model_name is not None else os.getenv("OPENAI_MODEL")
    )

    resolved_api_key = "" if resolved_api_key is None else resolved_api_key.strip()
    resolved_model_name = (
        "" if resolved_model_name is None else resolved_model_name.strip()
    )

    if not resolved_api_key:
        raise ValueError(
            "Не задан OPENAI_API_KEY: передайте api_key явно или установите "
            "переменную окружения OPENAI_API_KEY."
        )

    if not resolved_model_name:
        raise ValueError(
            "Не задано имя модели: передайте model_name явно или установите "
            "переменную окружения OPENAI_MODEL."
        )

    model = ChatOpenAI(
        model=resolved_model_name,
        api_key=resolved_api_key,
    )

    resolved_checkpointer = (
        checkpointer if checkpointer is not None else InMemorySaver()
    )

    return create_agent(
        model=model,
        tools=[get_framework_profile, compare_frameworks],
        system_prompt=SYSTEM_PROMPT,
        response_format=AdvisorResponse,
        checkpointer=resolved_checkpointer,
    )


__all__ = [
    "SYSTEM_PROMPT",
    "build_advisor_agent",
]
