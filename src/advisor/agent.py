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
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from advisor.schema import AdvisorResponse, FrameworkComparison
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
- если пользователь просит сравнить несколько фреймворков, сначала вызови compare_frameworks для всех названных вариантов и релевантных критериев — не пропускай этот шаг;
- для итоговой рекомендации по проекту сначала определи как минимум двух разумных кандидатов и сравни их по релевантным критериям через compare_frameworks;
- на КАЖДОЕ новое сообщение пользователя вызывай инструменты заново: данные, полученные в предыдущих сообщениях диалога, ответ не подтверждают, и ответ без успешного вызова get_framework_profile или compare_frameworks в текущем запросе будет отклонён;
- чтобы назвать suitable_components, включай критерий typical_components в compare_frameworks или вызови get_framework_profile для выбранного фреймворка;
- в пределах одного ответа не повторяй одинаковые вызовы без необходимости.

Критерии выбора: учитывай задачу проекта, необходимость RAG, поддержку агентов, потребность в многоагентной системе, важность явного и предсказуемого конвейера, экосистему интеграций, порог входа команды и известные ограничения фреймворков.

Явное сравнение (поле comparison):
- поле comparison должно содержать отдельный элемент для каждого фреймворка, который ты сравнил через compare_frameworks в этом ответе, и только для них (без повторов; recommended_framework тоже должен быть среди них);
- в каждом элементе comparison должны быть сильные стороны (strengths), ограничения (limitations) и оценка соответствия именно текущему проекту (fit_for_project) — не общие характеристики фреймворка, а применительно к описанному проекту;
- нельзя заменять сравнение перечислением достоинств только победителя: если рассмотрено несколько фреймворков, comparison должен описывать каждого кандидата, а не только recommended_framework;
- recommended_framework выбирается только после явного сравнения кандидатов, а не до него;
- если пользователь спрашивает только об одном фреймворке и сравнение не требуется, comparison может быть пустым списком.

Формат финального ответа должен соответствовать схеме AdvisorResponse:
- recommended_framework — ровно один канонический ключ из списка выше, причём именно тот фреймворк, данные о котором ты получил инструментами в этом ответе;
- comparison — список сравнений рассмотренных фреймворков (см. правила выше), может быть пустым только при вопросе об одном фреймворке;
- reasoning — конкретные причины выбора, основанные на данных инструментов;
- suitable_components — только компоненты из поля typical_components профиля выбранного фреймворка, названия дословно, без перевода и пояснений (придуманные названия будут отклонены);
- risks — реальные ограничения и компромиссы рекомендации.

Язык: формулируй comparison, reasoning, suitable_components и risks на русском языке, за исключением официальных названий компонентов и фреймворков (например, "LangGraph", "Crews", "Workflows").

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

    if checkpointer is not None:
        resolved_checkpointer = checkpointer
    else:
        serializer = JsonPlusSerializer(
            allowed_msgpack_modules=[
                AdvisorResponse,
                FrameworkComparison,
            ],
        )
        resolved_checkpointer = InMemorySaver(serde=serializer)

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
