"""Консольный интерфейс агента-советника по выбору фреймворка.

Запускает текстовый диалог: каждое сообщение пользователя передаётся
агенту, а структурированный ответ выводится в консоль в читаемом виде.
Не выполняет сетевых вызовов напрямую — только через build_advisor_agent
и переданный агент.
"""

from __future__ import annotations

import sys
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from pydantic import ValidationError

from advisor.agent import build_advisor_agent
from advisor.framework_data import FRAMEWORK_PROFILES
from advisor.schema import AdvisorResponse

EXIT_COMMANDS = {"/exit", "/quit"}
NEW_COMMAND = "/new"


def create_thread_id() -> str:
    """Возвращает новый непустой идентификатор диалога на основе uuid4()."""
    return str(uuid4())


def format_response(response: AdvisorResponse) -> str:
    """Форматирует AdvisorResponse в читаемый русскоязычный текст."""
    display_name = FRAMEWORK_PROFILES[response.recommended_framework][
        "display_name"
    ]
    lines = [
        f"Рекомендованный фреймворк: {display_name} "
        f"({response.recommended_framework})",
        "",
        "Почему:",
        "",
        *(f"- {reason}" for reason in response.reasoning),
        "",
        "Подходящие компоненты:",
        "",
        *(f"- {component}" for component in response.suitable_components),
        "",
        "Риски и ограничения:",
        "",
        *(f"- {risk}" for risk in response.risks),
    ]
    return "\n".join(lines)


def extract_structured_response(result: dict[str, Any]) -> AdvisorResponse:
    """Извлекает AdvisorResponse из результата agent.invoke()."""
    if "structured_response" not in result:
        raise ValueError(
            "Ответ агента не содержит structured_response: проверьте, что "
            "агент настроен с response_format=AdvisorResponse."
        )

    structured_response = result["structured_response"]

    if isinstance(structured_response, AdvisorResponse):
        return structured_response

    if isinstance(structured_response, dict):
        return AdvisorResponse.model_validate(structured_response)

    raise ValueError(
        "Неподдерживаемый тип structured_response: "
        f"{type(structured_response).__name__}."
    )


def invoke_agent(
    agent: Any,
    *,
    user_message: str,
    thread_id: str,
) -> AdvisorResponse:
    """Выполняет один вызов агента и возвращает структурированный ответ."""
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_message,
                }
            ]
        },
        config={
            "configurable": {
                "thread_id": thread_id,
            }
        },
    )
    return extract_structured_response(result)


def run_cli() -> int:
    """Запускает консольный диалог с агентом и возвращает код завершения."""
    load_dotenv()

    try:
        agent = build_advisor_agent()
    except ValueError as error:
        print(f"Ошибка конфигурации: {error}", file=sys.stderr)
        return 1

    thread_id = create_thread_id()

    print("Агент-советник по ИИ-фреймворкам")
    print("Опишите проект или задайте вопрос.")
    print("Команды: /new — новый диалог, /exit — выход.")

    while True:
        try:
            user_input = input("> ")
        except EOFError:
            print("Завершение работы.")
            return 0
        except KeyboardInterrupt:
            print("Завершение работы.")
            return 0

        stripped_input = user_input.strip()

        if not stripped_input:
            continue

        normalized_command = stripped_input.casefold()

        if normalized_command in EXIT_COMMANDS:
            print("До свидания!")
            return 0

        if normalized_command == NEW_COMMAND:
            thread_id = create_thread_id()
            print("Начат новый диалог.")
            continue

        try:
            response = invoke_agent(
                agent,
                user_message=stripped_input,
                thread_id=thread_id,
            )
        except (ValueError, ValidationError) as error:
            print(f"Не удалось обработать запрос: {error}")
            continue
        except KeyboardInterrupt:
            print("Завершение работы.")
            return 0

        print(format_response(response))


def main() -> None:
    raise SystemExit(run_cli())


if __name__ == "__main__":
    main()


__all__ = [
    "create_thread_id",
    "format_response",
    "extract_structured_response",
    "invoke_agent",
    "run_cli",
    "main",
]
