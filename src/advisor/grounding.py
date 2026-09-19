"""Проверка того, что ответ агента опирается на локальные данные о фреймворках.

LangChain делает инструменты доступными, но не заставляет модель их вызывать:
модель может вернуть валидный AdvisorResponse и без единого вызова. Этот
модуль после выполнения агента проверяет след СВОЕГО хода (сообщения после
последнего HumanMessage), а не всего накопленного диалога, поэтому успешный
вызов инструмента из прошлого хода новый ответ не подтверждает.

Трасса проверяется строго (fail closed): на каждый вызов инструмента данных
должен быть ровно один результат от того же инструмента, а успешный результат
обязан иметь полную структуру, совпадающую с локальными данными.

Проверка детерминированная и ограничена фактами, которые есть в локальных
профилях и результатах инструментов. Общая фактическая корректность текста
(strengths, reasoning и т. д.) не проверяется.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from advisor.framework_data import (
    CRITERIA_LABELS,
    FRAMEWORK_PROFILES,
    SUPPORTED_CRITERIA,
)
from advisor.schema import AdvisorResponse
from advisor.tools import normalize_framework_name, normalize_name

# Только эти инструменты подтверждают рекомендацию. Вызов инструмента
# структурированного ответа (AdvisorResponse) в их число не входит.
DATA_TOOL_NAMES = ("get_framework_profile", "compare_frameworks")

# Ключи успешного результата, как их возвращают инструменты из advisor.tools.
_PROFILE_RESULT_KEYS = {"ok", "framework", "profile"}
_COMPARE_RESULT_KEYS = {
    "ok",
    "frameworks",
    "display_names",
    "criteria",
    "criterion_labels",
    "comparison",
}


class GroundingError(Exception):
    """Ответ агента не подтверждён локальными данными текущего хода."""


def _current_turn_trace(messages: Any) -> list[Any]:
    """Возвращает сообщения после последнего HumanMessage (текущий ход)."""
    if not isinstance(messages, list):
        raise GroundingError(
            "Результат агента не содержит трассу сообщений: невозможно "
            "проверить использование локальных данных."
        )

    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return messages[index + 1 :]

    raise GroundingError(
        "В трассе агента нет сообщения пользователя: невозможно определить "
        "границы текущего запроса."
    )


def _parse_payload(content: Any) -> dict[str, Any] | None:
    """Разбирает содержимое ToolMessage в dict; иначе None.

    Инструменты сами перехватывают ошибки и возвращают {"ok": false, ...},
    при этом ToolMessage.status остаётся "success" — поэтому статуса мало.
    """
    if not isinstance(content, str):
        return None
    try:
        payload = json.loads(content)
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def _is_str_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _profile_result_frameworks(payload: dict[str, Any]) -> set[str] | None:
    """Полный успешный результат get_framework_profile → {ключ} или None.

    Ключ должен быть каноническим, а profile — совпадать с сохранённым
    профилем в FRAMEWORK_PROFILES (это то, что возвращает сам инструмент).
    """
    key = payload.get("framework")
    if (
        set(payload) == _PROFILE_RESULT_KEYS
        and isinstance(key, str)
        and key in FRAMEWORK_PROFILES
        and payload["profile"] == FRAMEWORK_PROFILES[key]
    ):
        return {key}
    return None


def _compare_result_frameworks(payload: dict[str, Any]) -> set[str] | None:
    """Полный успешный результат compare_frameworks → сравнённые ключи или None.

    От 2 до 5 уникальных канонических фреймворков, непустые уникальные
    поддерживаемые критерии; display_names, criterion_labels и comparison
    совпадают с тем, что собирается из FRAMEWORK_PROFILES и CRITERIA_LABELS.
    """
    frameworks = payload.get("frameworks")
    criteria = payload.get("criteria")
    if not (
        set(payload) == _COMPARE_RESULT_KEYS
        and _is_str_list(frameworks)
        and _is_str_list(criteria)
    ):
        return None
    if not (
        2 <= len(frameworks) <= 5
        and len(set(frameworks)) == len(frameworks)
        and set(frameworks) <= FRAMEWORK_PROFILES.keys()
    ):
        return None
    if not (
        criteria
        and len(set(criteria)) == len(criteria)
        and set(criteria) <= set(SUPPORTED_CRITERIA)
    ):
        return None
    if (
        payload["display_names"]
        == {key: FRAMEWORK_PROFILES[key]["display_name"] for key in frameworks}
        and payload["criterion_labels"]
        == {criterion: CRITERIA_LABELS[criterion] for criterion in criteria}
        and payload["comparison"]
        == {
            criterion: {key: FRAMEWORK_PROFILES[key][criterion] for key in frameworks}
            for criterion in criteria
        }
    ):
        return set(frameworks)
    return None


def _grounded_frameworks(trace: list[Any]) -> tuple[set[str], set[str]]:
    """Возвращает (profiled, compared) по успешным вызовам текущего хода.

    Для каждого вызова инструмента из DATA_TOOL_NAMES в AIMessage текущего
    хода требуется ровно один ToolMessage с тем же tool_call_id и тем же
    именем инструмента, идущий после вызова. Нарушение — GroundingError.
    Результат с status="error" или {"ok": false, ...} — штатно неудачный
    вызов: он ничего не подтверждает. Любой другой результат обязан быть
    полным успешным результатом инструмента, иначе — GroundingError.
    """
    calls: dict[str, tuple[str, int]] = {}
    results: dict[str, list[tuple[int, ToolMessage]]] = {}

    for index, message in enumerate(trace):
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                if call["name"] not in DATA_TOOL_NAMES:
                    continue
                call_id = call.get("id")
                if not call_id or call_id in calls:
                    raise GroundingError(
                        f"Вызов {call['name']} без идентификатора или с "
                        "повторяющимся идентификатором: трасса неоднозначна."
                    )
                calls[call_id] = (call["name"], index)
        elif isinstance(message, ToolMessage):
            results.setdefault(message.tool_call_id, []).append((index, message))

    profiled: set[str] = set()
    compared: set[str] = set()

    for call_id, (tool_name, call_index) in calls.items():
        found = results.get(call_id, [])
        if len(found) > 1:
            raise GroundingError(
                f"Для вызова {tool_name} получено больше одного результата."
            )
        if not found or found[0][0] < call_index:
            raise GroundingError(f"Для вызова {tool_name} нет результата инструмента.")

        message = found[0][1]
        if message.name != tool_name:
            raise GroundingError(
                f"Результат вызова {tool_name} принадлежит другому "
                f"инструменту: {message.name!r}."
            )
        if message.status != "success":
            continue

        payload = _parse_payload(message.content)
        if payload is not None and payload.get("ok") is False:
            continue

        keys: set[str] | None = None
        if payload is not None and payload.get("ok") is True:
            if tool_name == "get_framework_profile":
                keys = _profile_result_frameworks(payload)
            else:
                keys = _compare_result_frameworks(payload)
        if keys is None:
            raise GroundingError(
                f"Результат инструмента {tool_name} имеет неверную структуру."
            )
        (profiled if tool_name == "get_framework_profile" else compared).update(keys)

    return profiled, compared


def _canonical_framework(name: str) -> str:
    try:
        return normalize_framework_name(name)
    except ValueError:
        raise GroundingError(
            f"Фреймворк {name!r} не входит в число поддерживаемых."
        ) from None


def _check_components(framework: str, components: list[str]) -> None:
    """Компоненты должны быть из typical_components сохранённого профиля.

    Допускаются и части составных названий ("Documents/Nodes" →
    "Documents", "Nodes"); регистр, "_", "-" и лишние пробелы не важны.
    """
    allowed: set[str] = set()
    for stored in FRAMEWORK_PROFILES[framework]["typical_components"]:
        allowed.add(normalize_name(stored))
        allowed.update(normalize_name(part) for part in stored.split("/"))

    for component in components:
        if normalize_name(component) not in allowed:
            raise GroundingError(
                f"Компонент {component!r} отсутствует в локальном профиле "
                f"фреймворка {FRAMEWORK_PROFILES[framework]['display_name']}."
            )


def _check_comparison(
    response: AdvisorResponse, recommended: str, compared: set[str]
) -> None:
    frameworks = [_canonical_framework(item.framework) for item in response.comparison]

    if len(set(frameworks)) != len(frameworks):
        raise GroundingError("В сравнении есть повторяющиеся фреймворки.")

    if recommended not in frameworks:
        raise GroundingError(
            "Рекомендованный фреймворк отсутствует в сравнении."
        )

    if not compared:
        raise GroundingError(
            "Сравнение не подтверждено: в текущем запросе нет успешного "
            "вызова compare_frameworks."
        )

    if set(frameworks) != compared:
        raise GroundingError(
            "Сравнение не совпадает с фреймворками, сравнёнными инструментом "
            "compare_frameworks в текущем запросе: ожидались "
            f"{sorted(compared)}, получены {sorted(frameworks)}."
        )


def verify_grounded_response(
    messages: Any, response: AdvisorResponse
) -> AdvisorResponse:
    """Проверяет ответ по трассе текущего хода; при нарушении — GroundingError."""
    profiled, compared = _grounded_frameworks(_current_turn_trace(messages))

    if not profiled and not compared:
        raise GroundingError(
            "Локальные данные не использованы: в текущем запросе нет "
            "успешного вызова get_framework_profile или compare_frameworks."
        )

    recommended = _canonical_framework(response.recommended_framework)

    if recommended not in profiled | compared:
        raise GroundingError(
            f"Рекомендованный фреймворк {recommended!r} не входит в данные, "
            "полученные инструментами в текущем запросе."
        )

    _check_components(recommended, response.suitable_components)

    if response.comparison:
        _check_comparison(response, recommended, compared)

    return response


__all__ = [
    "DATA_TOOL_NAMES",
    "GroundingError",
    "verify_grounded_response",
]
