"""Собственные инструменты агента-советника по фреймворкам.

Все инструменты читают данные исключительно из advisor.framework_data
и не делают сетевых обращений, не обращаются к LLM и не имеют побочных эффектов.
"""

from __future__ import annotations

from copy import deepcopy

from langchain.tools import tool

from advisor.framework_data import (
    CRITERIA_LABELS,
    FRAMEWORK_ALIASES,
    FRAMEWORK_PROFILES,
    SUPPORTED_CRITERIA,
    FrameworkProfile,
)

ToolResult = dict[str, object]


def normalize_name(name: str) -> str:
    """Точная нормализация названия: обрезка пробелов, casefold, "_" и "-"
    как пробелы, схлопывание внутренних пробелов.

    Общее правило для названий фреймворков и компонентов профилей.
    """
    return " ".join(
        name.strip()
        .casefold()
        .replace("_", " ")
        .replace("-", " ")
        .split()
    )


def normalize_framework_name(name: str) -> str:
    """Приводит название фреймворка к каноническому ключу FRAMEWORK_PROFILES.

    Выполняет только точную нормализацию (normalize_name) и поиск в
    FRAMEWORK_ALIASES. Нечёткий подбор и угадывание названия не выполняются.
    """
    normalized = normalize_name(name)

    if not normalized:
        raise ValueError("Название фреймворка не может быть пустым.")

    canonical_key = FRAMEWORK_ALIASES.get(normalized)
    if canonical_key is None:
        raise ValueError(
            f"Неизвестное название фреймворка: {name!r}. "
            "Поддерживаются только LangChain, LlamaIndex, Haystack, "
            "Semantic Kernel и CrewAI (включая их известные алиасы)."
        )

    return canonical_key


def _get_profile(canonical_key: str) -> FrameworkProfile:
    """Возвращает независимую копию профиля по каноническому ключу."""
    profile = FRAMEWORK_PROFILES.get(canonical_key)
    if profile is None:
        raise ValueError(f"Профиль для ключа {canonical_key!r} не найден.")
    return deepcopy(profile)


def _validate_criteria(criteria: list[str]) -> list[str]:
    """Нормализует и проверяет список критериев сравнения.

    Выполняет только точную нормализацию (обрезка пробелов, casefold,
    замена пробелов и дефисов на "_", схлопывание повторов "_") и поиск
    в SUPPORTED_CRITERIA. Дубли удаляются с сохранением исходного
    порядка. Неизвестные критерии не угадываются.
    """
    if not criteria:
        raise ValueError("Список критериев сравнения не может быть пустым.")

    validated: list[str] = []
    for raw_criterion in criteria:
        normalized = "_".join(
            part
            for part in (
                raw_criterion.strip()
                .casefold()
                .replace(" ", "_")
                .replace("-", "_")
                .split("_")
            )
            if part
        )

        if not normalized:
            raise ValueError("Критерий сравнения не может быть пустым.")

        if normalized not in SUPPORTED_CRITERIA:
            raise ValueError(
                f"Неизвестный критерий сравнения: {raw_criterion!r}. "
                f"Поддерживаются только: {', '.join(SUPPORTED_CRITERIA)}."
            )

        if normalized not in validated:
            validated.append(normalized)

    return validated


@tool
def get_framework_profile(framework_name: str) -> ToolResult:
    """Возвращает полный профиль одного фреймворка для построения ИИ-проектов.

    Используй этот инструмент, когда нужно получить проверяемую информацию
    об одном конкретном фреймворке: его основное назначение, поддержку
    агентов, поддержку RAG, экосистему интеграций, порог входа, типовые
    компоненты и известные ограничения. Это нужно, чтобы сравнить
    фреймворк с требованиями пользователя или обосновать рекомендацию.

    Поддерживается ровно пять фреймворков: LangChain, LlamaIndex, Haystack,
    Semantic Kernel и CrewAI. Допускаются разные варианты написания
    названия (регистр, лишние пробелы, дефисы/подчёркивания вместо
    пробела) — см. зарегистрированные алиасы. Нечёткий подбор и угадывание
    названия не выполняются: при неизвестном названии инструмент вернёт
    структурированную ошибку со списком поддерживаемых фреймворков.

    Аргумент framework_name — это название ОДНОГО фреймворка (например,
    "LangChain", "Semantic Kernel" или "Crew AI"), а не список нескольких
    фреймворков.
    """
    try:
        canonical_key = normalize_framework_name(framework_name)
        profile = _get_profile(canonical_key)
    except ValueError as error:
        return {
            "ok": False,
            "error": str(error),
            "supported_frameworks": [
                profile["display_name"]
                for profile in FRAMEWORK_PROFILES.values()
            ],
        }

    return {
        "ok": True,
        "framework": canonical_key,
        "profile": profile,
    }


@tool
def compare_frameworks(
    framework_names: list[str],
    criteria: list[str],
) -> ToolResult:
    """Сравнивает от двух до пяти фреймворков по выбранным критериям.

    Используй этот инструмент для прямого сравнения нескольких вариантов
    построения ИИ-проекта. Для изучения только одного конкретного
    фреймворка вместо этого используй get_framework_profile.

    Поддерживается сравнение LangChain, LlamaIndex, Haystack, Semantic
    Kernel и CrewAI — от двух до пяти РАЗНЫХ фреймворков одновременно
    (после удаления дублей названий). Допускаются разные варианты
    написания названий (регистр, лишние пробелы, дефисы/подчёркивания
    вместо пробела) — см. зарегистрированные алиасы. Нечёткий подбор и
    угадывание названий не выполняются.

    Args:
        framework_names: Список названий фреймворков для сравнения.
            После нормализации и удаления дублей должно остаться от
            двух до пяти уникальных фреймворков.
        criteria: Список критериев сравнения. Допустимые значения:
            primary_use_case, agent_support, rag_support,
            integration_ecosystem, learning_curve, typical_components,
            limitations. Регистр, пробелы и дефисы/подчёркивания в
            названиях критериев не имеют значения.
    """
    try:
        canonical_frameworks: list[str] = []
        for framework_name in framework_names:
            canonical_key = normalize_framework_name(framework_name)
            if canonical_key not in canonical_frameworks:
                canonical_frameworks.append(canonical_key)

        if not (2 <= len(canonical_frameworks) <= 5):
            raise ValueError(
                "Нужно указать от 2 до 5 разных фреймворков для сравнения, "
                f"после удаления дублей получено уникальных: "
                f"{len(canonical_frameworks)}."
            )

        canonical_criteria = _validate_criteria(criteria)

        profiles = {
            canonical_key: _get_profile(canonical_key)
            for canonical_key in canonical_frameworks
        }
    except ValueError as error:
        return {
            "ok": False,
            "error": str(error),
            "supported_frameworks": [
                profile["display_name"]
                for profile in FRAMEWORK_PROFILES.values()
            ],
            "supported_criteria": {
                criterion: CRITERIA_LABELS[criterion]
                for criterion in SUPPORTED_CRITERIA
            },
        }

    display_names = {
        canonical_key: profiles[canonical_key]["display_name"]
        for canonical_key in canonical_frameworks
    }
    criterion_labels = {
        criterion: CRITERIA_LABELS[criterion] for criterion in canonical_criteria
    }
    comparison = {
        criterion: {
            canonical_key: profiles[canonical_key][criterion]
            for canonical_key in canonical_frameworks
        }
        for criterion in canonical_criteria
    }

    return {
        "ok": True,
        "frameworks": canonical_frameworks,
        "display_names": display_names,
        "criteria": canonical_criteria,
        "criterion_labels": criterion_labels,
        "comparison": comparison,
    }


__all__ = [
    "get_framework_profile",
    "compare_frameworks",
    "normalize_name",
    "normalize_framework_name",
]
