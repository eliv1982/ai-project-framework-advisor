"""Схема структурированного ответа агента-советника по фреймворкам.

Модель не зависит от LangChain и не выполняет никаких побочных действий —
это только описание формы ответа для response_format агента.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

FrameworkKey = Literal[
    "langchain",
    "llamaindex",
    "haystack",
    "semantic_kernel",
    "crewai",
]

NonEmptyText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class FrameworkComparison(BaseModel):
    """Сравнение одного рассмотренного фреймворка для конкретного проекта."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    framework: FrameworkKey
    strengths: list[NonEmptyText] = Field(
        min_length=1,
        max_length=3,
        description="Сильные стороны фреймворка именно для описанного проекта.",
    )
    limitations: list[NonEmptyText] = Field(
        min_length=1,
        max_length=3,
        description="Ограничения фреймворка именно для описанного проекта.",
    )
    fit_for_project: NonEmptyText = Field(
        description="Краткая оценка соответствия фреймворка требованиям проекта.",
    )


class AdvisorResponse(BaseModel):
    """Структурированная рекомендация по выбору фреймворка для ИИ-проекта."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    recommended_framework: FrameworkKey = Field(
        description=(
            "Канонический ключ одного рекомендованного фреймворка: "
            "langchain, llamaindex, haystack, semantic_kernel или crewai."
        ),
    )
    comparison: list[FrameworkComparison] = Field(
        default_factory=list,
        max_length=5,
        description=(
            "Явное сравнение рассмотренных фреймворков. "
            "При запросе сравнения или выборе между несколькими вариантами "
            "список должен содержать по одному элементу для каждого кандидата. "
            "При вопросе только об одном фреймворке список может быть пустым."
        ),
    )
    reasoning: list[NonEmptyText] = Field(
        min_length=1,
        max_length=5,
        description=(
            "Список основных причин, по которым выбран рекомендованный "
            "фреймворк (от 1 до 5 пунктов)."
        ),
    )
    suitable_components: list[NonEmptyText] = Field(
        min_length=1,
        max_length=8,
        description=(
            "Компоненты рекомендованного фреймворка, которые подходят для "
            "описанного проекта (от 1 до 8 пунктов)."
        ),
    )
    risks: list[NonEmptyText] = Field(
        min_length=1,
        max_length=5,
        description=(
            "Ограничения, риски или компромиссы рекомендации, о которых "
            "стоит знать заранее (от 1 до 5 пунктов)."
        ),
    )


__all__ = [
    "FrameworkKey",
    "NonEmptyText",
    "FrameworkComparison",
    "AdvisorResponse",
]
