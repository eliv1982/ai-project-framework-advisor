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
    "AdvisorResponse",
]
