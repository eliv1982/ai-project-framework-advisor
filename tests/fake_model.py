"""Детерминированная фейковая модель для тестов настоящего собранного агента.

Вместо ChatOpenAI в build_advisor_agent подставляется ScriptedModel: она по
очереди отдаёт заранее заданные AIMessage. Всё остальное (create_agent,
инструменты, response_format, InMemorySaver) — настоящее. Сетевых вызовов нет.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import advisor.agent as agent_module


class ScriptedModel(BaseChatModel):
    """Отдаёт AIMessage из script по одному на каждый вызов модели."""

    script: list[Any]
    index: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.index >= len(self.script):
            raise AssertionError("Сценарий фейковой модели исчерпан.")
        message = self.script[self.index]
        self.index += 1
        return ChatResult(generations=[ChatGeneration(message=message)])


def tool_call(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id}],
    )


def profile_call(framework_name: str = "langchain", call_id: str = "p1") -> AIMessage:
    return tool_call(
        "get_framework_profile", {"framework_name": framework_name}, call_id
    )


def compare_call(
    framework_names: list[str],
    criteria: list[str] | None = None,
    call_id: str = "c1",
) -> AIMessage:
    return tool_call(
        "compare_frameworks",
        {
            "framework_names": framework_names,
            "criteria": criteria or ["agent_support", "typical_components"],
        },
        call_id,
    )


def comparison_item(framework: str) -> dict[str, Any]:
    return {
        "framework": framework,
        "strengths": ["Сильная сторона"],
        "limitations": ["Ограничение"],
        "fit_for_project": "Оценка соответствия проекту.",
    }


def final_answer(call_id: str = "final", **overrides: Any) -> AIMessage:
    """Структурированный ответ AdvisorResponse (по умолчанию — langchain)."""
    payload: dict[str, Any] = {
        "recommended_framework": "langchain",
        "comparison": [],
        "reasoning": ["Причина"],
        "suitable_components": ["Tools", "Middleware"],
        "risks": ["Риск"],
    }
    payload.update(overrides)
    return tool_call("AdvisorResponse", payload, call_id)


def build_scripted_agent(monkeypatch, script: list[AIMessage]):
    """Собирает настоящего агента через build_advisor_agent с фейковой моделью."""
    model = ScriptedModel(script=list(script))
    monkeypatch.setattr(agent_module, "ChatOpenAI", lambda **kwargs: model)
    agent = agent_module.build_advisor_agent(
        api_key="test-key", model_name="test-model"
    )
    return agent, model
