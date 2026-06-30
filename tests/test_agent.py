from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import advisor.agent as agent_module


def test_build_advisor_agent_uses_passed_arguments(monkeypatch):
    fake_model_instance = object()
    fake_chat_openai = MagicMock(return_value=fake_model_instance)
    fake_create_agent = MagicMock(return_value="created-agent")
    custom_checkpointer = object()

    monkeypatch.setattr(agent_module, "ChatOpenAI", fake_chat_openai)
    monkeypatch.setattr(agent_module, "create_agent", fake_create_agent)

    result = agent_module.build_advisor_agent(
        api_key="test-key",
        model_name="test-model",
        checkpointer=custom_checkpointer,
    )

    fake_chat_openai.assert_called_once_with(model="test-model", api_key="test-key")
    fake_create_agent.assert_called_once_with(
        model=fake_model_instance,
        tools=[agent_module.get_framework_profile, agent_module.compare_frameworks],
        system_prompt=agent_module.SYSTEM_PROMPT,
        response_format=agent_module.AdvisorResponse,
        checkpointer=custom_checkpointer,
    )
    assert result == "created-agent"


def test_build_advisor_agent_reads_values_from_environment(monkeypatch):
    fake_model_instance = object()
    fake_chat_openai = MagicMock(return_value=fake_model_instance)
    fake_create_agent = MagicMock(return_value="created-agent")

    monkeypatch.setattr(agent_module, "ChatOpenAI", fake_chat_openai)
    monkeypatch.setattr(agent_module, "create_agent", fake_create_agent)
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("OPENAI_MODEL", "env-model")

    agent_module.build_advisor_agent()

    fake_chat_openai.assert_called_once_with(model="env-model", api_key="env-key")


def test_explicit_arguments_take_priority_over_environment(monkeypatch):
    fake_model_instance = object()
    fake_chat_openai = MagicMock(return_value=fake_model_instance)
    fake_create_agent = MagicMock(return_value="created-agent")

    monkeypatch.setattr(agent_module, "ChatOpenAI", fake_chat_openai)
    monkeypatch.setattr(agent_module, "create_agent", fake_create_agent)
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("OPENAI_MODEL", "env-model")

    agent_module.build_advisor_agent(
        api_key="explicit-key",
        model_name="explicit-model",
    )

    fake_chat_openai.assert_called_once_with(
        model="explicit-model",
        api_key="explicit-key",
    )


def test_creates_in_memory_saver_when_checkpointer_not_provided(monkeypatch):
    fake_model_instance = object()
    fake_chat_openai = MagicMock(return_value=fake_model_instance)
    fake_create_agent = MagicMock(return_value="created-agent")
    fake_checkpointer_instance = object()
    fake_in_memory_saver = MagicMock(return_value=fake_checkpointer_instance)

    monkeypatch.setattr(agent_module, "ChatOpenAI", fake_chat_openai)
    monkeypatch.setattr(agent_module, "create_agent", fake_create_agent)
    monkeypatch.setattr(agent_module, "InMemorySaver", fake_in_memory_saver)

    agent_module.build_advisor_agent(api_key="test-key", model_name="test-model")

    fake_in_memory_saver.assert_called_once_with()
    assert fake_create_agent.call_args.kwargs["checkpointer"] is fake_checkpointer_instance


def test_missing_api_key_raises_value_error(monkeypatch):
    fake_chat_openai = MagicMock()
    fake_create_agent = MagicMock()

    monkeypatch.setattr(agent_module, "ChatOpenAI", fake_chat_openai)
    monkeypatch.setattr(agent_module, "create_agent", fake_create_agent)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        agent_module.build_advisor_agent(model_name="test-model")

    fake_chat_openai.assert_not_called()
    fake_create_agent.assert_not_called()


def test_missing_model_name_raises_value_error(monkeypatch):
    fake_chat_openai = MagicMock()
    fake_create_agent = MagicMock()

    monkeypatch.setattr(agent_module, "ChatOpenAI", fake_chat_openai)
    monkeypatch.setattr(agent_module, "create_agent", fake_create_agent)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    with pytest.raises(ValueError, match="OPENAI_MODEL"):
        agent_module.build_advisor_agent(api_key="test-key")

    fake_chat_openai.assert_not_called()
    fake_create_agent.assert_not_called()


@pytest.mark.parametrize(
    ("api_key", "model_name"),
    [
        ("", "valid-model"),
        ("   ", "valid-model"),
        ("valid-key", ""),
        ("valid-key", "   "),
    ],
)
def test_blank_api_key_or_model_name_raises_value_error(
    monkeypatch, api_key, model_name
):
    fake_chat_openai = MagicMock()
    fake_create_agent = MagicMock()

    monkeypatch.setattr(agent_module, "ChatOpenAI", fake_chat_openai)
    monkeypatch.setattr(agent_module, "create_agent", fake_create_agent)

    with pytest.raises(ValueError):
        agent_module.build_advisor_agent(api_key=api_key, model_name=model_name)

    fake_chat_openai.assert_not_called()
    fake_create_agent.assert_not_called()


def test_system_prompt_contains_key_constraints():
    prompt = agent_module.SYSTEM_PROMPT
    casefolded_prompt = prompt.casefold()

    for canonical_key in (
        "langchain",
        "llamaindex",
        "haystack",
        "semantic_kernel",
        "crewai",
    ):
        assert canonical_key in prompt

    assert "get_framework_profile" in prompt
    assert "compare_frameworks" in prompt
    assert "инструмент" in casefolded_prompt
    assert "выдум" in casefolded_prompt or "придум" in casefolded_prompt
    assert "AdvisorResponse" in prompt or all(
        field in prompt
        for field in (
            "recommended_framework",
            "reasoning",
            "suitable_components",
            "risks",
        )
    )
