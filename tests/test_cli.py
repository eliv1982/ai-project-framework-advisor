from __future__ import annotations

import pytest
from pydantic import ValidationError

import advisor.cli as cli
from advisor.schema import AdvisorResponse


class QueueAgent:
    """Fake agent: yields one queued outcome per invoke() call.

    Each queued outcome is either a dict (used as structured_response
    payload) or an Exception instance to be raised.
    """

    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls = []

    def invoke(self, messages, config=None):
        self.calls.append({"messages": messages, "config": config})
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return {"structured_response": outcome}


def _make_response(**overrides):
    payload = {
        "recommended_framework": "langchain",
        "reasoning": ["Причина один", "Причина два"],
        "suitable_components": ["Tools", "Middleware"],
        "risks": ["Риск один"],
    }
    payload.update(overrides)
    return AdvisorResponse(**payload)


def _patch_input(monkeypatch, values):
    inputs = iter(values)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))


# --- create_thread_id ---


def test_create_thread_id_returns_non_empty_unique_strings():
    first = cli.create_thread_id()
    second = cli.create_thread_id()

    assert isinstance(first, str)
    assert first
    assert isinstance(second, str)
    assert second
    assert first != second


# --- format_response ---


def test_format_response_contains_headers_and_content_without_mutating_input():
    response = _make_response()
    original_dump = response.model_dump()

    formatted = cli.format_response(response)

    assert "Рекомендованный фреймворк:" in formatted
    assert "LangChain" in formatted
    assert "(langchain)" in formatted
    assert "Почему:" in formatted
    assert "Подходящие компоненты:" in formatted
    assert "Риски и ограничения:" in formatted
    for reason in response.reasoning:
        assert f"- {reason}" in formatted
    for component in response.suitable_components:
        assert component in formatted
    for risk in response.risks:
        assert risk in formatted
    assert response.model_dump() == original_dump


# --- extract_structured_response ---


def test_extract_structured_response_accepts_advisor_response_instance():
    response = _make_response()

    result = cli.extract_structured_response({"structured_response": response})

    assert result is response


def test_extract_structured_response_validates_dict():
    payload = {
        "recommended_framework": "crewai",
        "reasoning": ["Причина"],
        "suitable_components": ["Agents"],
        "risks": ["Риск"],
    }

    result = cli.extract_structured_response({"structured_response": payload})

    assert isinstance(result, AdvisorResponse)
    assert result.recommended_framework == "crewai"


def test_extract_structured_response_missing_key_raises_value_error():
    with pytest.raises(ValueError):
        cli.extract_structured_response({})


def test_extract_structured_response_unsupported_type_raises_value_error():
    with pytest.raises(ValueError):
        cli.extract_structured_response({"structured_response": 123})


def test_extract_structured_response_invalid_dict_raises_validation_error():
    with pytest.raises(ValidationError):
        cli.extract_structured_response(
            {"structured_response": {"recommended_framework": "django"}}
        )


# --- invoke_agent ---


def test_invoke_agent_passes_expected_arguments_and_returns_response():
    response = _make_response()
    agent = QueueAgent([response.model_dump()])

    result = cli.invoke_agent(agent, user_message="Нужен RAG", thread_id="thread-1")

    assert isinstance(result, AdvisorResponse)
    assert len(agent.calls) == 1
    call = agent.calls[0]
    assert call["messages"] == {
        "messages": [
            {"role": "user", "content": "Нужен RAG"},
        ]
    }
    assert call["config"] == {"configurable": {"thread_id": "thread-1"}}


# --- run_cli ---


def test_run_cli_returns_1_when_configuration_invalid(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)

    def fake_build_advisor_agent():
        raise ValueError("Не задан OPENAI_API_KEY")

    monkeypatch.setattr(cli, "build_advisor_agent", fake_build_advisor_agent)

    exit_code = cli.run_cli()

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "OPENAI_API_KEY" in captured.err


def test_run_cli_exit_command_returns_0_without_invoking_agent(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    agent = QueueAgent([])
    monkeypatch.setattr(cli, "build_advisor_agent", lambda: agent)
    _patch_input(monkeypatch, ["/exit"])

    exit_code = cli.run_cli()

    assert exit_code == 0
    assert agent.calls == []
    captured = capsys.readouterr()
    assert "До свидания!" in captured.out


def test_run_cli_skips_empty_input(monkeypatch):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    agent = QueueAgent([])
    monkeypatch.setattr(cli, "build_advisor_agent", lambda: agent)
    _patch_input(monkeypatch, ["", "/exit"])

    exit_code = cli.run_cli()

    assert exit_code == 0
    assert agent.calls == []


def test_run_cli_handles_normal_query(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    response_payload = _make_response().model_dump()
    agent = QueueAgent([response_payload])
    monkeypatch.setattr(cli, "build_advisor_agent", lambda: agent)
    _patch_input(monkeypatch, ["Нужен RAG по договорам", "/exit"])

    exit_code = cli.run_cli()

    assert exit_code == 0
    assert len(agent.calls) == 1
    captured = capsys.readouterr()
    assert "Рекомендованный фреймворк: LangChain (langchain)" in captured.out


def test_run_cli_new_command_starts_new_thread(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    response_payload = _make_response().model_dump()
    agent = QueueAgent([response_payload, response_payload])
    monkeypatch.setattr(cli, "build_advisor_agent", lambda: agent)

    thread_ids = iter(["thread-1", "thread-2"])
    monkeypatch.setattr(cli, "create_thread_id", lambda: next(thread_ids))

    _patch_input(
        monkeypatch,
        ["Первый вопрос", "/new", "Второй вопрос", "/exit"],
    )

    exit_code = cli.run_cli()

    assert exit_code == 0
    assert len(agent.calls) == 2
    assert agent.calls[0]["config"] == {"configurable": {"thread_id": "thread-1"}}
    assert agent.calls[1]["config"] == {"configurable": {"thread_id": "thread-2"}}
    captured = capsys.readouterr()
    assert "Начат новый диалог." in captured.out


def test_run_cli_recovers_from_value_error_during_request(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    agent = QueueAgent([ValueError("Ответ агента не содержит structured_response")])
    monkeypatch.setattr(cli, "build_advisor_agent", lambda: agent)
    _patch_input(monkeypatch, ["Сломанный запрос", "/exit"])

    exit_code = cli.run_cli()

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "structured_response" in captured.out


def test_run_cli_handles_eof_error(monkeypatch):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    agent = QueueAgent([])
    monkeypatch.setattr(cli, "build_advisor_agent", lambda: agent)

    def fake_input(prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", fake_input)

    exit_code = cli.run_cli()

    assert exit_code == 0


def test_run_cli_handles_keyboard_interrupt_on_input(monkeypatch):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    agent = QueueAgent([])
    monkeypatch.setattr(cli, "build_advisor_agent", lambda: agent)

    def fake_input(prompt=""):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", fake_input)

    exit_code = cli.run_cli()

    assert exit_code == 0


def test_run_cli_handles_keyboard_interrupt_during_invoke(monkeypatch):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    agent = QueueAgent([KeyboardInterrupt()])
    monkeypatch.setattr(cli, "build_advisor_agent", lambda: agent)
    _patch_input(monkeypatch, ["Запрос"])

    exit_code = cli.run_cli()

    assert exit_code == 0
