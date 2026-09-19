from __future__ import annotations

import subprocess
import sys
from inspect import signature

import pytest
from fake_model import build_scripted_agent, final_answer
from langchain.agents.structured_output import StructuredOutputValidationError
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError

import advisor.cli as cli
from advisor.schema import AdvisorResponse
from advisor.tools import compare_frameworks


def _grounded_messages():
    """Real LangChain trace of one turn with a successful compare_frameworks call."""
    call = {
        "name": "compare_frameworks",
        "args": {
            "framework_names": ["langchain", "llamaindex"],
            "criteria": ["typical_components"],
        },
        "id": "call-1",
    }
    return [
        HumanMessage("Нужен RAG"),
        AIMessage(content="", tool_calls=[call]),
        compare_frameworks.invoke({**call, "type": "tool_call"}),
    ]


class QueueAgent:
    """Fake agent: yields one queued outcome per invoke() call.

    Each queued outcome is either a dict (used as structured_response
    payload) or an Exception instance to be raised. By default the result
    carries a grounded message trace (see _grounded_messages); pass
    messages=... to override it.
    """

    def __init__(self, outcomes, messages=None):
        self._outcomes = list(outcomes)
        self._messages = messages
        self.calls = []

    def invoke(self, messages, config=None):
        self.calls.append({"messages": messages, "config": config})
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        trace = _grounded_messages() if self._messages is None else self._messages
        return {"messages": trace, "structured_response": outcome}


def _make_response(**overrides):
    payload = {
        "recommended_framework": "langchain",
        "comparison": [
            {
                "framework": "langchain",
                "strengths": ["Развитая поддержка агентов"],
                "limitations": ["Быстро меняющийся API между версиями"],
                "fit_for_project": "Хорошо подходит для оркестрации агентов.",
            },
            {
                "framework": "llamaindex",
                "strengths": ["Сильная поддержка RAG"],
                "limitations": ["Меньше возможностей для сложных агентов"],
                "fit_for_project": "Подходит, если важен поиск по документам.",
            },
        ],
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
    assert "Сравнение вариантов:" in formatted
    assert "LlamaIndex" in formatted
    assert "Сильные стороны:" in formatted
    assert "Ограничения:" in formatted
    assert "Соответствие проекту:" in formatted
    assert "Почему:" in formatted
    assert "Подходящие компоненты:" in formatted
    assert "Риски и ограничения:" in formatted
    for reason in response.reasoning:
        assert f"- {reason}" in formatted
    for component in response.suitable_components:
        assert component in formatted
    for risk in response.risks:
        assert risk in formatted
    for item in response.comparison:
        for strength in item.strengths:
            assert f"- {strength}" in formatted
        for limitation in item.limitations:
            assert f"- {limitation}" in formatted
        assert f"- {item.fit_for_project}" in formatted
    assert response.model_dump() == original_dump


def test_format_response_omits_comparison_section_when_empty():
    response = _make_response(comparison=[])

    formatted = cli.format_response(response)

    assert "Сравнение вариантов:" not in formatted
    assert "Сильные стороны:" not in formatted
    assert "Ограничения:" not in formatted
    assert "Соответствие проекту:" not in formatted


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


def test_run_cli_rejects_ungrounded_response_and_continues(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    payload = _make_response().model_dump()
    agent = QueueAgent([payload], messages=[HumanMessage("Нужен RAG")])
    monkeypatch.setattr(cli, "build_advisor_agent", lambda: agent)
    _patch_input(monkeypatch, ["Запрос без инструментов", "/exit"])

    exit_code = cli.run_cli()

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Рекомендация отклонена" in captured.out
    assert "Рекомендованный фреймворк" not in captured.out
    assert "Traceback" not in captured.out + captured.err
    assert captured.err == ""


def test_run_cli_rejects_direct_answer_from_compiled_agent(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    agent, _ = build_scripted_agent(monkeypatch, [final_answer()])
    monkeypatch.setattr(cli, "build_advisor_agent", lambda: agent)
    _patch_input(monkeypatch, ["Посоветуй фреймворк", "/exit"])

    exit_code = cli.run_cli()

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Рекомендация отклонена" in captured.out
    assert "Рекомендованный фреймворк" not in captured.out
    assert "Traceback" not in captured.out + captured.err


# --- обычные сбои провайдера/сети ---


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("upstream failed: sk-secret-token connection reset"),
        ConnectionError("sk-secret-token connection reset"),
        TimeoutError("sk-secret-token connection reset"),
        Exception("sk-secret-token connection reset"),
    ],
)
def test_run_cli_handles_ordinary_provider_error_without_leaking_details(
    monkeypatch, capsys, error
):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    agent = QueueAgent([error, _make_response().model_dump()])
    monkeypatch.setattr(cli, "build_advisor_agent", lambda: agent)
    _patch_input(monkeypatch, ["Первый запрос", "Второй запрос", "/exit"])

    exit_code = cli.run_cli()

    assert exit_code == 0
    assert len(agent.calls) == 2
    captured = capsys.readouterr()
    assert "Не удалось получить ответ от модели" in captured.out
    assert "sk-secret-token" not in captured.out + captured.err
    assert "connection reset" not in captured.out + captured.err
    assert "Traceback" not in captured.out + captured.err
    assert captured.err == ""
    # После сбоя диалог продолжается и следующий запрос обрабатывается.
    assert "Рекомендованный фреймворк: LangChain (langchain)" in captured.out


def test_run_cli_does_not_swallow_system_exit(monkeypatch):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    agent = QueueAgent([SystemExit(3)])
    monkeypatch.setattr(cli, "build_advisor_agent", lambda: agent)
    _patch_input(monkeypatch, ["Запрос"])

    with pytest.raises(SystemExit) as exit_info:
        cli.run_cli()

    assert exit_info.value.code == 3


# --- -h / --help ---


def _forbid_configuration(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Конфигурация не должна загружаться для справки.")

    monkeypatch.setattr(cli, "load_dotenv", fail)
    monkeypatch.setattr(cli, "build_advisor_agent", fail)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)


@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_help_flag_prints_help_and_exits_0_without_configuration(
    monkeypatch, capsys, flag
):
    _forbid_configuration(monkeypatch)

    with pytest.raises(SystemExit) as exit_info:
        cli.main([flag])

    assert exit_info.value.code == 0
    captured = capsys.readouterr()
    assert "framework-advisor" in captured.out
    assert "OPENAI_API_KEY" in captured.out
    assert "/new" in captured.out
    assert captured.err == ""


def test_unknown_argument_exits_with_usage_error_without_configuration(
    monkeypatch, capsys
):
    _forbid_configuration(monkeypatch)

    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--no-such-flag"])

    assert exit_info.value.code == 2
    assert "usage" in capsys.readouterr().err.casefold()


@pytest.mark.parametrize("exit_code", [0, 1])
def test_main_without_arguments_runs_interactive_cli(monkeypatch, exit_code):
    monkeypatch.setattr(cli, "run_cli", lambda: exit_code)

    with pytest.raises(SystemExit) as exit_info:
        cli.main([])

    assert exit_info.value.code == exit_code


@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_help_works_in_real_process_without_api_key(tmp_path, monkeypatch, flag):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.setenv("PYTHONIOENCODING", "utf-8")

    completed = subprocess.run(
        [sys.executable, "-m", "advisor.cli", flag],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )

    assert completed.returncode == 0
    assert "usage: framework-advisor" in completed.stdout
    assert completed.stderr == ""


# --- StructuredOutputValidationError ---


def test_structured_output_validation_error_signature_is_supported():
    sig = signature(StructuredOutputValidationError.__init__)

    assert list(sig.parameters) == ["self", "tool_name", "source", "ai_message"]


def test_run_cli_recovers_from_structured_output_validation_error(
    monkeypatch, capsys
):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    structured_output_error = StructuredOutputValidationError(
        tool_name="AdvisorResponse",
        source=ValueError("invalid structured payload"),
        ai_message=AIMessage(content=""),
    )
    agent = QueueAgent([structured_output_error])
    monkeypatch.setattr(cli, "build_advisor_agent", lambda: agent)
    _patch_input(monkeypatch, ["Сломанный структурированный ответ", "/exit"])

    exit_code = cli.run_cli()

    assert exit_code == 0
    assert len(agent.calls) == 1
    captured = capsys.readouterr()
    assert "структурированный ответ" in captured.out.casefold()
    assert "повтор" in captured.out.casefold()
    assert "Traceback" not in captured.out
    assert "Traceback" not in captured.err
    assert captured.err == ""
