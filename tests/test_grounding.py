"""Тесты проверки привязки ответа к локальным данным текущего хода.

Основные тесты работают с настоящим собранным агентом (build_advisor_agent +
create_agent + InMemorySaver) и фейковой моделью без сетевых вызовов; часть
граничных случаев проверяется на готовых списках настоящих сообщений LangChain.
"""

from __future__ import annotations

import copy
import json

import pytest
from fake_model import (
    build_scripted_agent,
    compare_call,
    comparison_item,
    final_answer,
    profile_call,
    tool_call,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import advisor.cli as cli
from advisor.framework_data import FRAMEWORK_PROFILES, SUPPORTED_CRITERIA
from advisor.grounding import GroundingError, verify_grounded_response
from advisor.schema import AdvisorResponse
from advisor.tools import compare_frameworks, get_framework_profile

CONFIG = {"configurable": {"thread_id": "t"}}


def _ask(agent, text="Нужен проект", thread_id="t"):
    return cli.invoke_agent(agent, user_message=text, thread_id=thread_id)


# --- допустимые ответы ---


def test_valid_recommendation_after_successful_profile_call(monkeypatch):
    agent, _ = build_scripted_agent(
        monkeypatch, [profile_call("langchain"), final_answer()]
    )

    response = _ask(agent)

    assert isinstance(response, AdvisorResponse)
    assert response.recommended_framework == "langchain"


def test_valid_comparison_after_successful_compare_call(monkeypatch):
    agent, _ = build_scripted_agent(
        monkeypatch,
        [
            compare_call(["LangChain", "Llama Index"]),
            final_answer(
                recommended_framework="llamaindex",
                comparison=[
                    comparison_item("langchain"),
                    comparison_item("llamaindex"),
                ],
                suitable_components=["Indexes", "Retrievers"],
            ),
        ],
    )

    response = _ask(agent)

    assert response.recommended_framework == "llamaindex"
    assert [item.framework for item in response.comparison] == [
        "langchain",
        "llamaindex",
    ]


def test_failed_call_followed_by_successful_retry_is_accepted(monkeypatch):
    agent, _ = build_scripted_agent(
        monkeypatch,
        [
            profile_call("django", call_id="bad"),
            profile_call("langchain", call_id="good"),
            final_answer(),
        ],
    )

    assert _ask(agent).recommended_framework == "langchain"


# --- нет успешного вызова локального инструмента ---


def test_direct_structured_output_without_data_tool_is_rejected(monkeypatch):
    agent, _ = build_scripted_agent(monkeypatch, [final_answer()])

    with pytest.raises(GroundingError, match="нет успешного вызова"):
        _ask(agent)


@pytest.mark.parametrize(
    "failing_call",
    [
        pytest.param(profile_call("django"), id="unknown-framework"),
        pytest.param(
            compare_call(["langchain", "llamaindex"], ["no_such_criterion"]),
            id="unknown-criterion",
        ),
        pytest.param(compare_call(["langchain"]), id="single-framework-compare"),
        pytest.param(
            tool_call("get_framework_profile", {}, "p1"),
            id="invalid-arguments-error-status",
        ),
    ],
)
def test_failed_data_tool_result_is_rejected(monkeypatch, failing_call):
    agent, _ = build_scripted_agent(monkeypatch, [failing_call, final_answer()])

    with pytest.raises(GroundingError, match="нет успешного вызова"):
        _ask(agent)


# --- память: старый вызов не подтверждает новый ход ---


def test_prior_turn_tool_call_does_not_ground_new_turn(monkeypatch):
    agent, _ = build_scripted_agent(
        monkeypatch,
        [
            profile_call("langchain"),
            final_answer(),
            final_answer(),  # ход 2: без вызова инструмента
            profile_call("langchain", call_id="p2"),
            final_answer(),  # ход 3: со свежим вызовом
        ],
    )

    assert _ask(agent, "Ход 1").recommended_framework == "langchain"

    with pytest.raises(GroundingError, match="нет успешного вызова"):
        _ask(agent, "Ход 2")

    # Сканирование всего треда приняло бы ход 2 из-за вызова в ходе 1.
    thread_messages = agent.get_state(CONFIG).values["messages"]
    assert any(
        isinstance(m, ToolMessage) and m.name == "get_framework_profile"
        for m in thread_messages
    )

    assert _ask(agent, "Ход 3").recommended_framework == "langchain"


def test_prior_turn_call_with_same_tool_call_id_does_not_ground_new_turn(
    monkeypatch,
):
    agent, _ = build_scripted_agent(
        monkeypatch,
        [profile_call("langchain", call_id="same"), final_answer(), final_answer()],
    )

    _ask(agent, "Ход 1")

    with pytest.raises(GroundingError):
        _ask(agent, "Ход 2")


# --- семантическая проверка ---


def test_invented_suitable_component_is_rejected(monkeypatch):
    agent, _ = build_scripted_agent(
        monkeypatch,
        [
            profile_call("langchain"),
            final_answer(suitable_components=["Tools", "QuantumRetriever"]),
        ],
    )

    with pytest.raises(GroundingError, match="QuantumRetriever"):
        _ask(agent)


def test_component_of_another_framework_is_rejected(monkeypatch):
    agent, _ = build_scripted_agent(
        monkeypatch,
        [
            profile_call("langchain"),
            final_answer(suitable_components=["Crews"]),
        ],
    )

    with pytest.raises(GroundingError, match="Crews"):
        _ask(agent)


@pytest.mark.parametrize(
    "components",
    [
        ["tools", "  STRUCTURED   output "],
        ["create-agent", "langgraph"],
    ],
)
def test_components_use_repository_normalization(monkeypatch, components):
    agent, _ = build_scripted_agent(
        monkeypatch,
        [profile_call("langchain"), final_answer(suitable_components=components)],
    )

    assert _ask(agent).suitable_components == [c.strip() for c in components]


def test_parts_of_compound_component_names_are_accepted(monkeypatch):
    agent, _ = build_scripted_agent(
        monkeypatch,
        [
            profile_call("llamaindex"),
            final_answer(
                recommended_framework="llamaindex",
                suitable_components=["Documents", "Nodes", "Documents/Nodes"],
            ),
        ],
    )

    assert _ask(agent).recommended_framework == "llamaindex"


def test_duplicate_comparison_framework_is_rejected(monkeypatch):
    agent, _ = build_scripted_agent(
        monkeypatch,
        [
            compare_call(["langchain", "llamaindex"]),
            final_answer(
                comparison=[
                    comparison_item("langchain"),
                    comparison_item("langchain"),
                ],
            ),
        ],
    )

    with pytest.raises(GroundingError, match="повторя"):
        _ask(agent)


def test_comparison_without_compare_call_is_rejected(monkeypatch):
    agent, _ = build_scripted_agent(
        monkeypatch,
        [
            profile_call("langchain"),
            final_answer(
                comparison=[
                    comparison_item("langchain"),
                    comparison_item("llamaindex"),
                ],
            ),
        ],
    )

    with pytest.raises(GroundingError, match="нет успешного вызова compare_frameworks"):
        _ask(agent)


def test_comparison_with_extra_framework_beyond_compared_set_is_rejected(
    monkeypatch,
):
    agent, _ = build_scripted_agent(
        monkeypatch,
        [
            compare_call(["langchain", "llamaindex"]),
            final_answer(
                comparison=[
                    comparison_item("langchain"),
                    comparison_item("llamaindex"),
                    comparison_item("haystack"),
                ],
            ),
        ],
    )

    with pytest.raises(GroundingError, match="не совпадает"):
        _ask(agent)


def test_profile_call_does_not_ground_a_comparison_entry(monkeypatch):
    agent, _ = build_scripted_agent(
        monkeypatch,
        [
            profile_call("haystack", call_id="p1"),
            compare_call(["langchain", "llamaindex"], call_id="c1"),
            final_answer(
                comparison=[
                    comparison_item("langchain"),
                    comparison_item("llamaindex"),
                    comparison_item("haystack"),
                ],
            ),
        ],
    )

    with pytest.raises(GroundingError, match="не совпадает"):
        _ask(agent)


def test_comparison_with_framework_not_compared_by_tool_is_rejected(monkeypatch):
    agent, _ = build_scripted_agent(
        monkeypatch,
        [
            compare_call(["langchain", "llamaindex"]),
            final_answer(
                comparison=[
                    comparison_item("langchain"),
                    comparison_item("haystack"),
                ],
            ),
        ],
    )

    with pytest.raises(GroundingError, match="не совпадает"):
        _ask(agent)


def test_comparison_dropping_a_compared_framework_is_rejected(monkeypatch):
    agent, _ = build_scripted_agent(
        monkeypatch,
        [
            compare_call(["langchain", "llamaindex", "haystack"]),
            final_answer(
                comparison=[
                    comparison_item("langchain"),
                    comparison_item("llamaindex"),
                ],
            ),
        ],
    )

    with pytest.raises(GroundingError, match="не совпадает"):
        _ask(agent)


def test_recommended_framework_missing_from_comparison_is_rejected(monkeypatch):
    agent, _ = build_scripted_agent(
        monkeypatch,
        [
            compare_call(["langchain", "llamaindex"]),
            final_answer(
                recommended_framework="langchain",
                comparison=[comparison_item("llamaindex")],
            ),
        ],
    )

    with pytest.raises(GroundingError, match="отсутствует в сравнении"):
        _ask(agent)


def test_recommendation_not_in_profile_data_is_rejected(monkeypatch):
    agent, _ = build_scripted_agent(
        monkeypatch,
        [
            profile_call("langchain"),
            final_answer(
                recommended_framework="crewai", suitable_components=["Crews"]
            ),
        ],
    )

    with pytest.raises(GroundingError, match="crewai"):
        _ask(agent)


def test_recommendation_not_in_compared_frameworks_is_rejected(monkeypatch):
    agent, _ = build_scripted_agent(
        monkeypatch,
        [
            compare_call(["langchain", "llamaindex"]),
            final_answer(
                recommended_framework="haystack",
                suitable_components=["Pipelines"],
            ),
        ],
    )

    with pytest.raises(GroundingError, match="haystack"):
        _ask(agent)


def test_empty_comparison_is_not_manufactured_or_required(monkeypatch):
    agent, _ = build_scripted_agent(
        monkeypatch,
        [
            compare_call(["langchain", "llamaindex"]),
            final_answer(comparison=[]),
        ],
    )

    assert _ask(agent).comparison == []


# --- граничные случаи на готовых сообщениях ---


def _profile_ok(call_id="p1", framework="langchain"):
    call = {
        "name": "get_framework_profile",
        "args": {"framework_name": framework},
        "id": call_id,
    }
    return call, AIMessage(content="", tool_calls=[call])


def _response(**overrides):
    payload = {
        "recommended_framework": "langchain",
        "reasoning": ["Причина"],
        "suitable_components": ["Tools"],
        "risks": ["Риск"],
    }
    payload.update(overrides)
    return AdvisorResponse(**payload)


def _real_tool_message(call):
    tool = {
        "get_framework_profile": get_framework_profile,
        "compare_frameworks": compare_frameworks,
    }[call["name"]]
    return tool.invoke({**call, "type": "tool_call"})


def test_verifier_accepts_real_trace():
    call, ai = _profile_ok()
    messages = [HumanMessage("q"), ai, _real_tool_message(call)]

    assert verify_grounded_response(messages, _response()) is not None


def test_verifier_rejects_tool_call_without_tool_message():
    _, ai = _profile_ok()

    with pytest.raises(GroundingError):
        verify_grounded_response([HumanMessage("q"), ai], _response())


def test_verifier_rejects_error_status_tool_message():
    call, ai = _profile_ok()
    real = _real_tool_message(call)
    errored = ToolMessage(
        content=real.content,
        name=call["name"],
        tool_call_id=call["id"],
        status="error",
    )

    with pytest.raises(GroundingError):
        verify_grounded_response([HumanMessage("q"), ai, errored], _response())


@pytest.mark.parametrize(
    "content",
    ["not json", "[]", '{"ok": "true"}', '{"framework": "langchain"}'],
)
def test_verifier_rejects_unparseable_or_not_ok_content(content):
    call, ai = _profile_ok()
    message = ToolMessage(content=content, name=call["name"], tool_call_id=call["id"])

    with pytest.raises(GroundingError):
        verify_grounded_response([HumanMessage("q"), ai, message], _response())


@pytest.mark.parametrize(
    ("tool_name", "content"),
    [
        ("get_framework_profile", '{"ok": true, "framework": "django"}'),
        ("get_framework_profile", '{"ok": true}'),
        ("compare_frameworks", '{"ok": true, "frameworks": ["langchain", "django"]}'),
        ("compare_frameworks", '{"ok": true, "frameworks": []}'),
        ("compare_frameworks", '{"ok": true, "frameworks": "langchain"}'),
    ],
)
def test_verifier_rejects_ok_payload_with_unknown_or_malformed_frameworks(
    tool_name, content
):
    call = {"name": tool_name, "args": {}, "id": "x1"}
    ai = AIMessage(content="", tool_calls=[call])
    message = ToolMessage(content=content, name=tool_name, tool_call_id="x1")

    with pytest.raises(GroundingError):
        verify_grounded_response([HumanMessage("q"), ai, message], _response())


@pytest.mark.parametrize("other_tool", ["some_other_tool", "AdvisorResponse"])
@pytest.mark.parametrize(
    "data_call",
    [
        {
            "name": "get_framework_profile",
            "args": {"framework_name": "langchain"},
            "id": "o1",
        },
        {
            "name": "compare_frameworks",
            "args": {
                "framework_names": ["langchain", "llamaindex"],
                "criteria": ["agent_support"],
            },
            "id": "o1",
        },
    ],
    ids=["profile-shaped", "compare-shaped"],
)
def test_verifier_ignores_results_of_tools_outside_the_data_tools(
    other_tool, data_call
):
    real = _real_tool_message(data_call)  # валидный успешный результат
    other_call = {"name": other_tool, "args": {}, "id": "o1"}
    ai = AIMessage(content="", tool_calls=[other_call])
    message = ToolMessage(content=real.content, name=other_tool, tool_call_id="o1")

    with pytest.raises(GroundingError):
        verify_grounded_response([HumanMessage("q"), ai, message], _response())


def test_verifier_rejects_tool_message_with_unmatched_call_id():
    call, ai = _profile_ok(call_id="p1")
    real = _real_tool_message(call)
    other = ToolMessage(content=real.content, name=call["name"], tool_call_id="other")

    with pytest.raises(GroundingError):
        verify_grounded_response([HumanMessage("q"), ai, other], _response())


def test_verifier_does_not_count_structured_output_tool():
    ai = tool_call("AdvisorResponse", {"recommended_framework": "langchain"}, "s1")
    done = ToolMessage(
        content="Returning structured response",
        name="AdvisorResponse",
        tool_call_id="s1",
    )

    with pytest.raises(GroundingError):
        verify_grounded_response([HumanMessage("q"), ai, done], _response())


def test_verifier_uses_only_messages_after_last_human_message():
    call, ai = _profile_ok()
    messages = [
        HumanMessage("turn 1"),
        ai,
        _real_tool_message(call),
        AIMessage(content="answer 1"),
        HumanMessage("turn 2"),
        AIMessage(content="answer 2 without tools"),
    ]

    with pytest.raises(GroundingError):
        verify_grounded_response(messages, _response())


@pytest.mark.parametrize("messages", [None, "text", [], [AIMessage(content="x")]])
def test_verifier_rejects_missing_trace_or_turn_boundary(messages):
    with pytest.raises(GroundingError):
        verify_grounded_response(messages, _response())


# --- строгая проверка трассы и структуры результатов инструментов ---
#
# Все «испорченные» результаты получены из настоящего вывода инструментов,
# а сообщения — настоящие AIMessage/ToolMessage LangChain.

PROFILE_CALL = {
    "name": "get_framework_profile",
    "args": {"framework_name": "langchain"},
    "id": "p1",
}
COMPARE_CALL = {
    "name": "compare_frameworks",
    "args": {
        "framework_names": ["langchain", "llamaindex"],
        "criteria": ["agent_support", "typical_components"],
    },
    "id": "c1",
}


def _ai(*calls):
    return AIMessage(content="", tool_calls=list(calls))


def _payload(call):
    """Настоящий успешный результат инструмента как dict."""
    return json.loads(_real_tool_message(call).content)


def _tool_message(call, content, *, name="same", status="success"):
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    return ToolMessage(
        content=content,
        name=call["name"] if name == "same" else name,
        tool_call_id=call["id"],
        status=status,
    )


def _mutated(payload, mutate):
    payload = copy.deepcopy(payload)
    mutate(payload)
    return payload


def _comparison_response(*frameworks):
    frameworks = frameworks or ("langchain", "llamaindex")
    return _response(comparison=[comparison_item(f) for f in frameworks])


def _verify_profile(content, **message_kwargs):
    messages = [
        HumanMessage("q"),
        _ai(PROFILE_CALL),
        _tool_message(PROFILE_CALL, content, **message_kwargs),
    ]
    return verify_grounded_response(messages, _response())


def _verify_compare(content, response=None, **message_kwargs):
    messages = [
        HumanMessage("q"),
        _ai(COMPARE_CALL),
        _tool_message(COMPARE_CALL, content, **message_kwargs),
    ]
    return verify_grounded_response(messages, response or _comparison_response())


def _single_framework_compare_payload():
    """Самосогласованный результат «сравнения» одного фреймворка."""
    payload = _payload(COMPARE_CALL)
    payload["frameworks"] = ["langchain"]
    payload["display_names"] = {"langchain": "LangChain"}
    payload["comparison"] = {
        criterion: {"langchain": row["langchain"]}
        for criterion, row in payload["comparison"].items()
    }
    return payload


# 9–10. Корректные результаты по-прежнему принимаются.


def test_valid_profile_result_is_still_accepted():
    assert _verify_profile(_payload(PROFILE_CALL)) is not None


def test_valid_comparison_result_is_still_accepted():
    assert _verify_compare(_payload(COMPARE_CALL)) is not None


@pytest.mark.parametrize("framework", sorted(FRAMEWORK_PROFILES))
def test_every_stored_profile_is_accepted_as_a_profile_result(framework):
    call = {
        "name": "get_framework_profile",
        "args": {"framework_name": framework},
        "id": "p1",
    }
    messages = [HumanMessage("q"), _ai(call), _real_tool_message(call)]
    response = _response(
        recommended_framework=framework,
        suitable_components=[FRAMEWORK_PROFILES[framework]["typical_components"][0]],
    )

    assert verify_grounded_response(messages, response) is not None


@pytest.mark.parametrize(
    "frameworks",
    [
        ["haystack", "crewai"],
        ["semantic_kernel", "langchain", "llamaindex"],
        ["langchain", "llamaindex", "haystack", "semantic_kernel", "crewai"],
    ],
)
def test_real_comparison_results_of_any_size_are_accepted(frameworks):
    call = {
        "name": "compare_frameworks",
        "args": {"framework_names": frameworks, "criteria": list(SUPPORTED_CRITERIA)},
        "id": "c1",
    }
    first = frameworks[0]
    response = _response(
        recommended_framework=first,
        suitable_components=[FRAMEWORK_PROFILES[first]["typical_components"][0]],
        comparison=[comparison_item(f) for f in frameworks],
    )
    messages = [HumanMessage("q"), _ai(call), _real_tool_message(call)]

    assert verify_grounded_response(messages, response) is not None


def test_failed_result_followed_by_valid_retry_is_accepted():
    bad = {**PROFILE_CALL, "id": "bad"}
    failed = {"ok": False, "error": "x", "supported_frameworks": []}
    messages = [
        HumanMessage("q"),
        _ai(bad),
        _tool_message(bad, failed),
        _ai(PROFILE_CALL),
        _tool_message(PROFILE_CALL, _payload(PROFILE_CALL)),
    ]

    assert verify_grounded_response(messages, _response()) is not None


def test_error_status_result_followed_by_valid_retry_is_accepted():
    bad = {**PROFILE_CALL, "id": "bad"}
    messages = [
        HumanMessage("q"),
        _ai(bad),
        _tool_message(bad, "Error invoking tool", status="error"),
        _ai(PROFILE_CALL),
        _tool_message(PROFILE_CALL, _payload(PROFILE_CALL)),
    ]

    assert verify_grounded_response(messages, _response()) is not None


def test_data_tool_called_in_parallel_with_structured_output_is_accepted(
    monkeypatch,
):
    """Настоящий агент выполняет инструмент данных и в паре с AdvisorResponse."""
    both = _ai(
        profile_call("langchain", call_id="p1").tool_calls[0],
        final_answer(call_id="s1").tool_calls[0],
    )
    agent, _ = build_scripted_agent(monkeypatch, [both])

    assert _ask(agent).recommended_framework == "langchain"


# 1–2. Профиль: полная структура, согласованная с локальными данными.

PROFILE_MUTATIONS = {
    "missing-profile": lambda p: p.pop("profile"),
    "missing-framework": lambda p: p.pop("framework"),
    "missing-ok": lambda p: p.pop("ok"),
    "extra-top-level-key": lambda p: p.update(extra=1),
    "profile-empty": lambda p: p.update(profile={}),
    "profile-none": lambda p: p.update(profile=None),
    "profile-not-an-object": lambda p: p.update(profile="text"),
    "profile-missing-field": lambda p: p["profile"].pop("limitations"),
    "profile-missing-components": lambda p: p["profile"].pop("typical_components"),
    "profile-extra-field": lambda p: p["profile"].update(invented="x"),
    "profile-altered-text": lambda p: p["profile"].update(primary_use_case="иное"),
    "profile-component-dropped": lambda p: p["profile"]["typical_components"].pop(),
    "profile-of-another-framework": lambda p: p.update(
        profile=copy.deepcopy(FRAMEWORK_PROFILES["crewai"])
    ),
    "unknown-framework": lambda p: p.update(framework="django"),
    "non-canonical-framework": lambda p: p.update(framework="LangChain"),
    "framework-not-a-string": lambda p: p.update(framework=["langchain"]),
}


@pytest.mark.parametrize("mutate", PROFILE_MUTATIONS.values(), ids=PROFILE_MUTATIONS)
def test_malformed_or_incomplete_profile_result_is_rejected(mutate):
    with pytest.raises(GroundingError, match="неверную структуру"):
        _verify_profile(_mutated(_payload(PROFILE_CALL), mutate))


def test_profile_result_with_only_ok_and_framework_is_rejected():
    with pytest.raises(GroundingError, match="неверную структуру"):
        _verify_profile({"ok": True, "framework": "langchain"})


# 4–5. Сравнение: 2–5 уникальных фреймворков и полная структура.

COMPARE_MUTATIONS = {
    "missing-comparison": lambda p: p.pop("comparison"),
    "missing-criteria": lambda p: p.pop("criteria"),
    "missing-frameworks": lambda p: p.pop("frameworks"),
    "missing-display-names": lambda p: p.pop("display_names"),
    "missing-criterion-labels": lambda p: p.pop("criterion_labels"),
    "missing-ok": lambda p: p.pop("ok"),
    "extra-top-level-key": lambda p: p.update(extra=1),
    "comparison-empty": lambda p: p.update(comparison={}),
    "comparison-not-an-object": lambda p: p.update(comparison=[]),
    "comparison-lacks-framework-column": lambda p: [
        row.pop("llamaindex") for row in p["comparison"].values()
    ],
    "comparison-lacks-criterion-row": lambda p: p["comparison"].pop("agent_support"),
    "comparison-altered-value": lambda p: p["comparison"]["agent_support"].update(
        langchain="иное"
    ),
    "criteria-empty": lambda p: p.update(criteria=[]),
    "criteria-unknown": lambda p: p.update(criteria=[*p["criteria"], "vibes"]),
    "criteria-duplicate": lambda p: p.update(criteria=[*p["criteria"], "agent_support"]),
    "frameworks-duplicate": lambda p: p.update(frameworks=["langchain", "langchain"]),
    "frameworks-unknown": lambda p: p.update(frameworks=["langchain", "django"]),
    "frameworks-not-a-list": lambda p: p.update(frameworks="langchain"),
    "frameworks-not-strings": lambda p: p.update(frameworks=[["langchain"], ["x"]]),
    "display-names-altered": lambda p: p["display_names"].update(langchain="X"),
    "criterion-labels-altered": lambda p: p["criterion_labels"].update(
        agent_support="X"
    ),
}


@pytest.mark.parametrize("mutate", COMPARE_MUTATIONS.values(), ids=COMPARE_MUTATIONS)
def test_comparison_result_missing_structure_or_inconsistent_is_rejected(mutate):
    with pytest.raises(GroundingError, match="неверную структуру"):
        _verify_compare(_mutated(_payload(COMPARE_CALL), mutate))


@pytest.mark.parametrize(
    "response",
    [
        pytest.param(_comparison_response("langchain", "llamaindex"), id="two-items"),
        pytest.param(_comparison_response("langchain"), id="one-item"),
        pytest.param(_response(), id="no-comparison"),
    ],
)
def test_single_framework_comparison_result_is_rejected(response):
    with pytest.raises(GroundingError, match="неверную структуру"):
        _verify_compare(_single_framework_compare_payload(), response)


def test_self_consistent_result_listing_one_framework_twice_is_rejected():
    payload = _single_framework_compare_payload()
    payload["frameworks"] = ["langchain", "langchain"]

    with pytest.raises(GroundingError, match="неверную структуру"):
        _verify_compare(payload)


def test_self_consistent_result_with_no_criteria_is_rejected():
    payload = _payload(COMPARE_CALL)
    payload["criteria"] = []
    payload["criterion_labels"] = {}
    payload["comparison"] = {}

    with pytest.raises(GroundingError, match="неверную структуру"):
        _verify_compare(payload)


def test_bare_comparison_results_are_rejected():
    for content in (
        {"ok": True, "frameworks": ["langchain"]},
        {"ok": True, "frameworks": ["langchain", "llamaindex"]},
    ):
        with pytest.raises(GroundingError, match="неверную структуру"):
            _verify_compare(content)


# 3. Имя ToolMessage должно совпадать с именем вызова.


@pytest.mark.parametrize(
    "wrong_name",
    ["compare_frameworks", "AdvisorResponse", "some_other_tool", "", None],
)
def test_profile_result_with_mismatched_tool_name_is_rejected(wrong_name):
    with pytest.raises(GroundingError, match="другому инструменту"):
        _verify_profile(_payload(PROFILE_CALL), name=wrong_name)


@pytest.mark.parametrize("wrong_name", ["get_framework_profile", "AdvisorResponse", None])
def test_comparison_result_with_mismatched_tool_name_is_rejected(wrong_name):
    with pytest.raises(GroundingError, match="другому инструменту"):
        _verify_compare(_payload(COMPARE_CALL), name=wrong_name)


def test_valid_result_of_another_data_tool_cannot_satisfy_a_call_by_id():
    # ID совпадает, содержимое — валидный результат compare, но вызов был profile.
    with pytest.raises(GroundingError, match="другому инструменту"):
        _verify_profile(_payload(COMPARE_CALL), name="compare_frameworks")


# 6–8. Ровно один результат на вызов.


def test_duplicate_successful_results_for_one_call_id_are_rejected():
    result = _tool_message(PROFILE_CALL, _payload(PROFILE_CALL))
    messages = [HumanMessage("q"), _ai(PROFILE_CALL), result, result.model_copy()]

    with pytest.raises(GroundingError, match="больше одного результата"):
        verify_grounded_response(messages, _response())


def test_duplicate_successful_results_with_different_content_are_rejected():
    messages = [
        HumanMessage("q"),
        _ai(PROFILE_CALL),
        _tool_message(PROFILE_CALL, _payload(PROFILE_CALL)),
        _tool_message(PROFILE_CALL, {"ok": True, "framework": "langchain"}),
    ]

    with pytest.raises(GroundingError, match="больше одного результата"):
        verify_grounded_response(messages, _response())


@pytest.mark.parametrize("success_first", [True, False], ids=["success-then-error", "error-then-success"])
def test_success_plus_duplicate_error_result_is_rejected(success_first):
    success = _tool_message(PROFILE_CALL, _payload(PROFILE_CALL))
    error = _tool_message(PROFILE_CALL, "Error invoking tool", status="error")
    results = [success, error] if success_first else [error, success]

    with pytest.raises(GroundingError, match="больше одного результата"):
        verify_grounded_response(
            [HumanMessage("q"), _ai(PROFILE_CALL), *results], _response()
        )


def test_success_plus_duplicate_failed_ok_false_result_is_rejected():
    failed = {"ok": False, "error": "x", "supported_frameworks": []}
    messages = [
        HumanMessage("q"),
        _ai(PROFILE_CALL),
        _tool_message(PROFILE_CALL, _payload(PROFILE_CALL)),
        _tool_message(PROFILE_CALL, failed),
    ]

    with pytest.raises(GroundingError, match="больше одного результата"):
        verify_grounded_response(messages, _response())


def test_call_without_any_result_is_rejected():
    with pytest.raises(GroundingError, match="нет результата"):
        verify_grounded_response(
            [HumanMessage("q"), _ai(PROFILE_CALL)], _response()
        )


def test_result_that_precedes_its_call_is_rejected():
    messages = [
        HumanMessage("q"),
        _tool_message(PROFILE_CALL, _payload(PROFILE_CALL)),
        _ai(PROFILE_CALL),
    ]

    with pytest.raises(GroundingError, match="нет результата"):
        verify_grounded_response(messages, _response())


def test_result_for_a_different_call_id_does_not_count_as_a_result():
    other = {**PROFILE_CALL, "id": "other"}
    messages = [
        HumanMessage("q"),
        _ai(PROFILE_CALL),
        _tool_message(other, _payload(PROFILE_CALL)),
    ]

    with pytest.raises(GroundingError, match="нет результата"):
        verify_grounded_response(messages, _response())


def test_one_dangling_call_rejects_the_turn_even_if_another_call_succeeded():
    dangling = {**PROFILE_CALL, "id": "p2"}
    messages = [
        HumanMessage("q"),
        _ai(PROFILE_CALL),
        _tool_message(PROFILE_CALL, _payload(PROFILE_CALL)),
        _ai(dangling),
    ]

    with pytest.raises(GroundingError, match="нет результата"):
        verify_grounded_response(messages, _response())


@pytest.mark.parametrize("call_id", [None, ""])
def test_data_call_without_id_is_rejected(call_id):
    call = {**PROFILE_CALL, "id": call_id}
    messages = [HumanMessage("q"), _ai(call)]

    with pytest.raises(GroundingError, match="идентификатор"):
        verify_grounded_response(messages, _response())


def test_call_id_reused_by_two_data_calls_is_rejected():
    reused = {**PROFILE_CALL, "args": {"framework_name": "haystack"}}
    messages = [
        HumanMessage("q"),
        _ai(PROFILE_CALL),
        _tool_message(PROFILE_CALL, _payload(PROFILE_CALL)),
        _ai(reused),
        _tool_message(reused, _payload(PROFILE_CALL)),
    ]

    with pytest.raises(GroundingError, match="идентификатор"):
        verify_grounded_response(messages, _response())


def test_structured_output_result_sharing_a_data_call_id_is_rejected():
    stray = ToolMessage(
        content="Returning structured response",
        name="AdvisorResponse",
        tool_call_id=PROFILE_CALL["id"],
    )
    messages = [
        HumanMessage("q"),
        _ai(PROFILE_CALL),
        _tool_message(PROFILE_CALL, _payload(PROFILE_CALL)),
        stray,
    ]

    with pytest.raises(GroundingError, match="больше одного результата"):
        verify_grounded_response(messages, _response())
