from __future__ import annotations

from advisor.framework_data import FRAMEWORK_PROFILES
from advisor.tools import compare_frameworks


def test_compare_two_frameworks_by_one_criterion():
    result = compare_frameworks.invoke(
        {
            "framework_names": ["LangChain", "LlamaIndex"],
            "criteria": ["agent_support"],
        }
    )

    assert result["ok"] is True
    assert result["frameworks"] == ["langchain", "llamaindex"]
    assert result["criteria"] == ["agent_support"]
    assert result["display_names"] == {
        "langchain": "LangChain",
        "llamaindex": "LlamaIndex",
    }
    assert result["criterion_labels"] == {
        "agent_support": "Поддержка агентов",
    }
    assert set(result["comparison"]["agent_support"].keys()) == {
        "langchain",
        "llamaindex",
    }
    assert result["comparison"]["agent_support"]["langchain"] == (
        FRAMEWORK_PROFILES["langchain"]["agent_support"]
    )
    assert result["comparison"]["agent_support"]["llamaindex"] == (
        FRAMEWORK_PROFILES["llamaindex"]["agent_support"]
    )


def test_compare_all_five_frameworks_by_three_criteria():
    result = compare_frameworks.invoke(
        {
            "framework_names": [
                "LangChain",
                "LlamaIndex",
                "Haystack",
                "Semantic Kernel",
                "CrewAI",
            ],
            "criteria": ["primary_use_case", "learning_curve", "limitations"],
        }
    )

    assert result["ok"] is True
    assert result["frameworks"] == [
        "langchain",
        "llamaindex",
        "haystack",
        "semantic_kernel",
        "crewai",
    ]
    assert result["criteria"] == [
        "primary_use_case",
        "learning_curve",
        "limitations",
    ]
    for criterion in result["criteria"]:
        assert set(result["comparison"][criterion].keys()) == set(
            result["frameworks"]
        )


def test_compare_normalizes_framework_names_and_criteria():
    result = compare_frameworks.invoke(
        {
            "framework_names": ["Crew-AI", "semantic_kernel"],
            "criteria": ["Agent Support", "learning-curve"],
        }
    )

    assert result["ok"] is True
    assert result["frameworks"] == ["crewai", "semantic_kernel"]
    assert result["criteria"] == ["agent_support", "learning_curve"]


def test_compare_deduplicates_framework_names_preserving_order():
    result = compare_frameworks.invoke(
        {
            "framework_names": ["LangChain", "lang chain", "CrewAI"],
            "criteria": ["agent_support"],
        }
    )

    assert result["ok"] is True
    assert result["frameworks"] == ["langchain", "crewai"]


def test_compare_fails_with_single_unique_framework_after_dedup():
    result = compare_frameworks.invoke(
        {
            "framework_names": ["LangChain", "lang chain"],
            "criteria": ["agent_support"],
        }
    )

    assert result["ok"] is False
    assert result["error"]


def test_compare_fails_with_empty_framework_names():
    result = compare_frameworks.invoke(
        {
            "framework_names": [],
            "criteria": ["agent_support"],
        }
    )

    assert result["ok"] is False
    assert result["error"]


def test_compare_fails_with_unknown_framework():
    result = compare_frameworks.invoke(
        {
            "framework_names": ["LangChain", "Django"],
            "criteria": ["agent_support"],
        }
    )

    assert result["ok"] is False
    assert result["error"]
    assert set(result["supported_frameworks"]) == {
        "LangChain",
        "LlamaIndex",
        "Haystack",
        "Semantic Kernel",
        "CrewAI",
    }
    assert set(result["supported_criteria"].keys()) == {
        "primary_use_case",
        "agent_support",
        "rag_support",
        "integration_ecosystem",
        "learning_curve",
        "typical_components",
        "limitations",
    }


def test_compare_fails_with_empty_criteria():
    result = compare_frameworks.invoke(
        {
            "framework_names": ["LangChain", "CrewAI"],
            "criteria": [],
        }
    )

    assert result["ok"] is False
    assert result["error"]


def test_compare_fails_with_unknown_criterion():
    result = compare_frameworks.invoke(
        {
            "framework_names": ["LangChain", "CrewAI"],
            "criteria": ["price"],
        }
    )

    assert result["ok"] is False
    assert result["error"]


def test_compare_deduplicates_criteria_preserving_order():
    result = compare_frameworks.invoke(
        {
            "framework_names": ["LangChain", "CrewAI"],
            "criteria": ["rag_support", "RAG Support", "learning_curve"],
        }
    )

    assert result["ok"] is True
    assert result["criteria"] == ["rag_support", "learning_curve"]


def test_compare_does_not_leak_global_state():
    first_result = compare_frameworks.invoke(
        {
            "framework_names": ["LangChain", "CrewAI"],
            "criteria": ["limitations"],
        }
    )
    first_result["comparison"]["limitations"]["langchain"][0] = "ИЗМЕНЕНО"

    second_result = compare_frameworks.invoke(
        {
            "framework_names": ["LangChain", "CrewAI"],
            "criteria": ["limitations"],
        }
    )

    assert second_result["comparison"]["limitations"]["langchain"][0] != "ИЗМЕНЕНО"
    assert FRAMEWORK_PROFILES["langchain"]["limitations"][0] != "ИЗМЕНЕНО"
