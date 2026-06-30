from __future__ import annotations

import pytest

from advisor.framework_data import FRAMEWORK_PROFILES
from advisor.tools import get_framework_profile

EXPECTED_PROFILE_FIELDS = {
    "display_name",
    "primary_use_case",
    "agent_support",
    "rag_support",
    "integration_ecosystem",
    "learning_curve",
    "typical_components",
    "limitations",
}

EXPECTED_DISPLAY_NAMES = {
    "LangChain",
    "LlamaIndex",
    "Haystack",
    "Semantic Kernel",
    "CrewAI",
}


def test_canonical_name_langchain():
    result = get_framework_profile.invoke({"framework_name": "langchain"})

    assert result["ok"] is True
    assert result["framework"] == "langchain"
    assert result["profile"]["display_name"] == "LangChain"
    assert set(result["profile"].keys()) == EXPECTED_PROFILE_FIELDS


def test_alias_with_case_and_extra_spaces():
    result = get_framework_profile.invoke({"framework_name": "  LANG CHAIN  "})

    assert result["ok"] is True
    assert result["framework"] == "langchain"


def test_alias_semantic_kernel():
    result = get_framework_profile.invoke({"framework_name": "Semantic Kernel"})

    assert result["ok"] is True
    assert result["framework"] == "semantic_kernel"


def test_alias_crew_ai():
    result = get_framework_profile.invoke({"framework_name": "Crew AI"})

    assert result["ok"] is True
    assert result["framework"] == "crewai"


def test_unknown_framework_returns_structured_error():
    result = get_framework_profile.invoke({"framework_name": "Django"})

    assert result["ok"] is False
    assert result["error"]
    assert set(result["supported_frameworks"]) == EXPECTED_DISPLAY_NAMES


def test_empty_name_returns_structured_error():
    result = get_framework_profile.invoke({"framework_name": ""})

    assert result["ok"] is False
    assert result["error"]


@pytest.mark.parametrize(
    ("input_name", "expected_key"),
    [
        ("Llama-Index", "llamaindex"),
        ("llama_index", "llamaindex"),
        ("Hay-Stack", "haystack"),
        ("hay_stack", "haystack"),
        ("Semantic-Kernel", "semantic_kernel"),
        ("semantic_kernel", "semantic_kernel"),
        ("Crew-AI", "crewai"),
        ("crew_ai", "crewai"),
    ],
)
def test_alias_with_underscore_or_hyphen(input_name, expected_key):
    result = get_framework_profile.invoke({"framework_name": input_name})

    assert result["ok"] is True
    assert result["framework"] == expected_key


def test_returned_profile_does_not_leak_global_state():
    first_result = get_framework_profile.invoke({"framework_name": "langchain"})
    first_result["profile"]["limitations"][0] = "ИЗМЕНЕНО"

    second_result = get_framework_profile.invoke({"framework_name": "langchain"})

    assert second_result["profile"]["limitations"][0] != "ИЗМЕНЕНО"
    assert FRAMEWORK_PROFILES["langchain"]["limitations"][0] != "ИЗМЕНЕНО"
