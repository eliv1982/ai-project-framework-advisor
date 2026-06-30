from __future__ import annotations

import pytest
from pydantic import ValidationError

from advisor.schema import AdvisorResponse


def _valid_payload(**overrides):
    payload = {
        "recommended_framework": "langchain",
        "reasoning": [
            "Хорошая поддержка агентов через create_agent",
            "Широкая экосистема интеграций",
        ],
        "suitable_components": ["Tools", "Middleware", "Checkpointers"],
        "risks": ["Быстро меняющийся API между версиями"],
    }
    payload.update(overrides)
    return payload


def test_valid_response_is_created_successfully():
    response = AdvisorResponse(**_valid_payload())

    assert response.model_dump() == {
        "recommended_framework": "langchain",
        "reasoning": [
            "Хорошая поддержка агентов через create_agent",
            "Широкая экосистема интеграций",
        ],
        "suitable_components": ["Tools", "Middleware", "Checkpointers"],
        "risks": ["Быстро меняющийся API между версиями"],
    }


@pytest.mark.parametrize(
    "framework_key",
    ["langchain", "llamaindex", "haystack", "semantic_kernel", "crewai"],
)
def test_all_five_framework_keys_are_valid(framework_key):
    response = AdvisorResponse(
        **_valid_payload(recommended_framework=framework_key)
    )

    assert response.recommended_framework == framework_key


def test_unknown_framework_raises_validation_error():
    with pytest.raises(ValidationError):
        AdvisorResponse(**_valid_payload(recommended_framework="django"))


def test_missing_required_field_raises_validation_error():
    payload = _valid_payload()
    del payload["risks"]

    with pytest.raises(ValidationError):
        AdvisorResponse(**payload)


def test_unknown_extra_field_raises_validation_error():
    with pytest.raises(ValidationError):
        AdvisorResponse(**_valid_payload(confidence=0.9))


def test_empty_reasoning_list_raises_validation_error():
    with pytest.raises(ValidationError):
        AdvisorResponse(**_valid_payload(reasoning=[]))


def test_too_long_reasoning_list_raises_validation_error():
    with pytest.raises(ValidationError):
        AdvisorResponse(**_valid_payload(reasoning=[f"Причина {i}" for i in range(6)]))


def test_blank_string_inside_reasoning_raises_validation_error():
    with pytest.raises(ValidationError):
        AdvisorResponse(**_valid_payload(reasoning=["   "]))


def test_string_is_stripped_of_surrounding_whitespace():
    response = AdvisorResponse(
        **_valid_payload(reasoning=["  Хорошая поддержка агентов  "])
    )

    assert response.reasoning == ["Хорошая поддержка агентов"]


def test_empty_suitable_components_list_raises_validation_error():
    with pytest.raises(ValidationError):
        AdvisorResponse(**_valid_payload(suitable_components=[]))


def test_empty_risks_list_raises_validation_error():
    with pytest.raises(ValidationError):
        AdvisorResponse(**_valid_payload(risks=[]))
