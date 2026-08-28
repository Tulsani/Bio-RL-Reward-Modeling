import json

import pytest

from llm_client import (
    FALLBACK_PROBABILITIES,
    LLMClient,
    PolicyDecision,
)

MESSAGES = [
    {"role": "system", "content": "Return JSON."},
    {"role": "user", "content": "Choose an action."},
]


def valid_response(**overrides):
    response = {
        "prob_maintain": 0.5,
        "prob_iv_fluids": 0.3,
        "prob_escalate_vasopressor": 0.2,
        "rationale": "The current pre-action state supports this distribution.",
    }
    response.update(overrides)
    return json.dumps(response)


class StaticBackend:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def complete(self, messages, *, model_id, temperature):
        self.calls += 1
        return self.response


class FailingBackend:
    def complete(self, messages, *, model_id, temperature):
        raise RuntimeError("provider unavailable")


def make_client(tmp_path, *, backend=None, **overrides):
    settings = {
        "model_id": "mock-model-v1",
        "prompt_version": "zero_shot_v1",
        "cache_dir": tmp_path,
        "backend": backend,
    }
    settings.update(overrides)
    return LLMClient(**settings)


def test_schema_normalizes_valid_probabilities():
    decision = PolicyDecision.model_validate_json(
        valid_response(
            prob_maintain=0.4,
            prob_iv_fluids=0.3,
            prob_escalate_vasopressor=0.2,
        )
    )

    assert sum(decision.probabilities) == pytest.approx(1.0)
    assert decision.probabilities == pytest.approx(
        (0.4 / 0.9, 0.3 / 0.9, 0.2 / 0.9)
    )


@pytest.mark.parametrize(
    "response",
    [
        "not json",
        valid_response(prob_maintain=-0.1),
        valid_response(
            prob_maintain=0.0,
            prob_iv_fluids=0.0,
            prob_escalate_vasopressor=0.0,
        ),
        json.dumps(
            {
                "prob_maintain": 0.5,
                "prob_iv_fluids": 0.3,
                "prob_escalate_vasopressor": 0.2,
                "rationale": "Valid rationale.",
                "unexpected": "field",
            }
        ),
    ],
)
def test_malformed_output_uses_deterministic_fallback(tmp_path, response):
    client = make_client(tmp_path, backend=StaticBackend(response))

    result = client.get_decision(MESSAGES)

    assert result.decision.probabilities == FALLBACK_PROBABILITIES
    assert result.fallback_used is True
    assert result.structured_output_failure is True
    assert result.failure_reason == "structured_output_error"


def test_successful_response_is_cached_and_reused(tmp_path):
    backend = StaticBackend(valid_response())
    client = make_client(tmp_path, backend=backend)

    first = client.get_decision(MESSAGES)
    second = client.get_decision(MESSAGES)

    assert backend.calls == 1
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert first.cache_key == second.cache_key
    assert first.decision == second.decision
    assert first.fallback_used is False


def test_cached_mode_requires_no_backend(tmp_path):
    online_client = make_client(
        tmp_path, backend=StaticBackend(valid_response())
    )
    expected = online_client.get_decision(MESSAGES)
    cached_client = make_client(tmp_path, backend=None)

    actual = cached_client.get_decision(MESSAGES)

    assert actual.cache_hit is True
    assert actual.fallback_used is False
    assert actual.decision == expected.decision


def test_cache_miss_without_backend_uses_fallback(tmp_path):
    result = make_client(tmp_path).get_decision(MESSAGES)

    assert result.decision.probabilities == FALLBACK_PROBABILITIES
    assert result.fallback_used is True
    assert result.failure_reason == "cache_miss"
    assert result.structured_output_failure is False


def test_backend_error_uses_fallback(tmp_path):
    result = make_client(
        tmp_path, backend=FailingBackend()
    ).get_decision(MESSAGES)

    assert result.decision.probabilities == FALLBACK_PROBABILITIES
    assert result.fallback_used is True
    assert result.failure_reason == "backend_error"


def test_cache_key_changes_with_reproducibility_inputs(tmp_path):
    base = make_client(tmp_path)
    other_model = make_client(tmp_path, model_id="other-model")
    other_prompt = make_client(tmp_path, prompt_version="zero_shot_v2")
    changed_messages = [*MESSAGES[:-1], {"role": "user", "content": "New"}]

    keys = {
        base.make_cache_key(MESSAGES),
        other_model.make_cache_key(MESSAGES),
        other_prompt.make_cache_key(MESSAGES),
        base.make_cache_key(changed_messages),
    }

    assert len(keys) == 4


def test_invalid_message_shape_is_rejected_before_model_call(tmp_path):
    backend = StaticBackend(valid_response())
    client = make_client(tmp_path, backend=backend)

    with pytest.raises(ValueError, match="role and content"):
        client.get_decision(
            [{"role": "user", "content": "hello", "extra": "bad"}]
        )

    assert backend.calls == 0
