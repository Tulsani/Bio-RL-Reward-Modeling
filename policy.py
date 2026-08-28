"""Common probability-policy interfaces and initial policy implementations."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from llm_client import LLMCallResult, LLMClient
from prompts import (
    ACTION_NAMES,
    OBSERVATION_COLUMNS,
    ZERO_SHOT_PROMPT_VERSION,
    build_zero_shot_messages,
)

HANDOFF_NOTE_KEY = "handoff_note"


class Policy(ABC):
    """Interface shared by baselines and LLM policies."""

    @abstractmethod
    def predict_proba(
        self, observations: Sequence[Mapping[str, Any]]
    ) -> np.ndarray:
        """Return an ``[N, 3]`` array in canonical action order."""

    def reset_diagnostics(self) -> None:
        """Clear any per-call evaluation diagnostics."""

    def diagnostics(self) -> dict[str, Any]:
        """Return evaluation diagnostics collected by the policy."""
        return {
            "num_calls": 0,
            "cache_hits": 0,
            "cache_hit_rate": 0.0,
            "structured_output_failures": 0,
            "structured_output_failure_rate": 0.0,
            "fallbacks": 0,
            "fallback_rate": 0.0,
            "failure_reasons": {},
        }


class AlwaysMaintainPolicy(Policy):
    """Supplied baseline assigning all probability to ``maintain``."""

    model_id = "supplied-always-maintain"
    prompt_version = "not-applicable"

    def predict_proba(
        self, observations: Sequence[Mapping[str, Any]]
    ) -> np.ndarray:
        output = np.zeros((len(observations), len(ACTION_NAMES)), dtype=float)
        output[:, 0] = 1.0
        return output


class ZeroShotLLMPolicy(Policy):
    """Zero-shot prompted policy backed by validated, cached LLM responses."""

    def __init__(self, client: LLMClient) -> None:
        if client.prompt_version != ZERO_SHOT_PROMPT_VERSION:
            raise ValueError(
                "ZeroShotLLMPolicy requires client prompt_version "
                f"{ZERO_SHOT_PROMPT_VERSION!r}, got "
                f"{client.prompt_version!r}."
            )
        self.client = client
        self.model_id = client.model_id
        self.prompt_version = client.prompt_version
        self.call_results: list[LLMCallResult] = []

    def predict_proba(
        self, observations: Sequence[Mapping[str, Any]]
    ) -> np.ndarray:
        """Prompt once per state and return normalized action probabilities."""
        probabilities = np.empty(
            (len(observations), len(ACTION_NAMES)), dtype=float
        )

        for index, observation in enumerate(observations):
            structured_state, handoff_note = _split_policy_observation(
                observation
            )
            messages = build_zero_shot_messages(
                structured_state, handoff_note=handoff_note
            )
            result = self.client.get_decision(messages)
            self.call_results.append(result)
            probabilities[index] = result.decision.probabilities

        _validate_probability_matrix(probabilities)
        return probabilities

    def reset_diagnostics(self) -> None:
        self.call_results.clear()

    def diagnostics(self) -> dict[str, Any]:
        count = len(self.call_results)
        cache_hits = sum(result.cache_hit for result in self.call_results)
        structured_failures = sum(
            result.structured_output_failure for result in self.call_results
        )
        fallbacks = sum(result.fallback_used for result in self.call_results)
        reasons = Counter(
            result.failure_reason
            for result in self.call_results
            if result.failure_reason is not None
        )

        return {
            "num_calls": count,
            "cache_hits": cache_hits,
            "cache_hit_rate": _rate(cache_hits, count),
            "structured_output_failures": structured_failures,
            "structured_output_failure_rate": _rate(
                structured_failures, count
            ),
            "fallbacks": fallbacks,
            "fallback_rate": _rate(fallbacks, count),
            "failure_reasons": dict(sorted(reasons.items())),
        }


def _split_policy_observation(
    observation: Mapping[str, Any],
) -> tuple[dict[str, Any], str | None]:
    if not isinstance(observation, Mapping):
        raise TypeError("Each observation must be a mapping.")

    allowed = set(OBSERVATION_COLUMNS)
    supplied = set(observation)
    unexpected = sorted(supplied - allowed - {HANDOFF_NOTE_KEY})
    if unexpected:
        raise ValueError(
            "Policy observation contains unexpected fields: "
            f"{unexpected}"
        )

    structured_state = {
        column: observation[column]
        for column in OBSERVATION_COLUMNS
        if column in observation
    }
    handoff_note = observation.get(HANDOFF_NOTE_KEY)
    if handoff_note is not None and not isinstance(handoff_note, str):
        raise TypeError("handoff_note must be a string or None.")
    return structured_state, handoff_note


def _validate_probability_matrix(probabilities: np.ndarray) -> None:
    if probabilities.ndim != 2 or probabilities.shape[1] != len(ACTION_NAMES):
        raise ValueError(
            f"Policy probabilities must have shape [N, {len(ACTION_NAMES)}]."
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("Policy probabilities must be finite.")
    if (probabilities < 0.0).any():
        raise ValueError("Policy probabilities must be non-negative.")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-9):
        raise ValueError("Each policy probability row must sum to one.")


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    rate = numerator / denominator
    if not math.isfinite(rate):
        raise ValueError("Diagnostic rate must be finite.")
    return rate
