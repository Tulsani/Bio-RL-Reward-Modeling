"""Provider-agnostic LLM response validation, caching, and fallback logic."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CACHE_FORMAT_VERSION = 1
RESPONSE_SCHEMA_VERSION = 1
DEFAULT_TEMPERATURE = 0.0
FALLBACK_PROBABILITIES = (1.0, 0.0, 0.0)
FALLBACK_RATIONALE = "Deterministic fallback to the supplied maintain baseline."

FailureReason = Literal[
    "cache_miss",
    "cache_error",
    "backend_error",
    "structured_output_error",
]


class CompletionBackend(Protocol):
    """Minimal interface to implement for a hosted or local model provider."""

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model_id: str,
        temperature: float,
    ) -> str:
        """Return the model's raw text response."""


class PolicyDecision(BaseModel):
    """Validated and normalized probability response from an LLM policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prob_maintain: float = Field(ge=0.0, le=1.0)
    prob_iv_fluids: float = Field(ge=0.0, le=1.0)
    prob_escalate_vasopressor: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1, max_length=500)

    @field_validator("rationale")
    @classmethod
    def validate_rationale(cls, rationale: str) -> str:
        rationale = rationale.strip()
        if not rationale:
            raise ValueError("rationale cannot be blank")
        if len(rationale.split()) > 40:
            raise ValueError("rationale must contain at most 40 words")
        return rationale

    @model_validator(mode="after")
    def validate_and_normalize_probabilities(self) -> PolicyDecision:
        probabilities = self.probabilities
        if not all(math.isfinite(value) for value in probabilities):
            raise ValueError("probabilities must be finite")

        total = math.fsum(probabilities)
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError("probabilities must have a positive finite sum")

        normalized = tuple(value / total for value in probabilities)
        object.__setattr__(self, "prob_maintain", normalized[0])
        object.__setattr__(self, "prob_iv_fluids", normalized[1])
        object.__setattr__(
            self, "prob_escalate_vasopressor", normalized[2]
        )
        return self

    @property
    def probabilities(self) -> tuple[float, float, float]:
        """Return probabilities in canonical action order."""
        return (
            self.prob_maintain,
            self.prob_iv_fluids,
            self.prob_escalate_vasopressor,
        )


@dataclass(frozen=True)
class LLMCallResult:
    """Decision plus diagnostics required by the evaluation harness."""

    decision: PolicyDecision
    cache_key: str
    cache_hit: bool
    fallback_used: bool
    failure_reason: FailureReason | None
    cache_error: bool
    raw_response: str | None

    @property
    def structured_output_failure(self) -> bool:
        return self.failure_reason == "structured_output_error"


class LLMClient:
    """Get validated decisions from a response cache or completion backend."""

    def __init__(
        self,
        *,
        model_id: str,
        prompt_version: str,
        cache_dir: str | Path = "cache/llm_outputs",
        backend: CompletionBackend | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        if not model_id.strip():
            raise ValueError("model_id cannot be blank.")
        if not prompt_version.strip():
            raise ValueError("prompt_version cannot be blank.")
        if not math.isfinite(temperature) or temperature < 0.0:
            raise ValueError("temperature must be finite and non-negative.")

        self.model_id = model_id
        self.prompt_version = prompt_version
        self.cache_dir = Path(cache_dir)
        self.backend = backend
        self.temperature = temperature

    def get_decision(
        self, messages: Sequence[Mapping[str, str]]
    ) -> LLMCallResult:
        """Return a cached/model decision, falling back deterministically."""
        canonical_messages = _canonical_messages(messages)
        cache_key = self.make_cache_key(canonical_messages)
        cache_path = self.cache_dir / f"{cache_key}.json"
        cache_error = False

        if cache_path.exists():
            try:
                raw_response = self._read_cached_response(
                    cache_path, expected_key=cache_key
                )
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                cache_error = True
            else:
                return self._result_from_response(
                    raw_response,
                    cache_key=cache_key,
                    cache_hit=True,
                    cache_error=False,
                )

        if self.backend is None:
            reason: FailureReason = (
                "cache_error" if cache_error else "cache_miss"
            )
            return self._fallback_result(
                cache_key=cache_key,
                failure_reason=reason,
                cache_error=cache_error,
            )

        try:
            raw_response = self.backend.complete(
                canonical_messages,
                model_id=self.model_id,
                temperature=self.temperature,
            )
            if not isinstance(raw_response, str):
                raise TypeError("Completion backend must return a string.")
        # Provider adapters can raise SDK-specific exception classes. No provider
        # failure should escape the policy boundary or make fallback nondeterministic.
        except Exception:  # noqa: BLE001
            return self._fallback_result(
                cache_key=cache_key,
                failure_reason="backend_error",
                cache_error=cache_error,
            )

        try:
            self._write_cached_response(
                cache_path,
                cache_key=cache_key,
                raw_response=raw_response,
            )
        except OSError:
            cache_error = True

        return self._result_from_response(
            raw_response,
            cache_key=cache_key,
            cache_hit=False,
            cache_error=cache_error,
        )

    def make_cache_key(
        self, messages: Sequence[Mapping[str, str]]
    ) -> str:
        """Hash every input that can affect deterministic model output."""
        payload = {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "response_schema_version": RESPONSE_SCHEMA_VERSION,
            "model_id": self.model_id,
            "prompt_version": self.prompt_version,
            "temperature": self.temperature,
            "messages": _canonical_messages(messages),
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _result_from_response(
        self,
        raw_response: str,
        *,
        cache_key: str,
        cache_hit: bool,
        cache_error: bool,
    ) -> LLMCallResult:
        try:
            decision = PolicyDecision.model_validate_json(raw_response)
        except (ValueError, TypeError):
            return self._fallback_result(
                cache_key=cache_key,
                failure_reason="structured_output_error",
                cache_error=cache_error,
                cache_hit=cache_hit,
                raw_response=raw_response,
            )

        return LLMCallResult(
            decision=decision,
            cache_key=cache_key,
            cache_hit=cache_hit,
            fallback_used=False,
            failure_reason=None,
            cache_error=cache_error,
            raw_response=raw_response,
        )

    def _fallback_result(
        self,
        *,
        cache_key: str,
        failure_reason: FailureReason,
        cache_error: bool,
        cache_hit: bool = False,
        raw_response: str | None = None,
    ) -> LLMCallResult:
        return LLMCallResult(
            decision=fallback_decision(),
            cache_key=cache_key,
            cache_hit=cache_hit,
            fallback_used=True,
            failure_reason=failure_reason,
            cache_error=cache_error,
            raw_response=raw_response,
        )

    def _read_cached_response(
        self, path: Path, *, expected_key: str
    ) -> str:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["cache_format_version"] != CACHE_FORMAT_VERSION:
            raise ValueError("Unsupported cache format version.")
        if record["response_schema_version"] != RESPONSE_SCHEMA_VERSION:
            raise ValueError("Unsupported response schema version.")
        if record["cache_key"] != expected_key:
            raise ValueError("Cache key does not match cache contents.")
        if record["model_id"] != self.model_id:
            raise ValueError("Cached model ID does not match client model ID.")
        if record["prompt_version"] != self.prompt_version:
            raise ValueError("Cached prompt version does not match client.")
        raw_response = record["raw_response"]
        if not isinstance(raw_response, str):
            raise TypeError("Cached response must be a string.")
        return raw_response

    def _write_cached_response(
        self, path: Path, *, cache_key: str, raw_response: str
    ) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "response_schema_version": RESPONSE_SCHEMA_VERSION,
            "cache_key": cache_key,
            "model_id": self.model_id,
            "prompt_version": self.prompt_version,
            "temperature": self.temperature,
            "raw_response": raw_response,
        }
        serialized = json.dumps(
            record, ensure_ascii=False, sort_keys=True, indent=2
        )
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.cache_dir,
                prefix=f".{cache_key}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_file.write(serialized)
                temporary_path = Path(temporary_file.name)
            temporary_path.replace(path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()


def fallback_decision() -> PolicyDecision:
    """Return the supplied always-maintain baseline as a deterministic fallback."""
    return PolicyDecision(
        prob_maintain=FALLBACK_PROBABILITIES[0],
        prob_iv_fluids=FALLBACK_PROBABILITIES[1],
        prob_escalate_vasopressor=FALLBACK_PROBABILITIES[2],
        rationale=FALLBACK_RATIONALE,
    )


def _canonical_messages(
    messages: Sequence[Mapping[str, str]],
) -> tuple[dict[str, str], ...]:
    if not messages:
        raise ValueError("messages cannot be empty.")

    canonical = []
    for message in messages:
        if set(message) != {"role", "content"}:
            raise ValueError(
                "Each message must contain exactly role and content."
            )
        role = message["role"]
        content = message["content"]
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported message role: {role!r}")
        if not isinstance(content, str) or not content:
            raise ValueError("Message content must be a non-empty string.")
        canonical.append({"role": role, "content": content})
    return tuple(canonical)
