"""Hosted Mistral chat adapter for the provider-agnostic LLM client."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from mistralai.client import Mistral

from llm_client import PolicyDecision


class MistralConfigurationError(ValueError):
    """Raised when the hosted Mistral backend is not configured locally."""


class MistralBackend:
    """Call Mistral chat completions with the policy's Pydantic schema."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        random_seed: int = 42,
        max_tokens: int = 256,
        timeout_ms: int = 60_000,
        dotenv_path: str | Path = ".env",
        client: Any | None = None,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive.")

        self.random_seed = random_seed
        self.max_tokens = max_tokens

        if client is not None:
            self._client = client
            return

        load_dotenv(dotenv_path=dotenv_path, override=False)
        api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not api_key or not api_key.strip():
            raise MistralConfigurationError(
                "MISTRAL_API_KEY is missing. Add it to the ignored .env file "
                "or export it in the shell."
            )
        self._client = Mistral(api_key=api_key, timeout_ms=timeout_ms)

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model_id: str,
        temperature: float,
    ) -> str:
        """Return the raw JSON string generated under the policy schema."""
        response = self._client.chat.parse(
            response_format=PolicyDecision,
            model=model_id,
            messages=[dict(message) for message in messages],
            temperature=temperature,
            random_seed=self.random_seed,
            max_tokens=self.max_tokens,
        )
        if not response.choices:
            raise RuntimeError("Mistral returned no completion choices.")
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Mistral returned empty or non-text content.")
        return content
