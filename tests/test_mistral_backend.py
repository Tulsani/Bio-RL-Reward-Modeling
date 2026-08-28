from types import SimpleNamespace

import pytest

from llm_client import PolicyDecision
from mistral_backend import MistralBackend, MistralConfigurationError

MESSAGES = [
    {"role": "system", "content": "Return JSON."},
    {"role": "user", "content": "Choose an action."},
]


class FakeChat:
    def __init__(self, content, *, choices=True):
        self.content = content
        self.include_choices = choices
        self.kwargs = None

    def parse(self, **kwargs):
        self.kwargs = kwargs
        if not self.include_choices:
            return SimpleNamespace(choices=[])
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeMistralClient:
    def __init__(self, content, *, choices=True):
        self.chat = FakeChat(content, choices=choices)


def test_backend_uses_structured_schema_and_deterministic_settings():
    content = '{"prob_maintain":1,"prob_iv_fluids":0,' \
        '"prob_escalate_vasopressor":0,"rationale":"Stable."}'
    fake_client = FakeMistralClient(content)
    backend = MistralBackend(
        client=fake_client,
        random_seed=42,
        max_tokens=128,
    )

    actual = backend.complete(
        MESSAGES,
        model_id="mistral-small-2603",
        temperature=0.0,
    )

    assert actual == content
    assert fake_client.chat.kwargs["response_format"] is PolicyDecision
    assert fake_client.chat.kwargs["model"] == "mistral-small-2603"
    assert fake_client.chat.kwargs["temperature"] == 0.0
    assert fake_client.chat.kwargs["random_seed"] == 42
    assert fake_client.chat.kwargs["max_tokens"] == 128
    assert fake_client.chat.kwargs["messages"] == MESSAGES


def test_backend_rejects_missing_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

    with pytest.raises(MistralConfigurationError, match="MISTRAL_API_KEY"):
        MistralBackend(dotenv_path=tmp_path / "missing.env")


def test_backend_rejects_empty_completion_choices():
    backend = MistralBackend(client=FakeMistralClient("", choices=False))

    with pytest.raises(RuntimeError, match="no completion choices"):
        backend.complete(
            MESSAGES,
            model_id="mistral-small-2603",
            temperature=0.0,
        )


@pytest.mark.parametrize("content", [None, "", "   "])
def test_backend_rejects_empty_or_non_text_content(content):
    backend = MistralBackend(client=FakeMistralClient(content))

    with pytest.raises(RuntimeError, match="empty or non-text"):
        backend.complete(
            MESSAGES,
            model_id="mistral-small-2603",
            temperature=0.0,
        )


def test_backend_validates_generation_limits():
    with pytest.raises(ValueError, match="max_tokens"):
        MistralBackend(client=FakeMistralClient("{}"), max_tokens=0)

    with pytest.raises(ValueError, match="timeout_ms"):
        MistralBackend(client=FakeMistralClient("{}"), timeout_ms=0)
