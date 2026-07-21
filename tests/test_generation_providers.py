import pytest

import generation_providers


class FakeResponse:
    def __init__(self, payload, status_error=None):
        self.payload = payload
        self.status_error = status_error

    def raise_for_status(self):
        if self.status_error is not None:
            raise self.status_error

    def json(self):
        return self.payload


def test_ollama_provider_sends_non_streaming_chat_request(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse({"message": {"content": " Local answer "}})

    monkeypatch.setattr(generation_providers.requests, "post", fake_post)
    provider = generation_providers.OllamaGenerationProvider(
        model="test-model",
        base_url="http://127.0.0.1:11434/",
        system_prompt="Context: {context}",
        timeout_seconds=12,
    )

    answer = provider.generate(context="Document text", question="Question?")

    assert answer == "Local answer"
    assert captured["url"] == "http://127.0.0.1:11434/api/chat"
    assert captured["timeout"] == 12
    assert captured["json"] == {
        "model": "test-model",
        "stream": False,
        "messages": [
            {"role": "system", "content": "Context: Document text"},
            {"role": "user", "content": "Question?"},
        ],
    }


def test_ollama_provider_maps_invalid_response_to_unavailable(monkeypatch):
    monkeypatch.setattr(
        generation_providers.requests,
        "post",
        lambda *args, **kwargs: FakeResponse({"message": {"content": ""}}),
    )
    provider = generation_providers.OllamaGenerationProvider(
        model="test-model",
        base_url="http://127.0.0.1:11434",
        system_prompt="Context: {context}",
        timeout_seconds=12,
    )

    with pytest.raises(generation_providers.ProviderUnavailableError):
        provider.generate(context="Document text", question="Question?")
