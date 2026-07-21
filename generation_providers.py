from typing import Any, Protocol

import requests
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


class GenerationProvider(Protocol):
    name: str
    model: str

    def generate(self, *, context: str, question: str) -> str: ...


class GeminiGenerationProvider:
    name = "gemini"

    def __init__(
        self,
        *,
        model: str,
        prompt: ChatPromptTemplate,
        client: Any,
    ) -> None:
        self.model = model
        self._chain = prompt | client | StrOutputParser()

    def generate(self, *, context: str, question: str) -> str:
        return self._chain.invoke({"context": context, "question": question})


class OllamaGenerationProvider:
    name = "ollama"

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        system_prompt: str,
        timeout_seconds: int,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.system_prompt = system_prompt
        self.timeout_seconds = timeout_seconds

    def generate(self, *, context: str, question: str) -> str:
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "stream": False,
                    "messages": [
                        {
                            "role": "system",
                            "content": self.system_prompt.format(context=context),
                        },
                        {"role": "user", "content": question},
                    ],
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            content = payload.get("message", {}).get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Ollama returned an empty response.")
            return content.strip()
        except (requests.RequestException, ValueError, TypeError) as exc:
            raise ProviderUnavailableError("Ollama is unavailable.") from exc


class ProviderUnavailableError(RuntimeError):
    status_code = 503
