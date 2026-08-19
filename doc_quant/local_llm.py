"""Minimal client for any OpenAI-compatible local inference server.

Deliberately vendor-free: Ollama, LM Studio and llama.cpp all speak
`/chat/completions`, so one tiny client covers every local backend we care
about without pulling an SDK in. It lives on its own - rather than inside
`synthetic` - because more than one caller now needs a local model: synthetic
prose is one, local detection is another, and they need different request
shapes over the same transport.

`chat_completion` is therefore the raw seam: it posts the payload it is handed
verbatim, so a caller wanting a system prompt, a JSON schema or different
sampling never has to fork the HTTP mechanics, error handling included.
"""

from __future__ import annotations

import httpx


class LocalLLMError(Exception):
    """Raised when the local OpenAI-compatible endpoint cannot be used."""


class LocalLLMClient:
    """Minimal client for an OpenAI-compatible `/chat/completions` endpoint.

    Deliberately tiny: only the one call this module needs, so that any server
    speaking the protocol (Ollama, LM Studio, llama.cpp) works without a
    vendor SDK.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float,
        timeout_seconds: float,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Configure the endpoint.

        Args:
            base_url: OpenAI-compatible root, e.g. http://localhost:11434/v1.
            model: model name as the local server knows it.
            temperature: sampling temperature for the prose.
            timeout_seconds: per-request timeout.
            transport: optional httpx transport override; tests use it to stay
                offline, production leaves it at None.
        """
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._temperature = temperature
        self._timeout_seconds = timeout_seconds
        self._transport = transport

    def chat_completion(self, payload: dict) -> str:
        """POST `payload` to /chat/completions and return the assistant content.

        The payload is sent exactly as given - nothing is injected or
        rewritten - so a caller can shape the request however the local server
        allows while still inheriting the connection, status and payload
        checks below.

        Raises:
            LocalLLMError: when the server is unreachable, times out, answers
                with a non-200 status, or returns an unusable payload.
        """
        url = f"{self._base_url}/chat/completions"
        try:
            with httpx.Client(
                transport=self._transport, timeout=self._timeout_seconds
            ) as client:
                response = client.post(url, json=payload)
        except httpx.ConnectError as exc:
            raise self._unusable(f"connection refused ({exc})") from exc
        except httpx.TimeoutException as exc:
            raise self._unusable(
                f"no answer within {self._timeout_seconds}s ({exc})"
            ) from exc

        if response.status_code != 200:
            raise self._unusable(f"HTTP {response.status_code}")

        try:
            body = response.json()
            content = body["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise self._unusable(f"unexpected response payload ({exc})") from exc

        if not isinstance(content, str):
            raise self._unusable(
                f"expected string content, got {type(content).__name__}"
            )
        return content

    def generate(self, prompt: str, seed: int) -> str:
        """Return the assistant message for `prompt`, sampled with `seed`.

        Raises:
            LocalLLMError: when the server is unreachable, times out, answers
                with a non-200 status, or returns an unusable payload.
        """
        return self.chat_completion(
            {
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self._temperature,
                "seed": seed,
                "stream": False,
            }
        )

    def _unusable(self, detail: str) -> LocalLLMError:
        """Build an error that says what to do about it, not just what broke."""
        return LocalLLMError(
            f"Local LLM server unreachable at {self._base_url}: {detail}. "
            f"Start one, e.g. Ollama (`ollama serve`, `ollama pull {self._model}`) "
            "or LM Studio, or change synthetic.llm.base_url in config/config.json."
        )
