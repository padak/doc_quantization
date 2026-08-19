"""Tests for the standalone local OpenAI-compatible client.

`chat_completion` is the raw seam: it must post whatever payload it is handed,
unchanged, so callers that need a different request shape (structured outputs,
a system prompt) do not have to fork the HTTP mechanics. `generate` is the
prose caller that used to own those mechanics and must keep behaving exactly
as before.
"""

import json

import httpx
import pytest

from doc_quant.local_llm import LocalLLMClient, LocalLLMError


def _client(handler, timeout=5.0):
    return LocalLLMClient(
        base_url="http://fake-llm/v1",
        model="test-model",
        temperature=0.0,
        timeout_seconds=timeout,
        transport=httpx.MockTransport(handler),
    )


def test_chat_completion_posts_payload_verbatim_and_returns_content():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        seen["url"] = str(request.url)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "hello"}}]},
        )

    payload = {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}
    assert _client(handler).chat_completion(payload) == "hello"
    assert seen["body"] == payload
    assert seen["url"] == "http://fake-llm/v1/chat/completions"


def test_chat_completion_raises_on_http_error():
    def handler(request):
        return httpx.Response(500)

    with pytest.raises(LocalLLMError, match="HTTP 500"):
        _client(handler).chat_completion({"model": "m", "messages": []})


def test_generate_still_works_through_chat_completion():
    def handler(request):
        body = json.loads(request.content)
        assert body["seed"] == 7
        assert body["stream"] is False
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "prose"}}]}
        )

    assert _client(handler).generate("write", seed=7) == "prose"


def test_synthetic_reexports_stay_importable():
    from doc_quant.synthetic import LocalLLMClient as A, LocalLLMError as B

    assert A is LocalLLMClient
    assert B is LocalLLMError
