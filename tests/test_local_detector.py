import json

import httpx
import pytest

from doc_quant.config import load_config
from doc_quant.detector import DETECTION_SYSTEM_PROMPT, ENTITY_SCHEMA
from doc_quant.local_detector import (
    LOCAL_BATCH_PREFIX,
    STATUS_ERROR,
    STATUS_OK,
    build_local_payload_template,
    build_local_request,
    detect_local,
    get_local_client,
    new_local_batch_id,
    probe_local_server,
)
from doc_quant.local_llm import LocalLLMError


def _answer(entities):
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": json.dumps({"entities": entities})}}]},
    )


def _client(handler):
    return get_local_client(load_config(), transport=httpx.MockTransport(handler))


def test_template_carries_shared_prompt_schema_and_endpoint():
    template = build_local_payload_template(load_config())
    assert template["system"] == DETECTION_SYSTEM_PROMPT
    assert template["response_format"]["json_schema"]["schema"] == ENTITY_SCHEMA
    assert template["temperature"] == 0.0
    assert "base_url" in template  # reported, stripped from the wire payload


def test_request_strips_reporting_keys_and_builds_messages():
    template = build_local_payload_template(load_config())
    payload = build_local_request(template, "Some fragment")
    assert "base_url" not in payload
    assert "system" not in payload
    assert payload["messages"] == [
        {"role": "system", "content": DETECTION_SYSTEM_PROMPT},
        {"role": "user", "content": "Some fragment"},
    ]


def test_detect_local_parses_entities():
    def handler(request):
        return _answer([{"text": "Jan Novak", "type": "person"}])

    template = build_local_payload_template(load_config())
    outcome = detect_local(_client(handler), template, "Jan Novak signed.")
    assert outcome.status == STATUS_OK
    assert outcome.entities == [("Jan Novak", "person")]
    assert outcome.dropped == 0


def test_verbatim_guard_drops_hallucinated_entities():
    def handler(request):
        return _answer(
            [
                {"text": "Jan Novak", "type": "person"},
                {"text": "Elvira Ghost", "type": "person"},
            ]
        )

    template = build_local_payload_template(load_config())
    outcome = detect_local(_client(handler), template, "Jan Novak signed.")
    assert outcome.status == STATUS_OK
    assert outcome.entities == [("Jan Novak", "person")]
    assert outcome.dropped == 1


def test_detect_local_retries_invalid_json_then_errors():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "not json"}}]}
        )

    template = build_local_payload_template(load_config())
    outcome = detect_local(_client(handler), template, "text")
    assert outcome.status == STATUS_ERROR
    assert calls["n"] == 2  # LOCAL_DETECTION_ATTEMPTS


def test_detect_local_retry_recovers():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "not json"}}]}
            )
        return _answer([])

    template = build_local_payload_template(load_config())
    outcome = detect_local(_client(handler), template, "text")
    assert outcome.status == STATUS_OK
    assert calls["n"] == 2


def test_transport_error_is_an_error_outcome_not_an_exception():
    def handler(request):
        raise httpx.ConnectError("refused")

    template = build_local_payload_template(load_config())
    outcome = detect_local(_client(handler), template, "text")
    assert outcome.status == STATUS_ERROR
    assert "unreachable" in outcome.detail


def test_probe_raises_actionable_error_when_server_down():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(LocalLLMError, match="unreachable"):
        probe_local_server(load_config(), transport=httpx.MockTransport(handler))


def test_probe_passes_when_models_endpoint_answers():
    def handler(request):
        assert request.url.path.endswith("/models")
        return httpx.Response(200, json={"data": []})

    probe_local_server(load_config(), transport=httpx.MockTransport(handler))


def test_batch_id_shape():
    batch_id = new_local_batch_id()
    assert batch_id.startswith(LOCAL_BATCH_PREFIX)
    assert len(batch_id) == len(LOCAL_BATCH_PREFIX) + 12
