"""Scripted, provider-shaped test doubles. None of these is a real model response."""

import json
from datetime import datetime, timezone

from auditable_ai.anthropic_adapter import ProviderCallError, RawResponse

SCRIPTED_USAGE = {
    "input_tokens": 900,
    "output_tokens": 300,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
    "service_tier": "standard",
}


def fixed_clock():
    return datetime(2026, 10, 28, 12, 0, tzinfo=timezone.utc)


def message_body(
    output=None, *, text=None, stop_reason="end_turn", model="claude-opus-5-5", blocks=()
):
    """A Messages API response body; `output` becomes the JSON text block."""
    content = [{"type": "thinking", "thinking": "", "signature": "scripted-fixture"}, *blocks]
    if output is not None or text is not None:
        content.append({"type": "text", "text": json.dumps(output) if text is None else text})
    return json.dumps(
        {
            "id": "msg_scripted_fixture",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "stop_details": None,
            "usage": dict(SCRIPTED_USAGE),
        }
    )


def response(body, *, request_id="req_scripted_fixture"):
    return RawResponse(200, request_id, body)


def transient_error():
    return ProviderCallError(
        "overloaded",
        error_type="OverloadedError",
        status_code=529,
        request_id="req_scripted_529",
        retryable=True,
    )


class ScriptedTransport:
    """Returns scripted outcomes in order and keeps every request it was sent."""

    origin = "scripted_fixture"
    sdk_version = None

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.requests = []

    def send(self, request):
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def call_center_output():
    """A scripted extraction for eval-call-center-v1 with one fabricated chunk citation."""
    source = "eval-call-center-v1"

    def cite(sentence):
        return {
            "source_id": source,
            "source_chunk_id": f"{source}:s{sentence}",
            "locator_note": None,
        }

    return {
        "facts": [
            {
                "claim_text": "Agents answered 9,320 calls.",
                "claim_type": "current_figure",
                "supporting_evidence": [cite(2)],
                "counterevidence": [],
            },
            {
                "claim_text": "The telephony system reports a 6 percent abandonment rate.",
                "claim_type": "disputed_figure",
                "supporting_evidence": [cite(3)],
                "counterevidence": [cite(3)],
            },
            {
                "claim_text": "The average wait time was 4.5 minutes.",
                "claim_type": "current_figure",
                "supporting_evidence": [cite(7)],
                "counterevidence": [],
            },
        ],
        "inferences": [],
        "unknowns": ["Customer satisfaction scores for the week"],
        "contradicting_evidence": [],
        "source_ids": [source],
    }
