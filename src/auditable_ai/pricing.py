"""Published list prices for estimating the cost of a recorded model call.

An estimate multiplies the token usage the API reported by the list prices
below. It is not an invoice. When the usage shows a pricing modifier this
table does not model (fast mode, non-default inference geography, a
non-standard service tier, a cache write of unknown duration) or the call was
refused, the estimate is reported as unavailable rather than guessed."""

from __future__ import annotations

from typing import Any, Mapping

PRICE_SOURCE = "https://platform.claude.com/docs/en/about-claude/pricing"
PRICES_RETRIEVED_ON = "2026-10-28"

# USD per million tokens, Claude API standard rates on PRICES_RETRIEVED_ON.
PRICES_PER_MTOK: dict[str, dict[str, float]] = {
    "claude-opus-5-5": {
        "input": 4.00,
        "output": 20.00,
        "cache_write_5m": 5.00,
        "cache_write_1h": 8.00,
        "cache_read": 0.20,
    },
    "claude-sonnet-5-5": {
        "input": 2.00,
        "output": 10.00,
        "cache_write_5m": 2.50,
        "cache_write_1h": 4.00,
        "cache_read": 0.10,
    },
    "claude-opus-5": {
        "input": 5.00,
        "output": 25.00,
        "cache_write_5m": 6.25,
        "cache_write_1h": 10.00,
        "cache_read": 0.50,
    },
    "claude-opus-4-8": {
        "input": 5.00,
        "output": 25.00,
        "cache_write_5m": 6.25,
        "cache_write_1h": 10.00,
        "cache_read": 0.50,
    },
}


def _unavailable(reason: str) -> dict[str, Any]:
    return {"status": "unavailable", "reason": reason, "usd": None}


def estimate_cost(
    usage: Mapping[str, Any] | None, served_model: str | None, *, stop_reason: str | None
) -> dict[str, Any]:
    if not usage:
        return _unavailable("the response reported no usage")
    if stop_reason == "refusal":
        return _unavailable("refusal billing depends on the refusal category")
    if usage.get("speed") not in (None, "standard"):
        return _unavailable(f"speed {usage.get('speed')!r} is not modeled")
    if usage.get("service_tier") not in (None, "standard"):
        return _unavailable(f"service tier {usage.get('service_tier')!r} is not modeled")
    if usage.get("inference_geo") not in (None, "global", "not_available"):
        return _unavailable(f"inference geography {usage.get('inference_geo')!r} is not modeled")
    attempts = usage.get("iterations") or [dict(usage, model=served_model)]
    total = 0.0
    lines = []
    for attempt in attempts:
        model = attempt.get("model") or served_model
        prices = PRICES_PER_MTOK.get(model or "")
        if prices is None:
            return _unavailable(f"no recorded list price for model {model!r}")
        written = attempt.get("cache_creation_input_tokens") or 0
        durations = attempt.get("cache_creation") or {}
        five_minute = durations.get("ephemeral_5m_input_tokens", 0 if not written else None)
        one_hour = durations.get("ephemeral_1h_input_tokens", 0 if not written else None)
        if five_minute is None or one_hour is None:
            return _unavailable("cache writes were reported without their duration")
        usd = (
            (attempt.get("input_tokens") or 0) * prices["input"]
            + (attempt.get("output_tokens") or 0) * prices["output"]
            + (attempt.get("cache_read_input_tokens") or 0) * prices["cache_read"]
            + five_minute * prices["cache_write_5m"]
            + one_hour * prices["cache_write_1h"]
        ) / 1_000_000
        total += usd
        lines.append({"model": model, "type": attempt.get("type", "message"), "usd": round(usd, 6)})
    return {
        "status": "estimated",
        "usd": round(total, 6),
        "attempts": lines,
        "basis": {
            "source": PRICE_SOURCE,
            "retrieved_on": PRICES_RETRIEVED_ON,
            "unit": "USD per million tokens",
            "prices": {line["model"]: PRICES_PER_MTOK[line["model"]] for line in lines},
        },
    }
