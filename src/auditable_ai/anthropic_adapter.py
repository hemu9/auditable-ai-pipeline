"""Claude extraction adapter: one Messages API call per document, with every exchange recorded.

The official `anthropic` SDK is optional (`pip install 'auditable-ai-pipeline[claude]'`)
and is imported only when a live transport is created. Request construction,
response interpretation and output parsing work on plain JSON, so a recorded
response can be replayed, and these paths tested, without the SDK or a
network connection. The adapter's output is still untrusted:
EvidenceExtractor validates it exactly as it validates any other adapter."""

from __future__ import annotations

import hashlib
import math
import random
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from auditable_ai._immutable import strict_loads
from auditable_ai.dataset import DatasetDocument
from auditable_ai.documents import RawDocumentContent
from auditable_ai.figure_task import SYSTEM_PROMPT, render_document, response_schema

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "medium"
DEFAULT_MAX_TOKENS = 16000
FALLBACK_BETA = "server-side-fallback-2026-07-01"
_SECRETS = re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}|(?i:bearer)\s+[A-Za-z0-9._\-]{8,}")


def redact(text: str) -> str:
    """Remove anything shaped like an Anthropic key or bearer token before it is recorded."""
    return _SECRETS.sub("[REDACTED]", text)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ProviderCallError(Exception):
    """The provider returned an error, or could not be reached, so no response exists."""

    def __init__(
        self,
        message: str,
        *,
        error_type: str,
        status_code: int | None = None,
        request_id: str | None = None,
        retryable: bool = False,
        retry_after_s: float | None = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.status_code = status_code
        self.request_id = request_id
        self.retryable = retryable
        self.retry_after_s = retry_after_s

    def describe(self) -> dict[str, Any]:
        return {
            "outcome": "error",
            "error_type": self.error_type,
            "message": redact(str(self)),
            "status_code": self.status_code,
            "request_id": self.request_id,
            "retryable": self.retryable,
        }


class ModelOutputError(Exception):
    """The provider responded, but the response cannot be used as extraction output."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True)
class RawResponse:
    status_code: int
    request_id: str | None
    text: str


@dataclass(frozen=True)
class ClaudeSettings:
    """Settings are checked when created, so every run record can be read back."""

    model: str = DEFAULT_MODEL
    effort: str = DEFAULT_EFFORT
    max_tokens: int = DEFAULT_MAX_TOKENS
    fallbacks: str | None = "default"
    max_attempts: int = 3
    timeout_s: float = 120.0

    def __post_init__(self) -> None:
        for name in ("model", "effort"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("max_tokens", "max_attempts"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.fallbacks is not None and not isinstance(self.fallbacks, str):
            raise ValueError("fallbacks must be a string or None")
        if (
            isinstance(self.timeout_s, bool)
            or not isinstance(self.timeout_s, (int, float))
            or not math.isfinite(self.timeout_s)
            or self.timeout_s <= 0
        ):
            raise ValueError("timeout_s must be a positive number of seconds")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_request(document: DatasetDocument, settings: ClaudeSettings) -> dict[str, Any]:
    """Request parameters for one document. Adaptive thinking is the model default; no
    sampling parameters are sent, because current models and API versions reject them."""
    request: dict[str, Any] = {
        "model": settings.model,
        "max_tokens": settings.max_tokens,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": render_document(document)}],
        "output_config": {
            "effort": settings.effort,
            "format": {"type": "json_schema", "schema": response_schema()},
        },
    }
    if settings.fallbacks:
        request["betas"] = [FALLBACK_BETA]
        request["fallbacks"] = settings.fallbacks
    return request


def _seconds(value: str | None) -> float | None:
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


class AnthropicTransport:
    """Sends one request through the official SDK and returns the exact response body."""

    def __init__(
        self, client: Any = None, *, timeout: float = 120.0, origin: str = "recorded_model_response"
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "the Claude adapter needs the optional SDK: "
                "pip install 'auditable-ai-pipeline[claude]'"
            ) from exc
        self._sdk = anthropic
        self._client = (
            client if client is not None else anthropic.Anthropic(max_retries=0, timeout=timeout)
        )
        self.sdk_version = anthropic.__version__
        self.origin = origin

    def send(self, request: dict[str, Any]) -> RawResponse:
        sdk = self._sdk
        messages = self._client.beta.messages if "betas" in request else self._client.messages
        try:
            raw = messages.with_raw_response.create(**request)
        except sdk.APIStatusError as exc:
            raise ProviderCallError(
                exc.message,
                error_type=type(exc).__name__,
                status_code=exc.status_code,
                request_id=exc.request_id,
                retryable=exc.status_code in (408, 409, 429) or exc.status_code >= 500,
                retry_after_s=_seconds(exc.response.headers.get("retry-after")),
            ) from None
        except sdk.APIConnectionError as exc:
            raise ProviderCallError(
                str(exc), error_type=type(exc).__name__, retryable=True
            ) from None
        return RawResponse(raw.status_code, raw.request_id, raw.http_response.text)


@dataclass(frozen=True)
class Interpretation:
    message_id: str | None
    served_model: str | None
    stop_reason: str | None
    stop_details: Any
    usage: dict[str, Any] | None
    fallbacks: tuple[dict[str, Any], ...]
    output_text: str | None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["fallbacks"] = list(self.fallbacks)
        return value


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _model_of(side: Any) -> str | None:
    return _text(side.get("model")) if isinstance(side, dict) else None


def interpret_response(raw_response: str) -> Interpretation:
    """Read a Messages API response body without the SDK, so recordings replay offline.

    Any body yields an Interpretation or a ModelOutputError: a field of an
    unexpected type reads as absent, and the raw body stays in the record."""
    try:
        body = strict_loads(raw_response)
    except (ValueError, RecursionError) as exc:
        raise ModelOutputError("unexpected_response", f"response body is not JSON: {exc}") from exc
    if (
        not isinstance(body, dict)
        or body.get("type") != "message"
        or not isinstance(body.get("content"), list)
    ):
        raise ModelOutputError("unexpected_response", "response body is not a Messages API message")
    blocks = [block for block in body["content"] if isinstance(block, dict)]
    texts = [
        block["text"]
        for block in blocks
        if block.get("type") == "text" and isinstance(block.get("text"), str)
    ]
    fallbacks = tuple(
        {"from": _model_of(block.get("from")), "to": _model_of(block.get("to"))}
        for block in blocks
        if block.get("type") == "fallback"
    )
    usage = body.get("usage")
    return Interpretation(
        message_id=_text(body.get("id")),
        served_model=_text(body.get("model")),
        stop_reason=_text(body.get("stop_reason")),
        stop_details=body.get("stop_details"),
        usage=usage if isinstance(usage, dict) else None,
        fallbacks=fallbacks,
        output_text="".join(texts) if texts else None,
    )


def parse_output(interpretation: Interpretation) -> Any:
    """Turn a usable response into the JSON value handed to the validator, or explain why not."""
    if interpretation.stop_reason == "refusal":
        raise ModelOutputError(
            "refused", f"the model declined (stop_details={interpretation.stop_details!r})"
        )
    if interpretation.stop_reason == "max_tokens":
        raise ModelOutputError("truncated", "the response reached max_tokens; output is incomplete")
    if interpretation.output_text is None:
        raise ModelOutputError(
            "no_text",
            f"the response has no text block (stop_reason={interpretation.stop_reason!r})",
        )
    try:
        return strict_loads(interpretation.output_text)
    except (ValueError, RecursionError) as exc:
        raise ModelOutputError("invalid_json", f"the output text is not JSON: {exc}") from exc


@dataclass
class ModelExchange:
    """Everything sent to and received from the provider during one adapter call.

    Request headers, and therefore credentials, are never part of the record."""

    provider: str
    request: dict[str, Any]
    started_at: str
    attempts: list[dict[str, Any]] = field(default_factory=list)
    completed_at: str | None = None
    status_code: int | None = None
    request_id: str | None = None
    latency_ms: float | None = None
    raw_response: str | None = None
    response_sha256: str | None = None
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ClaudeExtractionAdapter:
    """EvidenceExtractor callable that asks Claude for one document's figures.

    Retryable provider errors are retried with jittered backoff; every attempt is
    recorded, and latency is measured for the final attempt only."""

    def __init__(
        self,
        document: DatasetDocument,
        settings: ClaudeSettings = ClaudeSettings(),
        *,
        transport: Any = None,
        clock: Callable[[], datetime] = _utc_now,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        self.document = document
        self.settings = settings
        self.request = build_request(document, settings)
        self.transport = transport
        self.exchange: ModelExchange | None = None
        self.interpretation: Interpretation | None = None
        self._clock = clock
        self._sleep = sleep
        self._jitter = jitter

    def __call__(self, content: RawDocumentContent) -> Any:
        if content.text != self.document.text:
            raise ValueError("this adapter was built for a different document")
        if self.transport is None:
            self.transport = AnthropicTransport(timeout=self.settings.timeout_s)
        exchange = ModelExchange("anthropic", self.request, self._clock().isoformat())
        self.exchange = exchange
        for attempt in range(1, self.settings.max_attempts + 1):
            began = time.perf_counter()
            try:
                response = self.transport.send(self.request)
            except ProviderCallError as exc:
                elapsed = (time.perf_counter() - began) * 1000
                exchange.attempts.append(
                    {"attempt": attempt, "latency_ms": elapsed, **exc.describe()}
                )
                if not exc.retryable or attempt == self.settings.max_attempts:
                    exchange.error = exc.describe()
                    exchange.latency_ms = elapsed
                    exchange.completed_at = self._clock().isoformat()
                    raise
                backoff = 2 ** (attempt - 1) + self._jitter()
                self._sleep(min(max(backoff, exc.retry_after_s or 0.0), 60.0))
                continue
            elapsed = (time.perf_counter() - began) * 1000
            exchange.attempts.append(
                {
                    "attempt": attempt,
                    "latency_ms": elapsed,
                    "outcome": "response",
                    "status_code": response.status_code,
                    "request_id": response.request_id,
                }
            )
            exchange.status_code = response.status_code
            exchange.request_id = response.request_id
            exchange.latency_ms = elapsed
            exchange.raw_response = response.text
            exchange.response_sha256 = sha256_text(response.text)
            exchange.completed_at = self._clock().isoformat()
            break
        self.interpretation = interpret_response(exchange.raw_response or "")
        return parse_output(self.interpretation)


class RecordedResponseAdapter:
    """Replays a recorded response body through the same interpretation and parsing.

    It makes no network call; `model_calls` stays zero by construction."""

    model_calls = 0

    def __init__(self, document_text: str, raw_response: str) -> None:
        self._document_text = document_text
        self._raw_response = raw_response
        self.interpretation: Interpretation | None = None

    def __call__(self, content: RawDocumentContent) -> Any:
        if content.text != self._document_text:
            raise ValueError("this recording belongs to a different document")
        self.interpretation = interpret_response(self._raw_response)
        return parse_output(self.interpretation)
