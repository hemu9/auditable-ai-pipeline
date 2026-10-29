"""Model extraction runs: call through the extraction boundary, record, and replay offline.

A run record holds the embedded document, the exact request, every attempt,
the raw response body, the validated result or the failure diagnostics, usage,
latency and a cost estimate. Records are written with exclusive creation and
never overwritten. Reading one checks its shape first. Replay re-validates a
recorded response with no model call, under the recorded run_id, so an
unchanged validator reproduces the stored result exactly.

`origin` says where a record's response came from: "recorded_model_response"
for a real API call, "scripted_fixture" for test doubles. Replay output is
labeled "replayed_recording"."""

from __future__ import annotations

import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from auditable_ai._immutable import strict_loads
from auditable_ai.anthropic_adapter import (
    ClaudeExtractionAdapter,
    ClaudeSettings,
    ModelOutputError,
    ProviderCallError,
    RecordedResponseAdapter,
    sha256_text,
)
from auditable_ai.dataset import DatasetDocument, document_from_entry, document_to_entry, register
from auditable_ai.documents import RawDocumentContent
from auditable_ai.errors import ExtractionValidationError, ProvenanceError, TemporalIntegrityError
from auditable_ai.extraction import VALIDATOR_VERSION, EvidenceExtractor, ExtractionResult
from auditable_ai.figure_task import TASK_VERSION, prompt_sha256, template_sha256
from auditable_ai.pricing import estimate_cost
from auditable_ai.temporal import require_timezone_aware

RECORD_TYPE = "model_extraction_run"
RECORD_VERSION = 1
MODEL_NAME = "anthropic-messages-api"
RUN_STATUSES = (
    "completed",
    "refused",
    "truncated",
    "malformed_output",
    "validation_error",
    "provenance_error",
    "provider_error",
)
_OUTPUT_ERROR_STATUS = {"refused": "refused", "truncated": "truncated"}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def validated_outcome(
    call: Callable[[], ExtractionResult],
) -> tuple[str, ExtractionResult | None, Any]:
    """Run the extraction and classify every way it can end, so failures are never scored."""
    try:
        result = call()
    except ProviderCallError as exc:
        return "provider_error", None, {"stage": "provider_call", **exc.describe()}
    except ModelOutputError as exc:
        status = _OUTPUT_ERROR_STATUS.get(exc.kind, "malformed_output")
        return status, None, {"stage": "model_output", "kind": exc.kind, "message": str(exc)}
    except ExtractionValidationError as exc:
        return "validation_error", None, {"stage": "batch_validation", "message": str(exc)}
    except (ProvenanceError, TemporalIntegrityError) as exc:
        return (
            "provenance_error",
            None,
            {"stage": "batch_provenance", "error_type": type(exc).__name__, "message": str(exc)},
        )
    return "completed", result, None


def run_model_extraction(
    document: DatasetDocument,
    settings: ClaudeSettings = ClaudeSettings(),
    *,
    transport: Any = None,
    clock: Callable[[], datetime] = _utc_now,
    sleep: Callable[[float], None] = time.sleep,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Call the model through EvidenceExtractor and return a complete run record."""
    source, claims = register(document)
    adapter = ClaudeExtractionAdapter(
        document, settings, transport=transport, clock=clock, sleep=sleep
    )
    started = clock()
    require_timezone_aware("clock()", started)
    run_id = (
        run_id or f"{document.document_version_id}--{settings.model}--{started:%Y%m%dT%H%M%S%fZ}"
    )
    status, result, diagnostics = validated_outcome(
        lambda: EvidenceExtractor(claims=claims, extraction_fn=adapter).extract(
            source,
            source_as_of=started,
            extracted_at=started,
            subject_id=document.source_id,
            run_id=run_id,
            model_name=MODEL_NAME,
            model_version=settings.model,
            prompt_version=TASK_VERSION,
        )
    )
    interpretation = adapter.interpretation
    return {
        "record_type": RECORD_TYPE,
        "record_version": RECORD_VERSION,
        "origin": getattr(adapter.transport, "origin", "unknown"),
        "status": status,
        "run_id": run_id,
        "document": {**document_to_entry(document), "content_hash": source.content_hash},
        "task": {
            "version": TASK_VERSION,
            "template_sha256": template_sha256(),
            "prompt_sha256": prompt_sha256(document),
        },
        "settings": {
            "provider": "anthropic",
            **settings.to_dict(),
            "sdk_version": getattr(adapter.transport, "sdk_version", None),
        },
        "timestamps": {"source_as_of": started.isoformat(), "extracted_at": started.isoformat()},
        "exchange": adapter.exchange.to_dict() if adapter.exchange else None,
        "response": interpretation.to_dict() if interpretation else None,
        "served_model_matches_request": (
            interpretation.served_model == settings.model if interpretation else None
        ),
        "cost": (
            estimate_cost(
                interpretation.usage,
                interpretation.served_model,
                stop_reason=interpretation.stop_reason,
            )
            if interpretation
            else {"status": "unavailable", "reason": "no response was received", "usd": None}
        ),
        "validator_version": VALIDATOR_VERSION,
        "result": result.to_dict() if result else None,
        "diagnostics": diagnostics,
    }


def write_record(
    record: dict[str, Any], directory: Path, *, clock: Callable[[], datetime] = _utc_now
) -> Path:
    """Write a record to a new file; an existing record is never replaced.

    The record is serialized before the file is created, so a record that
    cannot be written leaves no partial file behind."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{re.sub(r'[^A-Za-z0-9._-]', '_', record['run_id'])}.json"
    written = {**record, "record_written_at": clock().isoformat()}
    text = json.dumps(written, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)
    return path


def _is_timestamp(value: Any) -> bool:
    """An ISO 8601 string with a UTC offset, the form records store every time in."""
    if not isinstance(value, str):
        return False
    try:
        return datetime.fromisoformat(value).tzinfo is not None
    except ValueError:
        return False


def _is_measurement(value: Any) -> bool:
    return value is None or (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


_KINDS: dict[str, Callable[[Any], bool]] = {
    "a string": lambda value: isinstance(value, str),
    "a non-empty string": lambda value: isinstance(value, str) and bool(value.strip()),
    "a string or null": lambda value: value is None or isinstance(value, str),
    "a timestamp with a UTC offset": _is_timestamp,
    "a list of strings": lambda value: (
        isinstance(value, list) and all(isinstance(item, str) for item in value)
    ),
    "a positive integer": lambda value: (
        isinstance(value, int) and not isinstance(value, bool) and value > 0
    ),
    "true, false or null": lambda value: value is None or isinstance(value, bool),
    "a list": lambda value: isinstance(value, list),
    "an object or null": lambda value: value is None or isinstance(value, dict),
    "a non-negative number or null": _is_measurement,
    f"one of {', '.join(RUN_STATUSES)}": lambda value: value in RUN_STATUSES,
}
# Every field replay, evaluation and the CLI read from a stored record, and its form.
_RECORD_SHAPE: dict[str, Any] = {
    "run_id": "a non-empty string",
    "origin": "a non-empty string",
    "status": f"one of {', '.join(RUN_STATUSES)}",
    "validator_version": "a non-empty string",
    "served_model_matches_request": "true, false or null",
    "result": "an object or null",
    "exchange": "an object or null",
    "document": {
        "document_version_id": "a non-empty string",
        "source_id": "a non-empty string",
        "title": "a string",
        "split": "a string",
        "content_hash": "a string",
        "publicly_available_at": "a timestamp with a UTC offset",
        "recorded_at": "a timestamp with a UTC offset",
        "sentences": "a list of strings",
    },
    "task": {"version": "a non-empty string"},
    "settings": {
        "provider": "a string",
        "model": "a string",
        "effort": "a string",
        "max_tokens": "a positive integer",
        "fallbacks": "a string or null",
    },
    "timestamps": {
        "source_as_of": "a timestamp with a UTC offset",
        "extracted_at": "a timestamp with a UTC offset",
    },
}
_EXCHANGE_SHAPE = {
    "attempts": "a list",
    "raw_response": "a string or null",
    "response_sha256": "a string or null",
    "latency_ms": "a non-negative number or null",
}


def _check_fields(value: Any, shape: Mapping[str, Any], where: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"malformed run record: {where or 'the record'} must be an object")
    for name, expected in shape.items():
        field = f"{where}.{name}" if where else name
        if name not in value:
            raise ValueError(f"malformed run record: {field} is missing")
        if isinstance(expected, dict):
            _check_fields(value[name], expected, field)
        elif not _KINDS[expected](value[name]):
            raise ValueError(f"malformed run record: {field} must be {expected}")


def check_record(record: Any) -> None:
    """Refuse anything not shaped like a record that run_model_extraction writes.

    Replay, evaluation and the CLI read only the fields checked here, so a
    malformed or hand-edited file is refused with a reason instead of crashing
    them. A well-formed record can still fail replay's hash checks."""
    if not isinstance(record, dict) or record.get("record_type") != RECORD_TYPE:
        raise ValueError(f"malformed run record: not a {RECORD_TYPE} record")
    if record.get("record_version") != RECORD_VERSION:
        raise ValueError(
            f"malformed run record: unsupported record_version {record.get('record_version')!r}"
        )
    _check_fields(record, _RECORD_SHAPE, "")
    if record.get("exchange") is not None:
        _check_fields(record["exchange"], _EXCHANGE_SHAPE, "exchange")


def read_record(path: Path) -> dict[str, Any]:
    """Load a record file, refusing anything a record writer could not have written."""
    try:
        record: dict[str, Any] = strict_loads(path.read_text(encoding="utf-8"))
        check_record(record)
    except (ValueError, RecursionError) as exc:
        raise ValueError(f"{path}: {exc}") from exc
    return record


def _comparable(result: Any) -> Any:
    """A result as replay must reproduce it: everything except the adapter's own latency."""
    if not isinstance(result, dict):
        return result
    return {key: value for key, value in result.items() if key != "latency_ms"}


def replay_record(
    record: Mapping[str, Any], *, clock: Callable[[], datetime] = _utc_now
) -> dict[str, Any]:
    """Re-validate a recorded response under the current validator, without a model call.

    The replay runs under the recorded run_id, so with an unchanged validator it
    must reproduce the stored result exactly, apart from latency."""
    check_record(record)
    document = document_from_entry(record["document"])
    if RawDocumentContent(document.text).content_hash() != record["document"]["content_hash"]:
        raise ProvenanceError("the embedded document does not match its recorded content hash")
    exchange = record.get("exchange") or {}
    raw_response = exchange.get("raw_response")
    replay = {
        "record_type": "replayed_model_run",
        "origin": "replayed_recording",
        "replayed_from": {
            "run_id": record["run_id"],
            "origin": record["origin"],
            "status": record["status"],
            "response_sha256": exchange.get("response_sha256"),
        },
        "replayed_at": clock().isoformat(),
        "validator_version": VALIDATOR_VERSION,
        "model_calls": 0,
    }
    if raw_response is None:
        return {
            **replay,
            "status": "not_replayable",
            "reason": "the recorded run received no provider response",
            "recorded_diagnostics": record.get("diagnostics"),
        }
    if sha256_text(raw_response) != exchange.get("response_sha256"):
        raise ProvenanceError("the recorded response does not match its recorded hash")
    source, claims = register(document)
    adapter = RecordedResponseAdapter(document.text, raw_response)
    timestamps = record["timestamps"]
    status, result, diagnostics = validated_outcome(
        lambda: EvidenceExtractor(claims=claims, extraction_fn=adapter).extract(
            source,
            source_as_of=datetime.fromisoformat(timestamps["source_as_of"]),
            extracted_at=datetime.fromisoformat(timestamps["extracted_at"]),
            subject_id=document.source_id,
            run_id=record["run_id"],
            model_name=MODEL_NAME,
            model_version=record["settings"]["model"],
            prompt_version=record["task"]["version"],
        )
    )
    replayed = result.to_dict() if result else None
    recorded, reproduced = _comparable(record["result"]), _comparable(replayed)
    if isinstance(recorded, dict) and isinstance(reproduced, dict):
        keys = recorded.keys() | reproduced.keys()
        differences = sorted(key for key in keys if recorded.get(key) != reproduced.get(key))
    else:
        differences = [] if recorded == reproduced else ["result"]
    if record["status"] != status:
        differences.insert(0, "status")
    return {
        **replay,
        "status": status,
        "response": adapter.interpretation.to_dict() if adapter.interpretation else None,
        "result": replayed,
        "diagnostics": diagnostics,
        "matches_recorded_outcome": not differences,
        "differences": differences,
    }
