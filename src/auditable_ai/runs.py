"""Model extraction runs: call through the extraction boundary, record, and replay offline.

A run record holds the embedded document, the exact request, every attempt,
the raw response body, the validated result or the failure diagnostics, usage,
latency and a cost estimate. Records are written with exclusive creation and
never overwritten. Replay re-validates a recorded response with no model call.

`origin` says where a record's response came from: "recorded_model_response"
for a real API call, "scripted_fixture" for test doubles. Replay output is
labeled "replayed_recording"."""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

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

RECORD_TYPE = "model_extraction_run"
RECORD_VERSION = 1
MODEL_NAME = "anthropic-messages-api"
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
    """Write a record to a new file; an existing record is never replaced."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{re.sub(r'[^A-Za-z0-9._-]', '_', record['run_id'])}.json"
    written = {**record, "record_written_at": clock().isoformat()}
    with path.open("x", encoding="utf-8") as stream:
        json.dump(written, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return path


def read_record(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("record_type") != RECORD_TYPE:
        raise ValueError(f"{path} is not a {RECORD_TYPE} record")
    if record.get("record_version") != RECORD_VERSION:
        raise ValueError(f"{path} has unsupported record_version {record.get('record_version')!r}")
    return record


def _outcome(status: str, result: dict[str, Any] | None) -> dict[str, Any]:
    if result is None:
        return {"status": status}
    return {
        "status": status,
        "claims": [
            [
                claim["category"],
                claim["claim_type"],
                claim["claim_text"],
                [reference["source_chunk_id"] for reference in claim["supporting_evidence"]],
            ]
            for claim in result["claims"]
        ],
        "rejected": [
            [item["bucket"], item["index"], item["reason"]] for item in result["rejected_items"]
        ],
        "unknowns": result["unknowns"],
    }


def replay_record(
    record: dict[str, Any], *, clock: Callable[[], datetime] = _utc_now
) -> dict[str, Any]:
    """Re-validate a recorded response under the current validator, without a model call."""
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
            run_id=f"{record['run_id']}--replay",
            model_name=MODEL_NAME,
            model_version=record["settings"]["model"],
            prompt_version=record["task"]["version"],
        )
    )
    replayed = result.to_dict() if result else None
    recorded_outcome = _outcome(record["status"], record.get("result"))
    replayed_outcome = _outcome(status, replayed)
    differences = sorted(
        key
        for key in recorded_outcome.keys() | replayed_outcome.keys()
        if recorded_outcome.get(key) != replayed_outcome.get(key)
    )
    return {
        **replay,
        "status": status,
        "response": adapter.interpretation.to_dict() if adapter.interpretation else None,
        "result": replayed,
        "diagnostics": diagnostics,
        "matches_recorded_outcome": not differences,
        "differences": differences,
    }
