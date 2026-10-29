"""An injected adapter produces untrusted JSON; deterministic code validates it.

The adapter is trusted executable application code. This module does not
sandbox it, prevent hallucinations, or prove semantic support for a claim.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable

from auditable_ai._immutable import canonical_hash
from auditable_ai.claims import Claim, ClaimCategory, ClaimRepository, EvidenceReference
from auditable_ai.documents import RawDocumentContent, SourceDocument
from auditable_ai.errors import (
    DuplicateRunError,
    ExtractionValidationError,
    ProvenanceError,
    TemporalIntegrityError,
)
from auditable_ai.temporal import require_not_in_future, require_timezone_aware

VALIDATOR_VERSION = "2"

TOP_FIELDS = frozenset(
    {"facts", "inferences", "unknowns", "contradicting_evidence", "source_ids", "confidence"}
)
ITEM_FIELDS = frozenset(
    {
        "claim_text",
        "claim_type",
        "supporting_evidence",
        "counterevidence",
        "confidence",
        "falsifier",
        "contradicting_claim_ids",
    }
)
REF_FIELDS = frozenset({"source_id", "source_chunk_id", "source_content_hash", "locator_note"})


class RejectionReason(Enum):
    INVALID_ITEM = "INVALID_ITEM"
    UNKNOWN_FIELD = "UNKNOWN_FIELD"
    PROVENANCE_VIOLATION = "PROVENANCE_VIOLATION"


@dataclass(frozen=True)
class RejectedItem:
    bucket: str
    index: int
    reason: RejectionReason
    detail: str


class _UnknownFieldError(ExtractionValidationError):
    """Classifies an item rejection as UNKNOWN_FIELD rather than INVALID_ITEM."""


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"cannot serialize {type(value).__name__}")


@dataclass(frozen=True)
class ExtractionResult:
    """Immutable audit result, including the exact JSON value the adapter returned.

    raw_response_json is untrusted content retained for inspection. Merely
    retaining it never makes it an accepted claim. A JSON response hash
    identifies canonical content, not the provider's original transport bytes.
    """

    claims: tuple[Claim, ...]
    rejected_items: tuple[RejectedItem, ...]
    unknowns: tuple[str, ...]
    contradicting_evidence: tuple[EvidenceReference, ...]
    declared_source_ids: tuple[str, ...]
    confidence: float | None
    run_id: str
    subject_id: str
    source_version_id: str
    input_hash: str
    response_hash: str
    raw_response_json: str
    model_name: str
    model_version: str
    prompt_version: str
    source_as_of: datetime
    recorded_at: datetime
    latency_ms: float
    validator_version: str = VALIDATOR_VERSION

    def to_dict(self) -> dict[str, Any]:
        value = json.loads(json.dumps(asdict(self), default=_json_default, allow_nan=False))
        value["raw_response"] = json.loads(value.pop("raw_response_json"))
        value["accuracy_status"] = "NOT_VERIFIED"
        value["source_query_mode"] = "RESEARCH_AS_OF"
        return value


def _confidence(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ExtractionValidationError("confidence must be null or a finite number in [0, 1]")
    return float(value)


def _strings(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ExtractionValidationError(f"{field} must be a list of non-empty strings")
    return tuple(value)


def _reference(raw: Any, declared: frozenset[str]) -> EvidenceReference:
    if not isinstance(raw, dict):
        raise ExtractionValidationError("evidence reference must be an object")
    if set(raw) - REF_FIELDS:
        raise _UnknownFieldError("evidence reference contains unknown fields")
    source_id = raw.get("source_id")
    if not isinstance(source_id, str) or not source_id or source_id not in declared:
        raise ExtractionValidationError("evidence source must appear in declared source_ids")
    for field in REF_FIELDS - {"source_id"}:
        if raw.get(field) is not None and not isinstance(raw[field], str):
            raise ExtractionValidationError(f"{field} must be a string or null")
    return EvidenceReference(**raw)


def _references(raw: Any, declared: frozenset[str]) -> tuple[EvidenceReference, ...]:
    if not isinstance(raw, list):
        raise ExtractionValidationError("evidence references must be a list")
    return tuple(_reference(item, declared) for item in raw)


class EvidenceExtractor:
    """Validate one-document extraction under RESEARCH_AS_OF(source_as_of).

    The injected adapter receives raw text only. All evidence that reaches the
    claim repository is checked again against real immutable document versions.
    Valid items can survive an invalid sibling, and accepted items are committed
    together. A reused run id is refused before the adapter runs; batch-shape
    errors fail before any claims are stored. A FACT without an explicit
    reference is attributed to the input document; this attribution is exposed
    and is not verification.
    """

    def __init__(
        self, *, claims: ClaimRepository, extraction_fn: Callable[[RawDocumentContent], Any]
    ) -> None:
        self.claims = claims
        self.extraction_fn = extraction_fn

    def extract(
        self,
        document: SourceDocument,
        *,
        source_as_of: datetime,
        extracted_at: datetime,
        subject_id: str,
        run_id: str,
        model_name: str,
        model_version: str,
        prompt_version: str,
    ) -> ExtractionResult:
        require_timezone_aware("source_as_of", source_as_of)
        require_timezone_aware("extracted_at", extracted_at)
        for field, value in (
            ("subject_id", subject_id),
            ("run_id", run_id),
            ("model_name", model_name),
            ("model_version", model_version),
            ("prompt_version", prompt_version),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ExtractionValidationError(f"{field} must be a non-empty string")
        if self.claims.has_run(run_id):
            raise DuplicateRunError(
                f"run_id {run_id!r} already has a committed result; refusing to run the adapter under a reused identity"
            )
        registered = self.claims.documents.get_version(document.document_version_id)
        if registered != document:
            raise ProvenanceError(
                "input must match an already registered immutable document version"
            )
        if document.publicly_available_at > source_as_of:
            raise ExtractionValidationError("input document was not public at source_as_of")
        if extracted_at < document.recorded_at or extracted_at < document.publicly_available_at:
            raise TemporalIntegrityError(
                "extraction cannot precede the input's recording or availability"
            )
        require_not_in_future("extracted_at", extracted_at, self.claims.clock)

        start = time.perf_counter()
        raw = self.extraction_fn(document.raw_content)
        latency_ms = (time.perf_counter() - start) * 1000
        try:
            raw_json = json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False)
            parsed = json.loads(raw_json)
            response_hash = canonical_hash(parsed)
        except (TypeError, ValueError, RecursionError) as exc:
            raise ExtractionValidationError("adapter output must be finite JSON data") from exc
        if not isinstance(parsed, dict) or set(parsed) - TOP_FIELDS:
            raise ExtractionValidationError(
                "response must be an object containing only allowed top-level fields"
            )
        for field in ("facts", "inferences"):
            if not isinstance(parsed.get(field, []), list):
                raise ExtractionValidationError(f"{field} must be a list")
        unknowns = _strings(parsed.get("unknowns", []), "unknowns")
        source_ids = _strings(parsed.get("source_ids", []), "source_ids")
        declared = frozenset(source_ids)
        confidence = _confidence(parsed.get("confidence"))
        contradicting = _references(parsed.get("contradicting_evidence", []), declared)
        for reference in contradicting:
            self.claims.validate_reference(
                reference,
                publicly_available_at=document.publicly_available_at,
                recorded_at=extracted_at,
            )

        accepted: list[Claim] = []
        rejected: list[RejectedItem] = []
        sequence = 0
        for bucket, category in (
            ("facts", ClaimCategory.FACT),
            ("inferences", ClaimCategory.INFERENCE),
        ):
            for index, item in enumerate(parsed.get(bucket, [])):
                sequence += 1
                try:
                    if not isinstance(item, dict):
                        raise ExtractionValidationError("claim item must be an object")
                    if set(item) - ITEM_FIELDS:
                        raise _UnknownFieldError("claim item contains unknown fields")
                    text = item.get("claim_text")
                    if not isinstance(text, str) or not text.strip():
                        raise ExtractionValidationError("claim_text must be non-empty")
                    claim_type = item.get("claim_type", "UNSPECIFIED")
                    if not isinstance(claim_type, str) or not claim_type.strip():
                        raise ExtractionValidationError("claim_type must be a non-empty string")
                    supporting = _references(item.get("supporting_evidence", []), declared)
                    if category is ClaimCategory.FACT and not supporting:
                        supporting = (EvidenceReference(source_id=document.document_version_id),)
                    counterevidence = _references(item.get("counterevidence", []), declared)
                    item_confidence = _confidence(item.get("confidence"))
                    falsifier = item.get("falsifier")
                    if falsifier is not None and not isinstance(falsifier, str):
                        raise ExtractionValidationError("falsifier must be a string or null")
                    contradicting_ids = _strings(
                        item.get("contradicting_claim_ids", []), "contradicting_claim_ids"
                    )
                    claim = Claim(
                        claim_id=f"{run_id}-claim-{sequence}",
                        run_id=run_id,
                        subject_id=subject_id,
                        category=category,
                        claim_type=claim_type.strip(),
                        claim_text=text.strip(),
                        supporting_evidence=supporting,
                        counterevidence=counterevidence,
                        publicly_available_at=document.publicly_available_at,
                        ingested_at=extracted_at,
                        recorded_at=extracted_at,
                        model_name=model_name,
                        model_version=model_version,
                        prompt_version=prompt_version,
                        input_hash=document.content_hash,
                        confidence=item_confidence,
                        falsifier=falsifier,
                        contradicting_claim_ids=contradicting_ids,
                        missing_information=unknowns,
                    )
                    self.claims.validate_evidence(claim)
                except (ProvenanceError, TemporalIntegrityError) as exc:
                    rejected.append(
                        RejectedItem(bucket, index, RejectionReason.PROVENANCE_VIOLATION, str(exc))
                    )
                except _UnknownFieldError as exc:
                    rejected.append(
                        RejectedItem(bucket, index, RejectionReason.UNKNOWN_FIELD, str(exc))
                    )
                except ExtractionValidationError as exc:
                    rejected.append(
                        RejectedItem(bucket, index, RejectionReason.INVALID_ITEM, str(exc))
                    )
                else:
                    accepted.append(claim)

        self.claims.add_run(run_id, accepted)
        return ExtractionResult(
            claims=tuple(accepted),
            rejected_items=tuple(rejected),
            unknowns=unknowns,
            contradicting_evidence=contradicting,
            declared_source_ids=source_ids,
            confidence=confidence,
            run_id=run_id,
            subject_id=subject_id,
            source_version_id=document.document_version_id,
            input_hash=document.content_hash,
            response_hash=response_hash,
            raw_response_json=raw_json,
            model_name=model_name,
            model_version=model_version,
            prompt_version=prompt_version,
            source_as_of=source_as_of,
            recorded_at=extracted_at,
            latency_ms=latency_ms,
        )
