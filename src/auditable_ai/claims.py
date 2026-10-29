"""Source-attributed model claims, with provenance checked at storage time.

FACT means the adapter classified a statement as a source assertion. It is
never a certificate that the statement is accurate or entailed by its source.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Sequence

from auditable_ai.documents import SourceChunkRepository, SourceDocumentRepository
from auditable_ai.errors import DuplicateRunError, ProvenanceError
from auditable_ai.temporal import (
    AppendOnlyStore,
    AsOfMode,
    Clock,
    require_dependent_not_before_dependency,
    require_not_in_future,
    require_timezone_aware,
    utc_now,
    validate_temporal_fields,
)


class ClaimCategory(Enum):
    FACT = "FACT"
    INFERENCE = "INFERENCE"


@dataclass(frozen=True)
class EvidenceReference:
    """source_id names an immutable document version, not a logical series.

    source_content_hash, when provided, hashes the chunk if source_chunk_id
    is set and otherwise hashes the entire document. locator_note is an
    unverified human-readable hint; it is never interpreted as a command.
    """

    source_id: str
    source_chunk_id: str | None = None
    source_content_hash: str | None = None
    locator_note: str | None = None


def _normalized(name: str, value: Any, item_type: type) -> tuple[Any, ...]:
    """Materialize a collection field once, before anything inspects its contents."""
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Iterable):
        raise ValueError(f"{name} must be a collection of {item_type.__name__} values")
    items = tuple(value)
    if not all(isinstance(item, item_type) for item in items):
        raise ValueError(f"{name} must contain only {item_type.__name__} values")
    return items


@dataclass(frozen=True)
class Claim:
    claim_id: str
    run_id: str
    subject_id: str
    category: ClaimCategory
    claim_type: str
    claim_text: str
    supporting_evidence: tuple[EvidenceReference, ...]
    counterevidence: tuple[EvidenceReference, ...]
    publicly_available_at: datetime
    ingested_at: datetime
    recorded_at: datetime
    model_name: str
    model_version: str
    prompt_version: str
    input_hash: str
    confidence: float | None = None
    falsifier: str | None = None
    contradicting_claim_ids: tuple[str, ...] = ()
    missing_information: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, item_type in (
            ("supporting_evidence", EvidenceReference),
            ("counterevidence", EvidenceReference),
            ("contradicting_claim_ids", str),
            ("missing_information", str),
        ):
            object.__setattr__(self, name, _normalized(name, getattr(self, name), item_type))
        validate_temporal_fields(
            ingested_at=self.ingested_at,
            recorded_at=self.recorded_at,
            publicly_available_at=self.publicly_available_at,
        )
        if not isinstance(self.category, ClaimCategory):
            raise ValueError("category must be a ClaimCategory")
        for name in ("claim_id", "run_id", "subject_id", "claim_type", "claim_text"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.category is ClaimCategory.FACT and not self.supporting_evidence:
            raise ValueError("a FACT requires a source reference, not a truth assumption")
        if self.confidence is not None and (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (float, int))
            or not 0 <= self.confidence <= 1
        ):
            raise ValueError("confidence must be null or a finite number in [0, 1]")


class ClaimRepository:
    """Append-only claims with exact source-version checks.

    Two disagreeing claims are separate identities; neither replaces the
    other. Each run is committed once, with all of its claims together.
    Querying public source availability is distinct from asking when this
    system recorded a model result. No claim-revision policy is defined.
    """

    def __init__(
        self,
        documents: SourceDocumentRepository,
        chunks: SourceChunkRepository,
        *,
        clock: Clock = utc_now,
    ) -> None:
        self.documents = documents
        self.chunks = chunks
        self.clock = clock
        self._store: AppendOnlyStore[Claim] = AppendOnlyStore(key_fn=lambda claim: claim.claim_id)
        self._by_id: dict[str, Claim] = {}
        self._run_ids: set[str] = set()

    def validate_reference(
        self,
        reference: EvidenceReference,
        *,
        publicly_available_at: datetime,
        recorded_at: datetime,
    ) -> None:
        document = self.documents.get_version(reference.source_id)
        if document is None:
            raise ProvenanceError(f"unknown document version: {reference.source_id!r}")
        require_dependent_not_before_dependency(
            dependent_publicly_available_at=publicly_available_at,
            dependent_recorded_at=recorded_at,
            dependency_publicly_available_at=document.publicly_available_at,
            dependency_recorded_at=document.recorded_at,
            dependent_label="claim or extraction result",
            dependency_label="cited document",
        )
        expected_hash = document.content_hash
        if reference.source_chunk_id is not None:
            chunk = self.chunks.get(reference.source_chunk_id)
            if chunk is None or chunk.document_version_id != reference.source_id:
                raise ProvenanceError("chunk is missing or belongs to a different document version")
            require_dependent_not_before_dependency(
                dependent_publicly_available_at=publicly_available_at,
                dependent_recorded_at=recorded_at,
                dependency_publicly_available_at=chunk.publicly_available_at,
                dependency_recorded_at=chunk.recorded_at,
                dependent_label="claim or extraction result",
                dependency_label="cited chunk",
            )
            expected_hash = chunk.content_hash
        if (
            reference.source_content_hash is not None
            and reference.source_content_hash != expected_hash
        ):
            raise ProvenanceError("supplied evidence hash does not match the referenced content")

    def validate_evidence(self, claim: Claim) -> None:
        """Check a claim's evidence without storing anything; every commit repeats this."""
        if not all(
            isinstance(group, tuple) and all(isinstance(item, EvidenceReference) for item in group)
            for group in (claim.supporting_evidence, claim.counterevidence)
        ):
            raise ProvenanceError("claim evidence must be a tuple of EvidenceReference values")
        if claim.category is ClaimCategory.FACT and not claim.supporting_evidence:
            raise ProvenanceError("a FACT cannot be stored without a supporting source reference")
        for reference in (*claim.supporting_evidence, *claim.counterevidence):
            self.validate_reference(
                reference,
                publicly_available_at=claim.publicly_available_at,
                recorded_at=claim.recorded_at,
            )

    def has_run(self, run_id: str) -> bool:
        return run_id in self._run_ids

    def add(self, claim: Claim) -> None:
        """Commit a single-claim run."""
        self.add_run(claim.run_id, (claim,))

    def add_run(self, run_id: str, claims: Sequence[Claim]) -> None:
        """Commit one run's claims together or not at all; a run identity is used once."""
        if self.has_run(run_id):
            raise DuplicateRunError(
                f"run_id {run_id!r} already has a committed result; runs are never merged"
            )
        seen: set[str] = set()
        for claim in claims:
            if claim.run_id != run_id:
                raise ProvenanceError(
                    f"claim {claim.claim_id!r} belongs to run {claim.run_id!r}, not {run_id!r}"
                )
            if claim.claim_id in self._by_id or claim.claim_id in seen:
                raise ProvenanceError(
                    f"claim identity {claim.claim_id!r} already exists; claims are never overwritten"
                )
            seen.add(claim.claim_id)
            require_not_in_future(
                f"claim {claim.claim_id!r} recorded_at", claim.recorded_at, self.clock
            )
            self.validate_evidence(claim)
        for claim in claims:
            self._store.add(claim)
            self._by_id[claim.claim_id] = claim
        self._run_ids.add(run_id)

    def get(self, claim_id: str) -> Claim | None:
        return self._by_id.get(claim_id)

    def _for(self, subject_id: str, as_of: datetime, mode: AsOfMode) -> tuple[Claim, ...]:
        require_timezone_aware("as_of", as_of)
        return tuple(
            self._store.as_of_matching(as_of, mode, lambda claim: claim.subject_id == subject_id)
        )

    def research_as_of(self, subject_id: str, as_of: datetime) -> tuple[Claim, ...]:
        """Filter by source availability; does not assert the model ran by T."""
        return self._for(subject_id, as_of, AsOfMode.RESEARCH)

    def operational_replay_as_of(self, subject_id: str, as_of: datetime) -> tuple[Claim, ...]:
        """Only model claims this system had actually recorded by T."""
        return self._for(subject_id, as_of, AsOfMode.OPERATIONAL_REPLAY)
