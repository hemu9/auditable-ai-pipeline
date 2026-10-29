"""Auditable AI Pipeline: explicit provenance and historical query semantics."""

from auditable_ai.claims import Claim, ClaimCategory, ClaimRepository, EvidenceReference
from auditable_ai.documents import (
    InMemorySourceChunkRepository,
    InMemorySourceDocumentRepository,
    RawDocumentContent,
    SourceChunk,
    SourceDocument,
)
from auditable_ai.errors import (
    AmbiguousRevisionError,
    ExtractionValidationError,
    ProvenanceError,
    TemporalIntegrityError,
)
from auditable_ai.extraction import EvidenceExtractor, ExtractionResult
from auditable_ai.temporal import AsOfMode

__version__ = "0.2.0"
__all__ = [
    "AmbiguousRevisionError",
    "AsOfMode",
    "Claim",
    "ClaimCategory",
    "ClaimRepository",
    "EvidenceExtractor",
    "EvidenceReference",
    "ExtractionResult",
    "ExtractionValidationError",
    "InMemorySourceChunkRepository",
    "InMemorySourceDocumentRepository",
    "ProvenanceError",
    "RawDocumentContent",
    "SourceChunk",
    "SourceDocument",
    "TemporalIntegrityError",
]
