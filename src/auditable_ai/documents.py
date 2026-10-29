"""Immutable document versions and evidence anchors.

A source_id identifies a logical document series. A document_version_id names
one permanent version. References always resolve exact versions, never a
moving latest-document pointer. Raw text is inert data in this module; this
wrapper is not a security sandbox for caller-provided adapter code."""

from __future__ import annotations

import datetime as dt
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from auditable_ai._immutable import deep_freeze
from auditable_ai.errors import ProvenanceError
from auditable_ai.temporal import (
    AppendOnlyStore,
    AsOfMode,
    require_dependent_not_before_dependency,
    validate_temporal_fields,
)


@dataclass(frozen=True)
class RawDocumentContent:
    """Untrusted text and its content fingerprint; this wrapper does not sandbox an adapter."""

    text: str

    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    def __repr__(self) -> str:
        preview = self.text[:40].replace("\n", " ")
        return f"RawDocumentContent(len={len(self.text)}, preview={preview!r})"


@dataclass(frozen=True)
class SourceDocument:
    """One immutable version of a logical source. Metadata is copied and deep-frozen."""

    document_version_id: str
    source_id: str
    source_type: str
    document_location: str
    raw_content: RawDocumentContent
    content_hash: str
    publicly_available_at: dt.datetime
    ingested_at: dt.datetime
    recorded_at: dt.datetime
    source_published_at: Optional[dt.datetime] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    superseded_publicly_at: Optional[dt.datetime] = None
    superseded_in_system_at: Optional[dt.datetime] = None

    def __post_init__(self) -> None:
        validate_temporal_fields(
            ingested_at=self.ingested_at,
            recorded_at=self.recorded_at,
            publicly_available_at=self.publicly_available_at,
            superseded_publicly_at=self.superseded_publicly_at,
            superseded_in_system_at=self.superseded_in_system_at,
        )
        object.__setattr__(self, "metadata", deep_freeze(self.metadata))
        if self.content_hash != self.raw_content.content_hash():
            raise ValueError(
                f"content_hash {self.content_hash!r} does not match the actual hash of raw_content ({self.raw_content.content_hash()!r}); provenance would be unverifiable"
            )


@dataclass(frozen=True)
class SourceChunk:
    """An exact version-bound evidence anchor with independently checked content and times."""

    chunk_id: str
    document_version_id: str
    locator: str
    raw_text: RawDocumentContent
    content_hash: str
    publicly_available_at: dt.datetime
    ingested_at: dt.datetime
    recorded_at: dt.datetime
    superseded_publicly_at: Optional[dt.datetime] = None
    superseded_in_system_at: Optional[dt.datetime] = None

    def __post_init__(self) -> None:
        validate_temporal_fields(
            ingested_at=self.ingested_at,
            recorded_at=self.recorded_at,
            publicly_available_at=self.publicly_available_at,
            superseded_publicly_at=self.superseded_publicly_at,
            superseded_in_system_at=self.superseded_in_system_at,
        )
        if self.content_hash != self.raw_text.content_hash():
            raise ValueError(
                f"content_hash {self.content_hash!r} does not match the actual hash of raw_text ({self.raw_text.content_hash()!r})"
            )


class SourceDocumentRepository(ABC):
    """Storage contract for exact-version lookup and both historical query modes."""

    @abstractmethod
    def add(self, document: SourceDocument) -> None: ...

    @abstractmethod
    def get_version(self, document_version_id: str) -> Optional[SourceDocument]:
        pass

    @abstractmethod
    def research_as_of(self, as_of: dt.datetime) -> tuple[SourceDocument, ...]:
        pass

    @abstractmethod
    def operational_replay_as_of(self, as_of: dt.datetime) -> tuple[SourceDocument, ...]: ...

    @abstractmethod
    def research_as_of_lineage(
        self, source_id: str, as_of: dt.datetime
    ) -> Optional[SourceDocument]:
        pass

    @abstractmethod
    def operational_replay_as_of_lineage(
        self, source_id: str, as_of: dt.datetime
    ) -> Optional[SourceDocument]:
        pass


class InMemorySourceDocumentRepository(SourceDocumentRepository):
    """Append new version identities without rewriting existing documents."""

    def __init__(self) -> None:
        self._store = AppendOnlyStore(key_fn=lambda d: d.source_id)
        self._by_version: dict[str, SourceDocument] = {}

    def add(self, document: SourceDocument) -> None:
        if document.document_version_id in self._by_version:
            raise ValueError(
                f"document_version_id {document.document_version_id!r} already exists; version identities are permanent and never reused"
            )
        self._store.add(document)
        self._by_version[document.document_version_id] = document

    def get_version(self, document_version_id: str) -> Optional[SourceDocument]:
        return self._by_version.get(document_version_id)

    def research_as_of(self, as_of: dt.datetime) -> tuple[SourceDocument, ...]:
        return tuple(self._store.as_of(as_of, AsOfMode.RESEARCH))

    def operational_replay_as_of(self, as_of: dt.datetime) -> tuple[SourceDocument, ...]:
        return tuple(self._store.as_of(as_of, AsOfMode.OPERATIONAL_REPLAY))

    def _lineage(
        self, source_id: str, as_of: dt.datetime, mode: AsOfMode
    ) -> Optional[SourceDocument]:
        current = self._store.as_of_matching(as_of, mode, lambda d: d.source_id == source_id)
        return current[0] if current else None

    def research_as_of_lineage(
        self, source_id: str, as_of: dt.datetime
    ) -> Optional[SourceDocument]:
        return self._lineage(source_id, as_of, AsOfMode.RESEARCH)

    def operational_replay_as_of_lineage(
        self, source_id: str, as_of: dt.datetime
    ) -> Optional[SourceDocument]:
        return self._lineage(source_id, as_of, AsOfMode.OPERATIONAL_REPLAY)


class SourceChunkRepository(ABC):
    @abstractmethod
    def add(self, chunk: SourceChunk) -> None: ...

    @abstractmethod
    def for_document_version(self, document_version_id: str) -> tuple[SourceChunk, ...]: ...

    @abstractmethod
    def get(self, chunk_id: str) -> Optional[SourceChunk]: ...


class InMemorySourceChunkRepository(SourceChunkRepository):
    def __init__(self, document_repository: SourceDocumentRepository) -> None:
        self._documents = document_repository
        self._chunks: dict[str, SourceChunk] = {}

    def add(self, chunk: SourceChunk) -> None:
        if chunk.chunk_id in self._chunks:
            raise ValueError(f"chunk_id {chunk.chunk_id!r} already exists")
        document = self._documents.get_version(chunk.document_version_id)
        if document is None:
            raise ProvenanceError(
                f"SourceChunk {chunk.chunk_id!r} references document_version_id {chunk.document_version_id!r}, which does not exist"
            )
        require_dependent_not_before_dependency(
            dependent_publicly_available_at=chunk.publicly_available_at,
            dependent_recorded_at=chunk.recorded_at,
            dependency_publicly_available_at=document.publicly_available_at,
            dependency_recorded_at=document.recorded_at,
            dependent_label=f"SourceChunk {chunk.chunk_id!r}",
            dependency_label=f"its document version {document.document_version_id!r}",
        )
        self._chunks[chunk.chunk_id] = chunk

    def for_document_version(self, document_version_id: str) -> tuple[SourceChunk, ...]:
        return tuple(
            (c for c in self._chunks.values() if c.document_version_id == document_version_id)
        )

    def get(self, chunk_id: str) -> Optional[SourceChunk]:
        return self._chunks.get(chunk_id)
