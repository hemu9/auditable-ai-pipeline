"""Reference JSONL backends sharing the in-memory historical query resolver.

These single-writer examples append one row per addition and never rewrite
earlier rows, but re-read the whole file on every call. They provide no fsync,
locking, encryption, or production-scale performance; a crash can leave a torn
final row, which fails loudly on the next read. Use the repository interface
for a stronger backend without duplicating temporal selection logic."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Optional

from auditable_ai._immutable import to_plain
from auditable_ai.documents import (
    RawDocumentContent,
    SourceChunk,
    SourceChunkRepository,
    SourceDocument,
    SourceDocumentRepository,
    validate_chunk_registration,
)
from auditable_ai.errors import ProvenanceError
from auditable_ai.temporal import (
    AsOfMode,
    Clock,
    require_not_in_future,
    resolve_current_per_key,
    utc_now,
)


def _append(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _serialize(doc: SourceDocument) -> dict[str, Any]:
    return {
        "document_version_id": doc.document_version_id,
        "source_id": doc.source_id,
        "source_type": doc.source_type,
        "document_location": doc.document_location,
        "raw_text": doc.raw_content.text,
        "content_hash": doc.content_hash,
        "publicly_available_at": doc.publicly_available_at.isoformat(),
        "ingested_at": doc.ingested_at.isoformat(),
        "recorded_at": doc.recorded_at.isoformat(),
        "source_published_at": doc.source_published_at.isoformat()
        if doc.source_published_at
        else None,
        "metadata": to_plain(doc.metadata),
        "superseded_publicly_at": doc.superseded_publicly_at.isoformat()
        if doc.superseded_publicly_at
        else None,
        "superseded_in_system_at": doc.superseded_in_system_at.isoformat()
        if doc.superseded_in_system_at
        else None,
    }


def _deserialize(data: dict[str, Any]) -> SourceDocument:
    return SourceDocument(
        document_version_id=data["document_version_id"],
        source_id=data["source_id"],
        source_type=data["source_type"],
        document_location=data["document_location"],
        raw_content=RawDocumentContent(data["raw_text"]),
        content_hash=data["content_hash"],
        publicly_available_at=dt.datetime.fromisoformat(data["publicly_available_at"]),
        ingested_at=dt.datetime.fromisoformat(data["ingested_at"]),
        recorded_at=dt.datetime.fromisoformat(data["recorded_at"]),
        source_published_at=dt.datetime.fromisoformat(data["source_published_at"])
        if data["source_published_at"]
        else None,
        metadata=data["metadata"],
        superseded_publicly_at=dt.datetime.fromisoformat(data["superseded_publicly_at"])
        if data["superseded_publicly_at"]
        else None,
        superseded_in_system_at=dt.datetime.fromisoformat(data["superseded_in_system_at"])
        if data["superseded_in_system_at"]
        else None,
    )


class JsonlSourceDocumentRepository(SourceDocumentRepository):
    def __init__(self, path: Path, *, clock: Clock = utc_now) -> None:
        self._path = path
        self._clock = clock
        if not self._path.exists():
            self._path.write_text("")

    def _read_all(self) -> list[SourceDocument]:
        text = self._path.read_text(encoding="utf-8")
        return [_deserialize(json.loads(line)) for line in text.splitlines() if line.strip()]

    def add(self, document: SourceDocument) -> None:
        if any((d.document_version_id == document.document_version_id for d in self._read_all())):
            raise ProvenanceError(
                f"document_version_id {document.document_version_id!r} already exists; version identities are permanent and never reused"
            )
        require_not_in_future(
            f"document version {document.document_version_id!r} recorded_at",
            document.recorded_at,
            self._clock,
        )
        _append(self._path, _serialize(document))

    def get_version(self, document_version_id: str) -> Optional[SourceDocument]:
        for doc in self._read_all():
            if doc.document_version_id == document_version_id:
                return doc
        return None

    def research_as_of(self, as_of: dt.datetime) -> tuple[SourceDocument, ...]:
        return tuple(
            resolve_current_per_key(
                self._read_all(), lambda d: d.source_id, as_of, AsOfMode.RESEARCH
            )
        )

    def operational_replay_as_of(self, as_of: dt.datetime) -> tuple[SourceDocument, ...]:
        return tuple(
            resolve_current_per_key(
                self._read_all(), lambda d: d.source_id, as_of, AsOfMode.OPERATIONAL_REPLAY
            )
        )

    def _lineage(
        self, source_id: str, as_of: dt.datetime, mode: AsOfMode
    ) -> Optional[SourceDocument]:
        current = resolve_current_per_key(
            [d for d in self._read_all() if d.source_id == source_id],
            lambda d: d.source_id,
            as_of,
            mode,
        )
        return current[0] if current else None

    def research_as_of_lineage(
        self, source_id: str, as_of: dt.datetime
    ) -> Optional[SourceDocument]:
        return self._lineage(source_id, as_of, AsOfMode.RESEARCH)

    def operational_replay_as_of_lineage(
        self, source_id: str, as_of: dt.datetime
    ) -> Optional[SourceDocument]:
        return self._lineage(source_id, as_of, AsOfMode.OPERATIONAL_REPLAY)


def _serialize_chunk(chunk: SourceChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "document_version_id": chunk.document_version_id,
        "locator": chunk.locator,
        "raw_text": chunk.raw_text.text,
        "content_hash": chunk.content_hash,
        "publicly_available_at": chunk.publicly_available_at.isoformat(),
        "ingested_at": chunk.ingested_at.isoformat(),
        "recorded_at": chunk.recorded_at.isoformat(),
        "superseded_publicly_at": chunk.superseded_publicly_at.isoformat()
        if chunk.superseded_publicly_at
        else None,
        "superseded_in_system_at": chunk.superseded_in_system_at.isoformat()
        if chunk.superseded_in_system_at
        else None,
    }


def _deserialize_chunk(data: dict[str, Any]) -> SourceChunk:
    return SourceChunk(
        chunk_id=data["chunk_id"],
        document_version_id=data["document_version_id"],
        locator=data["locator"],
        raw_text=RawDocumentContent(data["raw_text"]),
        content_hash=data["content_hash"],
        publicly_available_at=dt.datetime.fromisoformat(data["publicly_available_at"]),
        ingested_at=dt.datetime.fromisoformat(data["ingested_at"]),
        recorded_at=dt.datetime.fromisoformat(data["recorded_at"]),
        superseded_publicly_at=dt.datetime.fromisoformat(data["superseded_publicly_at"])
        if data["superseded_publicly_at"]
        else None,
        superseded_in_system_at=dt.datetime.fromisoformat(data["superseded_in_system_at"])
        if data["superseded_in_system_at"]
        else None,
    )


class JsonlSourceChunkRepository(SourceChunkRepository):
    def __init__(
        self, path: Path, *, documents: SourceDocumentRepository, clock: Clock = utc_now
    ) -> None:
        self._path = path
        self._documents = documents
        self._clock = clock
        if not self._path.exists():
            self._path.write_text("")

    def _read_all(self) -> list[SourceChunk]:
        text = self._path.read_text(encoding="utf-8")
        return [_deserialize_chunk(json.loads(line)) for line in text.splitlines() if line.strip()]

    def add(self, chunk: SourceChunk) -> None:
        if any((c.chunk_id == chunk.chunk_id for c in self._read_all())):
            raise ProvenanceError(f"chunk_id {chunk.chunk_id!r} already exists")
        validate_chunk_registration(
            chunk, self._documents.get_version(chunk.document_version_id), self._clock
        )
        _append(self._path, _serialize_chunk(chunk))

    def for_document_version(self, document_version_id: str) -> tuple[SourceChunk, ...]:
        return tuple((c for c in self._read_all() if c.document_version_id == document_version_id))

    def get(self, chunk_id: str) -> Optional[SourceChunk]:
        for chunk in self._read_all():
            if chunk.chunk_id == chunk_id:
                return chunk
        return None
