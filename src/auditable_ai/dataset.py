"""Synthetic figure-extraction documents, gold annotations and number parsing.

The documents and gold annotations were written by an AI model and have not
been reviewed by a person; `Dataset.provenance` says so and every evaluation
report repeats it. Each sentence is registered as a chunk so that a model's
citations can be checked against exact, immutable text."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from importlib.resources import files
from pathlib import Path
from typing import Any

from auditable_ai._immutable import canonical_hash
from auditable_ai.claims import ClaimRepository
from auditable_ai.documents import (
    InMemorySourceChunkRepository,
    InMemorySourceDocumentRepository,
    RawDocumentContent,
    SourceChunk,
    SourceDocument,
)

# Bundled datasets for the figure-extraction-v1 task. They share its prompt, schema and
# rubric; scores are comparable only within one dataset.
DATASETS = {
    "figure-extraction-v1": "datasets/figure_extraction_v1.json",
    "figure-extraction-v1-hard": "datasets/figure_extraction_v1_hard.json",
}
DEFAULT_DATASET = "figure-extraction-v1"
STATUSES = ("current", "superseded", "disputed", "attributed")
_NUMBER = re.compile(r"(?<![\w.,])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def numbers_in(text: str) -> tuple[float, ...]:
    """Every number written with digits, with thousands separators removed."""
    return tuple(float(match.group().replace(",", "")) for match in _NUMBER.finditer(text))


def format_number(value: float) -> str:
    return str(int(value)) if value.is_integer() else repr(value)


def normalize_words(text: str) -> str:
    """Lowercase, with hyphens treated as spaces, for keyword matching."""
    return " ".join(re.sub(r"[-‐-―]", " ", text.lower()).split())


@dataclass(frozen=True)
class GoldFigure:
    id: str
    value: float
    unit: str
    quantity: str
    status: str
    chunks: tuple[str, ...]
    counter_chunks: tuple[str, ...] = ()
    description_keywords: tuple[str, ...] = ()
    candidate_statuses: tuple[str, ...] = ()
    note: str | None = None


@dataclass(frozen=True)
class GoldUnknown:
    id: str
    description: str
    chunks: tuple[str, ...]
    match_any: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class Trap:
    """A number in the text that must not be reported as an established figure."""

    value: float
    chunks: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class DatasetDocument:
    document_version_id: str
    source_id: str
    title: str
    split: str
    publicly_available_at: datetime
    recorded_at: datetime
    sentences: tuple[str, ...]
    figures: tuple[GoldFigure, ...] = ()
    unknowns: tuple[GoldUnknown, ...] = ()
    traps: tuple[Trap, ...] = ()
    non_figure_numbers: tuple[float, ...] = ()
    phenomena: tuple[str, ...] = ()

    @property
    def text(self) -> str:
        return " ".join(self.sentences)

    def chunk_id(self, sentence_number: int) -> str:
        return f"{self.document_version_id}:s{sentence_number}"

    @property
    def chunk_texts(self) -> dict[str, str]:
        return {
            self.chunk_id(number): sentence
            for number, sentence in enumerate(self.sentences, start=1)
        }


@dataclass(frozen=True)
class Dataset:
    name: str
    task_version: str
    frozen_on: str
    provenance: dict[str, Any]
    splits: dict[str, str]
    documents: tuple[DatasetDocument, ...]
    sha256: str

    def split(self, name: str) -> tuple[DatasetDocument, ...]:
        if name == "all":
            return self.documents
        if name not in self.splits:
            raise ValueError(f"unknown split {name!r}; choose from {sorted(self.splits)} or 'all'")
        return tuple(document for document in self.documents if document.split == name)

    def document(self, document_version_id: str) -> DatasetDocument:
        for document in self.documents:
            if document.document_version_id == document_version_id:
                return document
        raise KeyError(f"no dataset document {document_version_id!r}")


def load_dataset(name: str = DEFAULT_DATASET) -> Dataset:
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose from {', '.join(DATASETS)}")
    data = json.loads(files("auditable_ai").joinpath(DATASETS[name]).read_text(encoding="utf-8"))
    return Dataset(
        name=data["dataset"],
        task_version=data["task_version"],
        frozen_on=data["frozen_on"],
        provenance=data["provenance"],
        splits=data["splits"],
        documents=tuple(document_from_entry(entry) for entry in data["documents"]),
        sha256=canonical_hash(data),
    )


def bundled_document(document_version_id: str) -> DatasetDocument:
    """A document from any bundled dataset; version IDs are unique across them."""
    for name in DATASETS:
        try:
            return load_dataset(name).document(document_version_id)
        except KeyError:
            continue
    raise KeyError(f"no bundled dataset document {document_version_id!r}")


def load_document_file(path: Path) -> DatasetDocument:
    """Read one document in the dataset's JSON shape; gold annotations are optional."""
    return document_from_entry(json.loads(path.read_text(encoding="utf-8")))


def document_from_entry(entry: dict[str, Any]) -> DatasetDocument:
    version_id = entry["document_version_id"]
    gold = entry.get("gold", {})

    def chunks(sentence_numbers: list[int]) -> tuple[str, ...]:
        return tuple(f"{version_id}:s{number}" for number in sentence_numbers)

    return DatasetDocument(
        document_version_id=version_id,
        source_id=entry["source_id"],
        title=entry["title"],
        split=entry.get("split", "external"),
        publicly_available_at=datetime.fromisoformat(entry["publicly_available_at"]),
        recorded_at=datetime.fromisoformat(entry["recorded_at"]),
        sentences=tuple(entry["sentences"]),
        figures=tuple(
            GoldFigure(
                id=figure["id"],
                value=float(figure["value"]),
                unit=figure["unit"],
                quantity=figure["quantity"],
                status=figure["status"],
                chunks=chunks(figure["chunks"]),
                counter_chunks=chunks(figure.get("counter_chunks", [])),
                description_keywords=tuple(figure.get("description_keywords", [])),
                candidate_statuses=tuple(figure.get("candidate_statuses", [])),
                note=figure.get("note"),
            )
            for figure in gold.get("figures", [])
        ),
        unknowns=tuple(
            GoldUnknown(
                id=unknown["id"],
                description=unknown["description"],
                chunks=chunks(unknown["chunks"]),
                match_any=tuple(tuple(group) for group in unknown["match_any"]),
            )
            for unknown in gold.get("unknowns", [])
        ),
        traps=tuple(
            Trap(value=float(trap["value"]), chunks=chunks(trap["chunks"]), reason=trap["reason"])
            for trap in gold.get("traps", [])
        ),
        non_figure_numbers=tuple(float(number) for number in gold.get("non_figure_numbers", [])),
        phenomena=tuple(entry.get("phenomena", [])),
    )


def document_to_entry(document: DatasetDocument) -> dict[str, Any]:
    """The JSON shape of a document without gold annotations, as embedded in run records."""
    return {
        "document_version_id": document.document_version_id,
        "source_id": document.source_id,
        "title": document.title,
        "split": document.split,
        "publicly_available_at": document.publicly_available_at.isoformat(),
        "recorded_at": document.recorded_at.isoformat(),
        "sentences": list(document.sentences),
    }


def register(document: DatasetDocument) -> tuple[SourceDocument, ClaimRepository]:
    """Store one document version, and one chunk per sentence, in fresh repositories."""
    documents = InMemorySourceDocumentRepository()
    content = RawDocumentContent(document.text)
    source = SourceDocument(
        document_version_id=document.document_version_id,
        source_id=document.source_id,
        source_type="operational_report",
        document_location=f"figure-task://{document.document_version_id}",
        raw_content=content,
        content_hash=content.content_hash(),
        publicly_available_at=document.publicly_available_at,
        ingested_at=document.recorded_at,
        recorded_at=document.recorded_at,
        metadata={"title": document.title, "split": document.split},
    )
    documents.add(source)
    chunks = InMemorySourceChunkRepository(documents)
    for number, sentence in enumerate(document.sentences, start=1):
        raw = RawDocumentContent(sentence)
        chunks.add(
            SourceChunk(
                document.chunk_id(number),
                document.document_version_id,
                f"sentence {number}",
                raw,
                raw.content_hash(),
                document.publicly_available_at,
                document.recorded_at,
                document.recorded_at,
            )
        )
    return source, ClaimRepository(documents, chunks)


def check_dataset(dataset: Dataset) -> list[str]:
    """Programmatic integrity checks over every document and annotation."""
    problems: list[str] = []
    seen: set[str] = set()
    for document in dataset.documents:
        name = document.document_version_id
        if name in seen:
            problems.append(f"{name}: duplicate document_version_id")
        seen.add(name)
        if document.split not in dataset.splits:
            problems.append(f"{name}: unknown split {document.split!r}")
        texts = document.chunk_texts
        values = [figure.value for figure in document.figures]
        annotated = set(values) | {trap.value for trap in document.traps}
        annotated |= set(document.non_figure_numbers)
        for number in numbers_in(document.text):
            if number not in annotated:
                problems.append(f"{name}: number {format_number(number)} is not annotated")
        if len(values) != len(set(values)):
            problems.append(f"{name}: two gold figures share a value")
        if set(values) & set(document.non_figure_numbers):
            problems.append(f"{name}: a value is annotated both as a figure and as a non-figure")
        for figure in document.figures:
            label = f"{name}/{figure.id}"
            if figure.status == "ambiguous":
                if len(figure.candidate_statuses) < 2 or not figure.note:
                    problems.append(f"{label}: an ambiguous status needs candidates and a note")
            elif figure.status not in STATUSES:
                problems.append(f"{label}: unknown status {figure.status!r}")
            if figure.status == "disputed" and not figure.counter_chunks:
                problems.append(f"{label}: a disputed figure needs counter_chunks")
            for chunk_id in (*figure.chunks, *figure.counter_chunks):
                if chunk_id not in texts:
                    problems.append(f"{label}: unknown chunk {chunk_id}")
            for chunk_id in figure.chunks:
                if chunk_id in texts and figure.value not in numbers_in(texts[chunk_id]):
                    problems.append(f"{label}: value is not stated in {chunk_id}")
        for unknown in document.unknowns:
            cited = " ".join(texts.get(chunk_id, "") for chunk_id in unknown.chunks)
            words = normalize_words(cited)
            if not any(all(keyword in words for keyword in group) for group in unknown.match_any):
                problems.append(f"{name}/{unknown.id}: no keyword group appears in its own chunks")
        for trap in document.traps:
            if not any(trap.value in numbers_in(texts.get(c, "")) for c in trap.chunks):
                problems.append(f"{name}: trap {format_number(trap.value)} is not in its chunks")
    return problems
