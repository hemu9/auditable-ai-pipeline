"""Evidence anchors pass the same integrity checks in every chunk backend."""

import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from auditable_ai import (
    InMemorySourceChunkRepository,
    InMemorySourceDocumentRepository,
    ProvenanceError,
    TemporalIntegrityError,
)
from auditable_ai.storage import JsonlSourceChunkRepository, JsonlSourceDocumentRepository
from tests.helpers import at, chunk, setup


class SourceIntegrity(unittest.TestCase):
    def test_document_content_must_match_its_supplied_hash(self):
        _, documents, _ = setup()
        with self.assertRaises(ProvenanceError):
            replace(documents.get_version("pilot-v1"), content_hash="incorrect")

    def test_chunk_content_must_match_its_supplied_hash(self):
        anchor = chunk("count", "pilot-v2", "24 sensors", public="15", recorded="20")
        with self.assertRaises(ProvenanceError):
            replace(anchor, content_hash="incorrect")

    def test_caller_mutation_cannot_change_nested_metadata(self):
        _, documents, _ = setup()
        metadata = {"labels": ["original"]}
        document = replace(documents.get_version("pilot-v1"), metadata=metadata)
        metadata["labels"].append("later")
        self.assertEqual(document.metadata["labels"], ("original",))
        with self.assertRaises(TypeError):
            document.metadata["new"] = "mutation"


class ChunkRepositoryContract:
    """Every chunk backend must pass these checks; subclass it for a new backend."""

    def new_repositories(self, **options):
        raise NotImplementedError

    def reopen(self, chunks):
        return chunks

    def setUp(self):
        _, fixtures, _ = setup()
        self.documents, self.chunks = self.new_repositories()
        for name in ("pilot-v1", "pilot-v2"):
            self.documents.add(fixtures.get_version(name))

    def test_anchor_remains_bound_to_its_version_after_a_correction(self):
        anchor = chunk("original-count", "pilot-v1", "36 sensors", public="01", recorded="10")
        self.chunks.add(anchor)
        self.assertEqual(self.documents.research_as_of(at("25"))[0].document_version_id, "pilot-v2")
        stored = self.reopen(self.chunks)
        self.assertEqual(stored.get("original-count"), anchor)
        self.assertEqual(stored.for_document_version("pilot-v1"), (anchor,))
        self.assertEqual(stored.for_document_version("pilot-v2"), ())
        self.assertIsNone(stored.get("missing"))

    def test_duplicate_chunk_identity_is_refused(self):
        anchor = chunk("count", "pilot-v2", "24 sensors", public="15", recorded="20")
        self.chunks.add(anchor)
        with self.assertRaises(ProvenanceError):
            self.reopen(self.chunks).add(anchor)

    def test_unknown_parent_version_is_refused(self):
        orphan = chunk("unknown", "missing-version", "24 sensors", public="15", recorded="20")
        with self.assertRaises(ProvenanceError):
            self.chunks.add(orphan)

    def test_chunk_cannot_predate_its_document_version(self):
        for public, recorded in (("14", "20"), ("15", "19")):
            early = chunk(
                f"early-{public}-{recorded}",
                "pilot-v2",
                "24 sensors",
                public=public,
                recorded=recorded,
            )
            with self.subTest(public=public, recorded=recorded):
                with self.assertRaises(TemporalIntegrityError):
                    self.chunks.add(early)

    def test_chunk_must_quote_the_version_it_names(self):
        for text in ("24 sensors were installed", ""):
            misanchored = chunk(
                f"misanchored-{len(text)}", "pilot-v1", text, public="01", recorded="10"
            )
            with self.subTest(text=text), self.assertRaises(ProvenanceError):
                self.chunks.add(misanchored)
        self.assertEqual(self.reopen(self.chunks).for_document_version("pilot-v1"), ())

    def test_recording_time_cannot_be_later_than_the_clock(self):
        documents, chunks = self.new_repositories(clock=lambda: at("21"))
        documents.add(self.documents.get_version("pilot-v2"))
        with self.assertRaises(TemporalIntegrityError):
            chunks.add(chunk("late", "pilot-v2", "24 sensors", public="15", recorded="22"))
        chunks.add(chunk("current", "pilot-v2", "24 sensors", public="15", recorded="21"))
        stored = self.reopen(chunks).for_document_version("pilot-v2")
        self.assertEqual([anchor.chunk_id for anchor in stored], ["current"])


class InMemoryChunks(ChunkRepositoryContract, unittest.TestCase):
    def new_repositories(self, **options):
        documents = InMemorySourceDocumentRepository(**options)
        return documents, InMemorySourceChunkRepository(documents, **options)


class JsonlChunks(ChunkRepositoryContract, unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.paths = {}
        super().setUp()

    def new_repositories(self, **options):
        document_path = self.directory / f"documents-{len(self.paths)}.jsonl"
        chunk_path = self.directory / f"chunks-{len(self.paths)}.jsonl"
        documents = JsonlSourceDocumentRepository(document_path, **options)
        chunks = JsonlSourceChunkRepository(chunk_path, documents=documents, **options)
        self.paths[chunks] = (document_path, chunk_path)
        return documents, chunks


class ReopenedJsonlChunks(JsonlChunks):
    def reopen(self, chunks):
        document_path, chunk_path = self.paths[chunks]
        return JsonlSourceChunkRepository(
            chunk_path, documents=JsonlSourceDocumentRepository(document_path)
        )
