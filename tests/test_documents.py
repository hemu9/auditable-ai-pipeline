import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from auditable_ai import ProvenanceError, RawDocumentContent, SourceChunk
from auditable_ai.storage import JsonlSourceChunkRepository, JsonlSourceDocumentRepository
from tests.helpers import at, setup


class SourceIntegrity(unittest.TestCase):
    def test_document_content_must_match_its_supplied_hash(self):
        _, documents, _ = setup()
        with self.assertRaises(ValueError):
            replace(documents.get_version("pilot-v1"), content_hash="incorrect")

    def test_caller_mutation_cannot_change_nested_metadata(self):
        _, documents, _ = setup()
        metadata = {"labels": ["original"]}
        document = replace(documents.get_version("pilot-v1"), metadata=metadata)
        metadata["labels"].append("later")
        self.assertEqual(document.metadata["labels"], ("original",))
        with self.assertRaises(TypeError):
            document.metadata["new"] = "mutation"

    def test_chunk_anchor_remains_bound_to_original_after_a_correction(self):
        _, documents, claims = setup()
        raw = RawDocumentContent("36 sensors")
        chunk = SourceChunk(
            "original-count",
            "pilot-v1",
            "sentence 1",
            raw,
            raw.content_hash(),
            at("01"),
            at("10"),
            at("10"),
        )
        claims.chunks.add(chunk)
        self.assertEqual(documents.research_as_of(at("25"))[0].document_version_id, "pilot-v2")
        self.assertEqual(claims.chunks.get("original-count").document_version_id, "pilot-v1")

    def test_chunk_storage_reopens_with_same_evidence_and_rejects_duplicate_identity(self):
        _, documents, _ = setup()
        raw = RawDocumentContent("24 sensors")
        chunk = SourceChunk(
            "count", "pilot-v2", "sentence 1", raw, raw.content_hash(), at("15"), at("20"), at("20")
        )
        with TemporaryDirectory() as tmp:
            document_path = Path(tmp) / "documents.jsonl"
            chunk_path = Path(tmp) / "chunks.jsonl"
            disk_documents = JsonlSourceDocumentRepository(document_path)
            for name in ("pilot-v1", "pilot-v2"):
                disk_documents.add(documents.get_version(name))
            disk_chunks = JsonlSourceChunkRepository(chunk_path, documents=disk_documents)
            disk_chunks.add(chunk)
            reopened = JsonlSourceChunkRepository(
                chunk_path, documents=JsonlSourceDocumentRepository(document_path)
            )
            self.assertEqual(reopened.get("count"), chunk)
            with self.assertRaises(ValueError):
                reopened.add(chunk)

    def test_unknown_chunk_parent_is_rejected_by_both_backends(self):
        _, documents, claims = setup()
        raw = RawDocumentContent("unknown source")
        chunk = SourceChunk(
            "unknown",
            "missing-version",
            "sentence 1",
            raw,
            raw.content_hash(),
            at("15"),
            at("20"),
            at("20"),
        )
        with TemporaryDirectory() as tmp:
            disk = JsonlSourceChunkRepository(Path(tmp) / "chunks.jsonl", documents=documents)
            for backend in (claims.chunks, disk):
                with self.assertRaises(ProvenanceError):
                    backend.add(chunk)
