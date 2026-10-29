"""Historical invariants are checked across ingestion order and backends."""

import unittest
from dataclasses import FrozenInstanceError, replace
from itertools import permutations
from pathlib import Path
from tempfile import TemporaryDirectory

from auditable_ai.documents import InMemorySourceDocumentRepository
from auditable_ai.errors import AmbiguousRevisionError, TemporalIntegrityError
from auditable_ai.storage import JsonlSourceDocumentRepository
from auditable_ai.temporal import AsOfMode, resolve_current_per_key
from tests.helpers import at, setup


class HistoricalQueries(unittest.TestCase):
    def setUp(self):
        _, self.documents, _ = setup()
        self.original = self.documents.get_version("pilot-v1")
        self.correction = self.documents.get_version("pilot-v2")

    def test_late_backfill_is_public_but_not_recorded_at_earlier_cutoff(self):
        self.assertEqual(self.documents.research_as_of(at("05")), (self.original,))
        self.assertEqual(self.documents.operational_replay_as_of(at("05")), ())

    def test_public_correction_can_precede_its_recording(self):
        self.assertEqual(self.documents.research_as_of(at("17")), (self.correction,))
        self.assertEqual(self.documents.operational_replay_as_of(at("17")), (self.original,))

    def test_cutoff_includes_exact_timestamp(self):
        self.assertEqual(self.documents.research_as_of(at("15")), (self.correction,))
        self.assertEqual(self.documents.operational_replay_as_of(at("20")), (self.correction,))

    def test_future_source_is_absent(self):
        self.assertEqual(self.documents.research_as_of(at("01").replace(hour=11)), ())

    def test_query_never_overwrites_historical_versions(self):
        before = self.documents.get_version("pilot-v1")
        self.documents.research_as_of(at("25"))
        self.assertIs(self.documents.get_version("pilot-v1"), before)
        with self.assertRaises(FrozenInstanceError):
            before.source_id = "edited"

    def test_duplicate_version_id_is_refused(self):
        with self.assertRaises(ValueError):
            self.documents.add(self.original)

    def test_every_insertion_order_has_the_same_history_in_both_backends(self):
        third = replace(self.original, document_version_id="other-v1", source_id="other")
        with TemporaryDirectory() as tmp:
            for index, records in enumerate(permutations((self.original, self.correction, third))):
                memory = InMemorySourceDocumentRepository()
                disk = JsonlSourceDocumentRepository(Path(tmp) / f"documents-{index}.jsonl")
                for record in records:
                    memory.add(record)
                    disk.add(record)
                reopened = JsonlSourceDocumentRepository(Path(tmp) / f"documents-{index}.jsonl")
                for cutoff in (at("05"), at("17"), at("25")):
                    for mode in AsOfMode:
                        expected = sorted(
                            doc.document_version_id
                            for doc in resolve_current_per_key(
                                records, lambda doc: doc.source_id, cutoff, mode
                            )
                        )
                        for backend in (memory, disk, reopened):
                            query = (
                                backend.research_as_of
                                if mode is AsOfMode.RESEARCH
                                else backend.operational_replay_as_of
                            )
                            self.assertEqual(
                                sorted(doc.document_version_id for doc in query(cutoff)), expected
                            )

    def test_tied_revisions_fail_instead_of_using_identity_or_order(self):
        tied = replace(self.correction, document_version_id="ambiguous-v3")
        for records in permutations((self.correction, tied)):
            repository = InMemorySourceDocumentRepository()
            for record in records:
                repository.add(record)
            for query in (repository.research_as_of, repository.operational_replay_as_of):
                with self.assertRaises(AmbiguousRevisionError):
                    query(at("25"))

    def test_unrelated_ambiguity_cannot_block_a_scoped_query(self):
        other = replace(self.original, document_version_id="other-v1", source_id="other")
        self.documents.add(other)
        self.documents.add(replace(other, document_version_id="other-v2"))
        with self.assertRaises(AmbiguousRevisionError):
            self.documents.research_as_of(at("25"))
        self.assertEqual(
            self.documents.research_as_of_lineage("sensor-pilot", at("25")), self.correction
        )

    def test_invalid_record_timestamps_are_rejected(self):
        for values in (
            {"publicly_available_at": at("01").replace(tzinfo=None)},
            {"recorded_at": at("05")},
            {"superseded_publicly_at": at("01").replace(hour=11)},
        ):
            with self.subTest(values=values), self.assertRaises(TemporalIntegrityError):
                replace(self.original, **values)

    def test_supersession_annotations_do_not_override_shared_resolution(self):
        annotated = replace(self.original, superseded_publicly_at=at("02"))
        repository = InMemorySourceDocumentRepository()
        repository.add(annotated)
        self.assertEqual(repository.research_as_of(at("05")), (annotated,))
