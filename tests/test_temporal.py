"""Historical invariants are checked across ingestion order and every backend."""

import unittest
from dataclasses import FrozenInstanceError, replace
from itertools import permutations
from pathlib import Path
from tempfile import TemporaryDirectory

from auditable_ai.documents import InMemorySourceDocumentRepository
from auditable_ai.errors import AmbiguousRevisionError, ProvenanceError, TemporalIntegrityError
from auditable_ai.storage import JsonlSourceDocumentRepository
from auditable_ai.temporal import AsOfMode, resolve_current_per_key
from tests.helpers import at, setup


class DocumentRepositoryContract:
    """Every document backend must pass these checks; subclass it for a new backend."""

    def new_repository(self, **options):
        raise NotImplementedError

    def reopen(self, repository):
        return repository

    def setUp(self):
        _, fixtures, _ = setup()
        self.original = fixtures.get_version("pilot-v1")
        self.correction = fixtures.get_version("pilot-v2")
        self.documents = self.new_repository()
        for record in (self.correction, self.original):
            self.documents.add(record)

    def stored(self):
        return self.reopen(self.documents)

    def test_late_backfill_is_public_but_not_recorded_at_earlier_cutoff(self):
        self.assertEqual(self.stored().research_as_of(at("05")), (self.original,))
        self.assertEqual(self.stored().operational_replay_as_of(at("05")), ())

    def test_public_correction_can_precede_its_recording(self):
        self.assertEqual(self.stored().research_as_of(at("17")), (self.correction,))
        self.assertEqual(self.stored().operational_replay_as_of(at("17")), (self.original,))

    def test_cutoff_includes_exact_timestamp(self):
        self.assertEqual(self.stored().research_as_of(at("15")), (self.correction,))
        self.assertEqual(self.stored().operational_replay_as_of(at("20")), (self.correction,))

    def test_future_source_is_absent(self):
        self.assertEqual(self.stored().research_as_of(at("01").replace(hour=11)), ())

    def test_lineage_queries_answer_the_same_two_questions(self):
        for cutoff, public, recorded in (
            (at("05"), self.original, None),
            (at("17"), self.correction, self.original),
            (at("25"), self.correction, self.correction),
        ):
            with self.subTest(cutoff=cutoff):
                stored = self.stored()
                self.assertEqual(stored.research_as_of_lineage("sensor-pilot", cutoff), public)
                self.assertEqual(
                    stored.operational_replay_as_of_lineage("sensor-pilot", cutoff), recorded
                )

    def test_query_never_overwrites_historical_versions(self):
        before = self.stored().get_version("pilot-v1")
        self.stored().research_as_of(at("25"))
        self.assertEqual(self.stored().get_version("pilot-v1"), before)
        with self.assertRaises(FrozenInstanceError):
            before.source_id = "edited"

    def test_stored_versions_round_trip_unchanged(self):
        annotated = replace(
            self.original,
            document_version_id="annotated-v1",
            source_id="annotated",
            metadata={"labels": ["synthetic", "pilot"], "reviewed": None},
            superseded_publicly_at=at("15"),
            superseded_in_system_at=at("20"),
        )
        self.documents.add(annotated)
        for record in (self.original, self.correction, annotated):
            self.assertEqual(self.stored().get_version(record.document_version_id), record)
        self.assertIsNone(self.stored().get_version("missing"))

    def test_duplicate_version_id_is_refused(self):
        with self.assertRaises(ProvenanceError):
            self.stored().add(self.original)

    def test_recording_time_cannot_be_later_than_the_clock(self):
        repository = self.new_repository(clock=lambda: at("15"))
        repository.add(self.original)
        with self.assertRaises(TemporalIntegrityError):
            repository.add(self.correction)
        self.assertEqual(
            self.reopen(repository).operational_replay_as_of(at("25")), (self.original,)
        )

    def test_every_insertion_order_has_the_same_history(self):
        third = replace(self.original, document_version_id="other-v1", source_id="other")
        for records in permutations((self.original, self.correction, third)):
            repository = self.new_repository()
            for record in records:
                repository.add(record)
            repository = self.reopen(repository)
            for cutoff in (at("05"), at("17"), at("25")):
                for mode in AsOfMode:
                    expected = sorted(
                        doc.document_version_id
                        for doc in resolve_current_per_key(
                            records, lambda doc: doc.source_id, cutoff, mode
                        )
                    )
                    query = (
                        repository.research_as_of
                        if mode is AsOfMode.RESEARCH
                        else repository.operational_replay_as_of
                    )
                    self.assertEqual(
                        sorted(doc.document_version_id for doc in query(cutoff)), expected
                    )

    def test_tied_revisions_fail_instead_of_using_identity_or_order(self):
        tied = replace(self.correction, document_version_id="ambiguous-v3")
        for records in permutations((self.correction, tied)):
            repository = self.new_repository()
            for record in records:
                repository.add(record)
            repository = self.reopen(repository)
            for query in (repository.research_as_of, repository.operational_replay_as_of):
                with self.assertRaises(AmbiguousRevisionError):
                    query(at("25"))

    def test_unrelated_ambiguity_cannot_block_a_scoped_query(self):
        other = replace(self.original, document_version_id="other-v1", source_id="other")
        self.documents.add(other)
        self.documents.add(replace(other, document_version_id="other-v2"))
        stored = self.stored()
        for whole, scoped in (
            (stored.research_as_of, stored.research_as_of_lineage),
            (stored.operational_replay_as_of, stored.operational_replay_as_of_lineage),
        ):
            with self.assertRaises(AmbiguousRevisionError):
                whole(at("25"))
            self.assertEqual(scoped("sensor-pilot", at("25")), self.correction)

    def test_supersession_annotations_do_not_override_shared_resolution(self):
        annotated = replace(self.original, superseded_publicly_at=at("02"))
        repository = self.new_repository()
        repository.add(annotated)
        self.assertEqual(self.reopen(repository).research_as_of(at("05")), (annotated,))


class InMemoryDocuments(DocumentRepositoryContract, unittest.TestCase):
    def new_repository(self, **options):
        return InMemorySourceDocumentRepository(**options)


class JsonlDocuments(DocumentRepositoryContract, unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.paths = {}
        super().setUp()

    def new_repository(self, **options):
        path = self.directory / f"documents-{len(self.paths)}.jsonl"
        repository = JsonlSourceDocumentRepository(path, **options)
        self.paths[repository] = path
        return repository


class ReopenedJsonlDocuments(JsonlDocuments):
    def reopen(self, repository):
        return JsonlSourceDocumentRepository(self.paths[repository])


class RecordTimestamps(unittest.TestCase):
    def test_invalid_record_timestamps_are_rejected(self):
        original = setup()[1].get_version("pilot-v1")
        for values in (
            {"publicly_available_at": at("01").replace(tzinfo=None)},
            {"recorded_at": at("05")},
            {"superseded_publicly_at": at("01").replace(hour=11)},
            {"superseded_in_system_at": at("09")},
        ):
            with self.subTest(values=values), self.assertRaises(TemporalIntegrityError):
                replace(original, **values)
