"""Trust-boundary tests include explicit demonstrations of semantic limits."""

import json
import unittest
from dataclasses import replace
from math import inf, nan
from pathlib import Path
from tempfile import TemporaryDirectory

from auditable_ai import (
    ClaimCategory,
    ClaimRepository,
    DuplicateRunError,
    EvidenceExtractor,
    ExtractionValidationError,
    InMemorySourceChunkRepository,
    ProvenanceError,
    RawDocumentContent,
    SourceChunk,
    TemporalIntegrityError,
)
from auditable_ai.storage import JsonlSourceChunkRepository
from tests.helpers import at, chunk, extract, setup, valid_response


class ExtractionBoundary(unittest.TestCase):
    def test_valid_result_preserves_categories_unknowns_and_counterevidence(self):
        result, _, _ = extract(valid_response())
        self.assertEqual(
            [claim.category for claim in result.claims],
            [ClaimCategory.FACT, ClaimCategory.INFERENCE],
        )
        self.assertEqual(len(result.unknowns), 1)
        self.assertEqual(result.contradicting_evidence[0].source_id, "pilot-v1")
        self.assertIsNone(result.confidence)
        self.assertEqual(result.claims[0].missing_information, result.unknowns)

    def test_missing_explicit_fact_reference_is_disclosed_as_input_attribution(self):
        result, _, _ = extract({"facts": [{"claim_text": "24 sensors"}]})
        self.assertEqual(result.claims[0].supporting_evidence[0].source_id, "pilot-v2")

    def test_fact_evidence_is_normalized_before_it_is_validated(self):
        result, _, _ = extract({"facts": [{"claim_text": "24 sensors"}]}, run_id="seed")
        attributed = result.claims[0]
        cases = (
            iter(()),
            (reference for reference in ()),
            [],
            "pilot-v2",
            [{"source_id": "pilot-v2"}],
        )
        for evidence in cases:
            with self.subTest(evidence=repr(evidence)), self.assertRaises(ValueError):
                replace(attributed, claim_id="unattributed", supporting_evidence=evidence)
        rebuilt = replace(attributed, supporting_evidence=iter(attributed.supporting_evidence))
        self.assertEqual(rebuilt.supporting_evidence, attributed.supporting_evidence)

    def test_claim_text_collections_must_hold_strings(self):
        result, _, _ = extract({"facts": [{"claim_text": "24 sensors"}]})
        for field, value in (
            ("missing_information", "unmeasured"),
            ("contradicting_claim_ids", [7]),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                replace(result.claims[0], **{field: value})

    def test_storage_refuses_a_fact_without_supporting_evidence(self):
        result, _, claims = extract({"facts": [{"claim_text": "24 sensors"}]}, run_id="seed")
        unattributed = replace(result.claims[0], claim_id="bypassed", run_id="bypass")
        object.__setattr__(unattributed, "supporting_evidence", ())
        with self.assertRaisesRegex(ProvenanceError, "FACT cannot be stored"):
            claims.add(unattributed)
        self.assertIsNone(claims.get("bypassed"))
        self.assertFalse(claims.has_run("bypass"))
        untyped = replace(result.claims[0], claim_id="untyped", run_id="untyped")
        object.__setattr__(untyped, "supporting_evidence", [{"source_id": "pilot-v2"}])
        with self.assertRaisesRegex(ProvenanceError, "tuple of EvidenceReference"):
            claims.add(untyped)

    def test_fabricated_citation_does_not_discard_a_valid_sibling(self):
        bundle = setup()
        case = bundle[0]["cases"][1]
        result, _, claims = extract(case["response"], bundle=bundle)
        self.assertEqual(len(result.claims), 1)
        self.assertEqual(len(result.rejected_items), 1)
        self.assertEqual(result.rejected_items[0].reason.value, "PROVENANCE_VIOLATION")
        self.assertIsNone(claims.get("test-run-claim-2"))

    def test_no_tool_field_can_enter_the_accepted_schema(self):
        for field in ("execute_command", "tools", "system_prompt"):
            with self.subTest(field=field), self.assertRaises(ExtractionValidationError):
                extract({"facts": [], field: "do something"})

    def test_invalid_sibling_fields_are_itemized(self):
        response = {
            "facts": [
                {"claim_text": "24 sensors"},
                {"claim_text": "attack", "execute_command": "do something"},
            ]
        }
        result, _, _ = extract(response)
        self.assertEqual(len(result.claims), 1)
        self.assertEqual(result.rejected_items[0].reason.value, "UNKNOWN_FIELD")

    def test_citation_fields_and_declarations_are_checked_per_item(self):
        result, _, _ = extract(
            {
                "facts": [
                    {
                        "claim_text": "unknown citation field",
                        "supporting_evidence": [{"source_id": "pilot-v2", "execute": "x"}],
                    },
                    {
                        "claim_text": "undeclared citation",
                        "supporting_evidence": [{"source_id": "pilot-v1"}],
                    },
                    {
                        "claim_text": "declared citation",
                        "supporting_evidence": [{"source_id": "pilot-v2"}],
                    },
                ],
                "source_ids": ["pilot-v2"],
            }
        )
        self.assertEqual([claim.claim_text for claim in result.claims], ["declared citation"])
        self.assertEqual(
            [item.reason.value for item in result.rejected_items],
            ["UNKNOWN_FIELD", "INVALID_ITEM"],
        )

    def test_well_formed_false_claim_passes_structural_and_provenance_checks(self):
        bundle = setup()
        result, _, _ = extract(bundle[0]["cases"][4]["response"], bundle=bundle)
        self.assertIn("900", result.claims[0].claim_text)
        self.assertEqual(result.rejected_items, ())

    def test_future_input_is_rejected_before_the_adapter_runs(self):
        _, documents, claims = setup()
        calls = []
        extractor = EvidenceExtractor(claims=claims, extraction_fn=lambda raw: calls.append(raw))
        with self.assertRaises(ExtractionValidationError):
            extractor.extract(
                documents.get_version("pilot-v2"),
                source_as_of=at("14"),
                extracted_at=at("21"),
                subject_id="pilot",
                run_id="future",
                model_name="fixture",
                model_version="1",
                prompt_version="none",
            )
        self.assertEqual(calls, [])

    def test_extraction_cannot_predate_source_recording(self):
        with self.assertRaises(TemporalIntegrityError):
            extract(valid_response(), extracted_at=at("19"))

    def test_naive_cutoff_is_refused(self):
        with self.assertRaises(TemporalIntegrityError):
            extract(valid_response(), source_as_of=at("21").replace(tzinfo=None))

    def test_recording_time_cannot_be_later_than_the_clock(self):
        _, documents, _ = setup()
        chunks = InMemorySourceChunkRepository(documents)
        claims = ClaimRepository(documents, chunks, clock=lambda: at("21"))
        bundle = (None, documents, claims)
        calls = []
        with self.assertRaises(TemporalIntegrityError):
            extract(valid_response(), extracted_at=at("22"), bundle=bundle, calls=calls)
        self.assertEqual(calls, [])
        current, _, _ = extract(valid_response(), bundle=bundle)
        future = replace(
            current.claims[0],
            claim_id="future-claim",
            run_id="future",
            ingested_at=at("22"),
            recorded_at=at("22"),
        )
        with self.assertRaises(TemporalIntegrityError):
            claims.add(future)
        self.assertIsNone(claims.get("future-claim"))

    def test_cited_future_evidence_cannot_leak_into_an_older_source_claim(self):
        response = {
            "facts": [
                {"claim_text": "later evidence", "supporting_evidence": [{"source_id": "pilot-v2"}]}
            ],
            "source_ids": ["pilot-v2"],
        }
        result, _, _ = extract(response, document_id="pilot-v1")
        self.assertEqual(len(result.claims), 0)
        self.assertEqual(result.rejected_items[0].reason.value, "PROVENANCE_VIOLATION")

    def test_invalid_top_level_negative_evidence_is_not_silently_trusted(self):
        with self.assertRaises(ProvenanceError):
            extract(
                {"contradicting_evidence": [{"source_id": "invented"}], "source_ids": ["invented"]}
            )

    def test_supplied_document_hash_is_checked(self):
        response = {
            "facts": [
                {
                    "claim_text": "count",
                    "supporting_evidence": [
                        {"source_id": "pilot-v2", "source_content_hash": "bad"}
                    ],
                }
            ],
            "source_ids": ["pilot-v2"],
        }
        result, _, _ = extract(response)
        self.assertEqual(result.claims, ())
        self.assertIn("hash", result.rejected_items[0].detail)

    def test_chunk_hash_and_parent_are_checked_independently(self):
        bundle = setup()
        _, _, claims = bundle
        raw = RawDocumentContent("24 sensors")
        claims.chunks.add(
            SourceChunk(
                "count-chunk",
                "pilot-v2",
                "sentence 1",
                raw,
                raw.content_hash(),
                at("15"),
                at("20"),
                at("20"),
            )
        )
        for source_id, content_hash, accepted in (
            ("pilot-v2", raw.content_hash(), 1),
            ("pilot-v2", "bad", 0),
            ("pilot-v1", raw.content_hash(), 0),
        ):
            response = {
                "facts": [
                    {
                        "claim_text": "count",
                        "supporting_evidence": [
                            {
                                "source_id": source_id,
                                "source_chunk_id": "count-chunk",
                                "source_content_hash": content_hash,
                            }
                        ],
                    }
                ],
                "source_ids": [source_id],
            }
            result, _, _ = extract(
                response, run_id=f"chunk-{source_id}-{content_hash}", bundle=bundle
            )
            self.assertEqual(len(result.claims), accepted)

    def test_chunk_recorded_after_claim_cannot_be_cited(self):
        bundle = setup()
        raw = RawDocumentContent("24 sensors")
        bundle[2].chunks.add(
            SourceChunk(
                "late-chunk",
                "pilot-v2",
                "sentence 1",
                raw,
                raw.content_hash(),
                at("15"),
                at("22"),
                at("22"),
            )
        )
        result, _, _ = extract(
            {
                "facts": [
                    {
                        "claim_text": "count",
                        "supporting_evidence": [
                            {"source_id": "pilot-v2", "source_chunk_id": "late-chunk"}
                        ],
                    }
                ],
                "source_ids": ["pilot-v2"],
            },
            bundle=bundle,
        )
        self.assertEqual(result.claims, ())

    def test_forged_input_does_not_bypass_registered_source_content(self):
        _, documents, claims = setup()
        original = documents.get_version("pilot-v2")
        content = RawDocumentContent("forged input")
        forged = replace(original, raw_content=content, content_hash=content.content_hash())
        with self.assertRaises(ProvenanceError):
            EvidenceExtractor(claims=claims, extraction_fn=lambda _: {}).extract(
                forged,
                source_as_of=at("21"),
                extracted_at=at("21"),
                subject_id="pilot",
                run_id="forged",
                model_name="fixture",
                model_version="1",
                prompt_version="none",
            )

    def test_reused_run_id_is_refused_before_the_adapter_runs(self):
        bundle = setup()
        first, _, claims = extract(valid_response(), bundle=bundle)
        calls = []
        retry = {"facts": [{"claim_text": f"retry {n}"} for n in range(3)]}
        with self.assertRaises(DuplicateRunError):
            extract(retry, document_id="pilot-v1", bundle=bundle, calls=calls)
        self.assertEqual(calls, [])
        self.assertEqual(claims.get("test-run-claim-1"), first.claims[0])
        self.assertIsNone(claims.get("test-run-claim-3"))

    def test_repository_refuses_a_second_commit_for_a_run(self):
        first, _, claims = extract(valid_response())
        with self.assertRaises(DuplicateRunError):
            claims.add(replace(first.claims[0], claim_id="test-run-extra"))
        self.assertIsNone(claims.get("test-run-extra"))

    def test_repository_refuses_inconsistent_run_batches(self):
        first, _, claims = extract(valid_response())
        stray = replace(first.claims[0], claim_id="stray")
        repeated = replace(first.claims[0], claim_id="repeated", run_id="batch")
        for run_id, batch in (("other-run", [stray]), ("batch", [repeated, repeated])):
            with self.subTest(run_id=run_id), self.assertRaises(ProvenanceError):
                claims.add_run(run_id, batch)
            self.assertFalse(claims.has_run(run_id))

    def test_run_without_accepted_claims_still_consumes_its_identity(self):
        bundle = setup()
        empty, _, claims = extract(
            {"facts": [{"claim_text": "count", "confidence": True}]}, bundle=bundle
        )
        self.assertEqual(empty.claims, ())
        self.assertTrue(claims.has_run("test-run"))
        with self.assertRaises(DuplicateRunError):
            extract(valid_response(), bundle=bundle)

    def test_malformed_batch_stores_nothing_and_leaves_the_run_id_unused(self):
        bundle = setup()
        with self.assertRaises(ExtractionValidationError):
            extract({"facts": "not a list"}, bundle=bundle)
        result, _, _ = extract(valid_response(), bundle=bundle)
        self.assertEqual(len(result.claims), 2)

    def test_run_is_committed_completely_or_not_at_all(self):
        bundle = setup()
        seed, _, claims = extract(
            {"facts": [{"claim_text": "36 sensors"}]},
            document_id="pilot-v1",
            run_id="seed",
            bundle=bundle,
        )
        claims.add(replace(seed.claims[0], claim_id="test-run-claim-2", run_id="manual"))
        with self.assertRaises(ProvenanceError):
            extract(valid_response(), bundle=bundle)
        self.assertIsNone(claims.get("test-run-claim-1"))
        self.assertFalse(claims.has_run("test-run"))

    def test_corrupt_evidence_store_fails_the_run_instead_of_rejecting_items(self):
        _, documents, _ = setup()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "chunks.jsonl"
            chunks = JsonlSourceChunkRepository(path, documents=documents)
            chunks.add(chunk("count", "pilot-v2", "24 sensors", public="15", recorded="20"))
            with path.open("a", encoding="utf-8") as stream:
                stream.write('{"chunk_id": "torn"\n')
            claims = ClaimRepository(documents, chunks)
            citation = {"source_id": "pilot-v2", "source_chunk_id": "count"}
            response = {
                "facts": [{"claim_text": "count", "supporting_evidence": [citation]}],
                "source_ids": ["pilot-v2"],
            }
            with self.assertRaises(json.JSONDecodeError):
                extract(response, bundle=(None, documents, claims))
            self.assertFalse(claims.has_run("test-run"))

    def test_conflicting_runs_are_preserved_as_separate_claims(self):
        bundle = setup()
        older, _, _ = extract(
            {"facts": [{"claim_text": "36 sensors"}]},
            document_id="pilot-v1",
            run_id="original",
            bundle=bundle,
        )
        correction, _, claims = extract(
            {
                "facts": [
                    {
                        "claim_text": "24 sensors",
                        "contradicting_claim_ids": [older.claims[0].claim_id],
                    }
                ]
            },
            run_id="correction",
            bundle=bundle,
        )
        visible = claims.operational_replay_as_of("sensor-pilot", at("25"))
        self.assertEqual(len(visible), 2)
        self.assertEqual(correction.claims[0].contradicting_claim_ids, ("original-claim-1",))

    def test_source_eligible_claims_and_recorded_model_outputs_are_distinct(self):
        result, _, claims = extract(
            {"facts": [{"claim_text": "36 sensors"}]}, document_id="pilot-v1"
        )
        self.assertEqual(claims.research_as_of("sensor-pilot", at("05")), result.claims)
        self.assertEqual(claims.operational_replay_as_of("sensor-pilot", at("05")), ())

    def test_claim_queries_are_scoped_to_their_subject(self):
        bundle = setup()
        extract({"facts": [{"claim_text": "24 sensors"}]}, run_id="pilot", bundle=bundle)
        _, _, claims = extract(
            {"facts": [{"claim_text": "other subject"}]},
            run_id="other",
            subject_id="other-subject",
            bundle=bundle,
        )
        for query in (claims.research_as_of, claims.operational_replay_as_of):
            self.assertEqual([claim.run_id for claim in query("sensor-pilot", at("25"))], ["pilot"])

    def test_validation_and_audit_share_one_json_snapshot(self):
        response = valid_response()
        response["facts"] = tuple(response["facts"])
        result, _, _ = extract(response)
        self.assertEqual(len(result.claims), 2)
        response["facts"][0]["claim_text"] = "changed later"
        self.assertNotIn("changed later", result.raw_response_json)
        self.assertNotIn("changed later", result.claims[0].claim_text)

    def test_same_input_has_same_content_hashes_independent_of_key_order(self):
        response = valid_response()
        first, _, _ = extract(response)
        second, _, _ = extract(dict(reversed(list(response.items()))))
        self.assertEqual(first.input_hash, second.input_hash)
        self.assertEqual(first.response_hash, second.response_hash)

    def test_nonfinite_or_nonjson_adapter_output_is_a_batch_error(self):
        for response in (
            {"confidence": nan},
            {"confidence": inf},
            {"facts": {"not": "a list"}},
            {"facts": [object()]},
            [],
        ):
            with self.subTest(response=response), self.assertRaises(ExtractionValidationError):
                extract(response)

    def test_deeply_nested_adapter_output_is_a_batch_error(self):
        nested = []
        for _ in range(100_000):
            nested = [nested]
        with self.assertRaises(ExtractionValidationError):
            extract({"facts": nested})

    def test_confidence_is_never_coerced_from_boolean_or_text(self):
        for confidence in (True, "0.9", -0.1, 1.1):
            result, _, _ = extract({"facts": [{"claim_text": "count", "confidence": confidence}]})
            self.assertEqual(result.claims, ())
            self.assertEqual(len(result.rejected_items), 1)

    def test_valid_item_confidence_is_preserved(self):
        result, _, _ = extract({"facts": [{"claim_text": "24 sensors", "confidence": 0.7}]})
        self.assertEqual(result.claims[0].confidence, 0.7)

    def test_per_claim_counterevidence_is_checked_and_preserved(self):
        result, _, _ = extract(
            {
                "facts": [
                    {"claim_text": "24 sensors", "counterevidence": [{"source_id": "pilot-v1"}]},
                    {"claim_text": "lab result", "counterevidence": [{"source_id": "invented"}]},
                ],
                "source_ids": ["pilot-v1", "invented"],
            }
        )
        self.assertEqual(result.claims[0].counterevidence[0].source_id, "pilot-v1")
        self.assertEqual(
            [item.reason.value for item in result.rejected_items], ["PROVENANCE_VIOLATION"]
        )

    def test_valid_counterevidence_is_checked_even_without_a_positive_claim(self):
        result, _, _ = extract(
            {
                "unknowns": ["unresolved"],
                "contradicting_evidence": [{"source_id": "pilot-v1"}],
                "source_ids": ["pilot-v1"],
            }
        )
        self.assertEqual(result.claims, ())
        self.assertEqual(result.contradicting_evidence[0].source_id, "pilot-v1")

    def test_unknown_claim_relationships_remain_unverified_pointers(self):
        result, _, _ = extract(
            {"inferences": [{"claim_text": "disagrees", "contradicting_claim_ids": ["not-loaded"]}]}
        )
        self.assertEqual(result.claims[0].contradicting_claim_ids, ("not-loaded",))
