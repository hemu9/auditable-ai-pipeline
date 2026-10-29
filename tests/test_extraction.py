"""Trust-boundary tests include explicit demonstrations of semantic limits."""

import unittest
from dataclasses import replace
from math import inf, nan

from auditable_ai import (
    ClaimCategory,
    EvidenceExtractor,
    ExtractionValidationError,
    ProvenanceError,
    RawDocumentContent,
    SourceChunk,
    TemporalIntegrityError,
)
from tests.helpers import at, extract, setup, valid_response


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
        self.assertEqual(result.to_dict()["accuracy_status"], "NOT_VERIFIED")

    def test_fact_evidence_is_normalized_before_it_is_validated(self):
        result, _, claims = extract({"facts": [{"claim_text": "24 sensors"}]}, run_id="seed")
        attributed = result.claims[0]
        cases = (
            iter(()),
            (reference for reference in ()),
            [],
            "pilot-v2",
            [{"source_id": "pilot-v2"}],
        )
        for number, evidence in enumerate(cases):
            claim_id = f"unattributed-{number}"
            with self.subTest(evidence=repr(evidence)):
                with self.assertRaises(ValueError):
                    claims.add(replace(attributed, claim_id=claim_id, supporting_evidence=evidence))
                self.assertIsNone(claims.get(claim_id))
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
        unattributed = replace(result.claims[0], claim_id="bypassed")
        object.__setattr__(unattributed, "supporting_evidence", ())
        with self.assertRaises(ProvenanceError):
            claims.add(unattributed)
        self.assertIsNone(claims.get("bypassed"))

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

    def test_well_formed_false_claim_passes_with_accuracy_unverified(self):
        bundle = setup()
        result, _, _ = extract(bundle[0]["cases"][4]["response"], bundle=bundle)
        self.assertIn("900", result.claims[0].claim_text)
        self.assertEqual(result.to_dict()["accuracy_status"], "NOT_VERIFIED")

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
        raw = RawDocumentContent("count")
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

    def test_duplicate_run_cannot_overwrite_a_claim(self):
        bundle = setup()
        first, _, claims = extract(valid_response(), bundle=bundle)
        second, _, _ = extract(valid_response(), bundle=bundle)
        self.assertEqual(second.claims, ())
        self.assertEqual(claims.get("test-run-claim-1"), first.claims[0])

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

    def test_adapter_mutation_after_return_cannot_rewrite_the_audit_snapshot(self):
        response = valid_response()
        result, _, _ = extract(response)
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

    def test_confidence_is_never_coerced_from_boolean_or_text(self):
        for confidence in (True, "0.9", -0.1, 1.1):
            result, _, _ = extract({"facts": [{"claim_text": "count", "confidence": confidence}]})
            self.assertEqual(result.claims, ())
            self.assertEqual(len(result.rejected_items), 1)

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
