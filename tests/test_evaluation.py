"""The scorer implements rubric v1 exactly, and its harness passes known-good and known-bad runs."""

import itertools
import unittest
from copy import deepcopy
from pathlib import Path

from auditable_ai.dataset import DATASETS, load_dataset
from auditable_ai.evaluation import (
    build_report,
    format_report,
    label_claim,
    markdown_summary,
    offline_entries,
    record_entries,
    score_document,
    settings_of,
)
from auditable_ai.runs import run_model_extraction
from tests.provider_fixtures import (
    ScriptedTransport,
    call_center_output,
    fixed_clock,
    message_body,
    response,
)

DOCS = Path(__file__).resolve().parent.parent / "docs" / "evaluation.md"


def fact(document_id, text, claim_type, *chunks, counter=()):
    def cite(sentence):
        return {"source_id": document_id, "source_chunk_id": f"{document_id}:s{sentence}"}

    return {
        "category": "FACT",
        "claim_text": text,
        "claim_type": claim_type,
        "supporting_evidence": [cite(sentence) for sentence in chunks],
        "counterevidence": [cite(sentence) for sentence in counter],
    }


def result(*claims, unknowns=()):
    return {"claims": list(claims), "rejected_items": [], "unknowns": list(unknowns)}


def recorded_run():
    return run_model_extraction(
        load_dataset().document("eval-call-center-v1"),
        transport=ScriptedTransport(response(message_body(call_center_output()))),
        clock=fixed_clock,
    )


class Harness(unittest.TestCase):
    def test_gold_answers_score_fully_and_an_empty_output_scores_nothing(self):
        for name, split in itertools.product(DATASETS, ("dev", "eval")):
            dataset = load_dataset(name)
            with self.subTest(dataset=name, split=split):
                documents = dataset.split(split)
                oracle = build_report(
                    {"name": "oracle", "origin": "harness_check"},
                    offline_entries("oracle", documents),
                    dataset,
                    split,
                )["summary"]
                empty = build_report(
                    {"name": "null", "origin": "harness_check"},
                    offline_entries("null", documents),
                    dataset,
                    split,
                )["summary"]
                for measure in ("figure_coverage", "claim_support", "unknown_coverage"):
                    self.assertEqual(oracle[measure]["rate"], 1.0)
                self.assertEqual(oracle["description_check"]["rate"], 1.0)
                self.assertEqual(empty["figure_coverage"]["numerator"], 0)
                self.assertEqual(empty["claim_support"]["denominator"], 0)

    def test_a_report_names_the_dataset_it_scored(self):
        dataset = load_dataset("figure-extraction-v1-hard")
        documents = dataset.split("eval")
        report = build_report(
            {"name": "null", "origin": "harness_check"},
            offline_entries("null", documents),
            dataset,
            "eval",
        )
        self.assertEqual(report["dataset"], "figure-extraction-v1-hard")
        self.assertTrue(
            format_report(report).startswith(
                "figure-extraction-v1-hard | split=eval (16 documents)"
            )
        )


class LabelRules(unittest.TestCase):
    def setUp(self):
        self.dataset = load_dataset()

    def label(self, document_id, text, claim_type, *chunks, covered=None):
        document = self.dataset.document(document_id)
        claim = fact(document_id, text, claim_type, *chunks)
        return label_claim(document, claim, set() if covered is None else covered)

    def test_each_rule_assigns_the_documented_label(self):
        water, food, data = "eval-water-plant-v2", "eval-food-bank-v1", "eval-data-center-v1"
        cases = (
            (
                "SUPPORTED",
                water,
                "The plant treated 4.2 million gallons a day.",
                "current_figure",
                2,
            ),
            (
                "WRONG_STATUS",
                water,
                "The plant treated 4.8 million gallons a day.",
                "current_figure",
                2,
            ),
            ("WRONG_STATUS", water, "Capacity will rise by 15 percent.", "current_figure", 5),
            (
                "UNSUPPORTED_VALUE",
                water,
                "The plant treated 5.1 million gallons.",
                "current_figure",
                2,
            ),
            (
                "CITATION_MISMATCH",
                water,
                "Turbidity exceeded the limit on 2 days.",
                "current_figure",
                2,
            ),
            ("OUT_OF_SCOPE", water, "The summary was revised on 8 July.", "current_figure", 1),
            ("UNRESOLVED", water, "Turbidity exceeded the internal limit.", "current_figure", 3),
            ("TRAP", data, "500 servers failed.", "current_figure", 5),
            ("UNRESOLVED", food, "75 boxes were damaged in transit.", "current_figure", 6),
        )
        for expected, document_id, text, claim_type, chunk in cases:
            with self.subTest(text=text):
                self.assertEqual(
                    self.label(document_id, text, claim_type, chunk)["label"], expected
                )

    def test_one_claim_can_cover_several_figures_and_repeats_are_duplicates(self):
        covered = set()
        text = "Crews installed panels on 18 of the 25 schools."
        first = self.label("eval-solar-install-v1", text, "current_figure", 2, covered=covered)
        again = self.label("eval-solar-install-v1", text, "current_figure", 2, covered=covered)
        self.assertEqual((first["label"], first["figures"]), ("SUPPORTED", ["F1", "F2"]))
        self.assertEqual(again["label"], "DUPLICATE")
        self.assertTrue(first["description_confirmed"])

    def test_a_correction_restated_in_one_claim_credits_only_the_figure_its_type_names(self):
        text = "Average daily volume was revised to 4.2 million gallons from 4.8 million."
        labeled = self.label("eval-water-plant-v2", text, "current_figure", 2)
        self.assertEqual((labeled["label"], labeled["figures"]), ("SUPPORTED", ["F1"]))

    def test_omissions_record_why_a_figure_was_not_covered(self):
        document_id = "eval-water-plant-v2"
        score = score_document(
            self.dataset.document(document_id),
            result(
                fact(document_id, "4.8 million gallons a day.", "current_figure", 2),
                fact(document_id, "Turbidity exceeded the limit on 2 days.", "current_figure", 2),
            ),
        )
        omissions = {figure["id"]: figure["omission"] for figure in score["figures"]}
        self.assertEqual(
            omissions,
            {"F1": "not_found", "F2": "wrong_status", "F3": "citation_mismatch", "F4": "not_found"},
        )

    def test_unknowns_match_keyword_groups_and_unmatched_outputs_are_listed(self):
        score = score_document(
            self.dataset.document("eval-water-plant-v2"),
            result(unknowns=["Chlorine-residual readings (north reservoir)", "Commissioning date"]),
        )
        self.assertEqual(score["unknowns"]["matched"], ["U1"])
        self.assertEqual(score["unknowns"]["unmatched_output"], ["Commissioning date"])

    def test_disputed_figures_are_checked_for_competing_value_counterevidence(self):
        document_id = "eval-call-center-v1"
        text = "The telephony system reports a 6 percent abandonment rate."
        with_counter = fact(document_id, text, "disputed_figure", 3, counter=(3,))
        without = fact(document_id, text, "disputed_figure", 3)
        document = self.dataset.document(document_id)
        self.assertEqual(
            score_document(document, result(with_counter))["disputed"]["with_counterevidence"], 1
        )
        self.assertEqual(
            score_document(document, result(without))["disputed"]["with_counterevidence"], 0
        )


class RecordedRuns(unittest.TestCase):
    def test_failures_and_served_model_mismatches_are_listed_but_never_scored(self):
        dataset = load_dataset()
        document = dataset.document("eval-call-center-v1")
        outcomes = (
            response(message_body(call_center_output())),
            response(message_body(text="not json")),
            response(message_body(call_center_output(), model="claude-opus-4-8")),
        )
        records = [
            run_model_extraction(
                document, transport=ScriptedTransport(outcome), clock=fixed_clock, run_id=f"run-{n}"
            )
            for n, outcome in enumerate(outcomes)
        ]
        report = build_report(
            {"name": "claude", "origin": "replayed_recording", "settings": settings_of(records)},
            record_entries(records, dataset),
            dataset,
            "eval",
        )
        self.assertEqual(report["runs"]["scored"], 1)
        self.assertEqual(
            report["runs"]["failed"], {"malformed_output": 1, "served_model_mismatch": 1}
        )
        self.assertEqual(report["response_origins"], {"scripted_fixture": 3})
        self.assertTrue(
            any("did not come from a real API call" in note for note in report["caveats"])
        )
        self.assertEqual(len(report["runs"]["documents_without_a_scored_run"]), 7)

    def test_records_with_different_settings_are_evaluated_separately(self):
        document = load_dataset().document("eval-call-center-v1")
        body = response(message_body(call_center_output()))
        first = run_model_extraction(document, transport=ScriptedTransport(body), clock=fixed_clock)
        second = {**first, "settings": {**first["settings"], "effort": "high"}}
        with self.assertRaises(ValueError):
            settings_of([first, second])

    def test_a_tampered_record_is_reported_as_an_integrity_failure_not_a_crash(self):
        dataset = load_dataset()
        document = dataset.document("eval-call-center-v1")
        record = run_model_extraction(
            document,
            transport=ScriptedTransport(response(message_body(call_center_output()))),
            clock=fixed_clock,
        )
        exchange = record["exchange"]
        tampered = {
            **record,
            "exchange": {**exchange, "raw_response": exchange["raw_response"].replace("6", "7")},
        }
        entries = record_entries([tampered], dataset)
        self.assertEqual(entries[0]["status"], "integrity_error")
        self.assertIn("hash", entries[0]["diagnostics"]["message"])

    def test_a_completed_record_without_its_response_is_an_integrity_failure(self):
        record = recorded_run()
        for exchange in (None, {**record["exchange"], "raw_response": None}):
            with self.subTest(exchange=exchange is None):
                [entry] = record_entries([{**record, "exchange": exchange}], load_dataset())
                self.assertEqual(entry["status"], "integrity_error")
                self.assertIn("holds no provider response", entry["diagnostics"]["message"])

    def test_a_stored_outcome_that_contradicts_its_response_is_an_integrity_failure(self):
        record = recorded_run()
        result = record["result"]
        for altered in (
            {**record, "status": "refused"},
            {**record, "result": {**result, "claims": result["claims"][:1]}},
            {**record, "result": {**result, "confidence": 0.9}},
            {**record, "served_model_matches_request": False},
        ):
            with self.subTest():
                [entry] = record_entries([altered], load_dataset())
                self.assertEqual(entry["status"], "integrity_error")
                self.assertIsNone(entry["result"])

    def test_an_older_validators_outcome_may_differ_and_the_replay_is_scored(self):
        record = recorded_run()
        older = {**record, "validator_version": "1", "result": {**record["result"], "claims": []}}
        [entry] = record_entries([older], load_dataset())
        self.assertEqual(entry["status"], "completed")
        self.assertFalse(entry["replay_matches_record"])
        self.assertEqual(len(entry["result"]["claims"]), 2)

    def test_tokens_cost_and_served_model_come_from_the_hash_checked_response(self):
        record = recorded_run()
        edited = deepcopy(record)
        edited["response"]["usage"]["input_tokens"] = 1
        edited["response"]["served_model"] = "claude-opus-4-8"
        edited["cost"] = {"status": "estimated", "usd": 99.0}
        [original], [altered] = (record_entries([run], load_dataset()) for run in (record, edited))
        for key in ("usage", "cost", "served_model"):
            self.assertEqual(altered[key], original[key])
        self.assertEqual(
            (original["usage"]["input_tokens"], original["cost"]["usd"]), (900, 0.0096)
        )

    def test_a_malformed_record_is_refused_rather_than_scored(self):
        record = recorded_run()
        malformed = {**record, "settings": {**record["settings"], "model": None}}
        with self.assertRaisesRegex(ValueError, "malformed run record: settings.model"):
            record_entries([malformed], load_dataset())

    def test_token_totals_count_only_well_formed_usage(self):
        dataset = load_dataset()
        document = dataset.document("eval-call-center-v1")
        bodies = [message_body(call_center_output())]
        for count in ("[null]", "-5"):
            bodies.append(bodies[0].replace('"input_tokens": 900', f'"input_tokens": {count}'))
        records = [
            run_model_extraction(
                document,
                transport=ScriptedTransport(response(body)),
                clock=fixed_clock,
                run_id=f"run-{number}",
            )
            for number, body in enumerate(bodies)
        ]
        report = build_report(
            {"name": "claude", "origin": "replayed_recording"},
            record_entries(records, dataset),
            dataset,
            "eval",
        )
        self.assertEqual(report["runs"]["scored"], 3)
        performance = report["performance"]
        self.assertEqual((performance["tokens"]["runs"], performance["tokens"]["input"]), (1, 900))
        self.assertEqual(
            (performance["cost"]["runs_estimated"], performance["cost"]["runs_unavailable"]),
            (1, 2),
        )


class PublishedResults(unittest.TestCase):
    def test_the_baseline_results_in_the_documentation_are_current(self):
        dataset = load_dataset()
        report = build_report(
            {"name": "rule_baseline", "origin": "rule_based"},
            offline_entries("rule_baseline", dataset.split("eval")),
            dataset,
            "eval",
        )
        self.assertIn(markdown_summary(report), DOCS.read_text(encoding="utf-8"))
        self.assertEqual(report["annotation_provenance"]["review_status"], "unreviewed")
        self.assertTrue(any("not a model" in note for note in report["caveats"]))
