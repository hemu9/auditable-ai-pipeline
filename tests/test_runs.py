"""Run records keep what is needed to audit or replay a model call; failures stay diagnosable."""

import json
import math
import unittest
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from auditable_ai.anthropic_adapter import ClaudeSettings, ProviderCallError, sha256_text
from auditable_ai.dataset import load_dataset
from auditable_ai.errors import ProvenanceError, TemporalIntegrityError
from auditable_ai.figure_task import SYSTEM_PROMPT, prompt_sha256
from auditable_ai.runs import read_record, replay_record, run_model_extraction, write_record
from tests.provider_fixtures import (
    ScriptedTransport,
    call_center_output,
    fixed_clock,
    message_body,
    response,
)


def run_with(*outcomes, settings=ClaudeSettings()):
    return run_model_extraction(
        load_dataset().document("eval-call-center-v1"),
        settings,
        transport=ScriptedTransport(*outcomes),
        clock=fixed_clock,
        sleep=lambda _seconds: None,
    )


class RunRecords(unittest.TestCase):
    def test_a_completed_run_keeps_the_raw_response_result_and_measurements(self):
        body = message_body(call_center_output())
        record = run_with(response(body))
        self.assertEqual((record["origin"], record["status"]), ("scripted_fixture", "completed"))
        exchange = record["exchange"]
        self.assertEqual(exchange["raw_response"], body)
        self.assertEqual(exchange["response_sha256"], sha256_text(body))
        self.assertEqual(exchange["request_id"], "req_scripted_fixture")
        self.assertEqual(exchange["request"]["system"], SYSTEM_PROMPT)
        self.assertIsNotNone(exchange["latency_ms"])
        result = record["result"]
        self.assertEqual(len(result["claims"]), 2)
        self.assertEqual(
            [item["reason"] for item in result["rejected_items"]], ["PROVENANCE_VIOLATION"]
        )
        self.assertEqual(result["unknowns"], ["Customer satisfaction scores for the week"])
        counterevidence = result["claims"][1]["counterevidence"]
        self.assertEqual(counterevidence[0]["source_chunk_id"], "eval-call-center-v1:s3")
        self.assertEqual(record["response"]["served_model"], "claude-opus-5-5")
        self.assertTrue(record["served_model_matches_request"])
        self.assertEqual((record["cost"]["status"], record["cost"]["usd"]), ("estimated", 0.0096))
        self.assertEqual(record["settings"]["effort"], "medium")
        document = load_dataset().document("eval-call-center-v1")
        self.assertEqual(record["task"]["prompt_sha256"], prompt_sha256(document))
        self.assertEqual(record["timestamps"]["extracted_at"], fixed_clock().isoformat())

    def test_each_failure_is_recorded_with_its_class_and_diagnostics(self):
        invented = {"source_id": "invented", "source_chunk_id": None, "locator_note": None}
        cases = {
            "refused": response(message_body(stop_reason="refusal")),
            "truncated": response(message_body(text='{"facts": [', stop_reason="max_tokens")),
            "malformed_output": response(message_body(text="The report mentions 9,320 calls.")),
            "validation_error": response(message_body({"facts": [], "execute": "x"})),
            "provenance_error": response(
                message_body({"contradicting_evidence": [invented], "source_ids": ["invented"]})
            ),
            "provider_error": ProviderCallError(
                "bad request", error_type="BadRequestError", status_code=400
            ),
        }
        for status, outcome in cases.items():
            with self.subTest(status=status):
                record = run_with(outcome)
                self.assertEqual(record["status"], status)
                self.assertIsNone(record["result"])
                self.assertTrue(record["diagnostics"]["message"])
                if status == "provider_error":
                    self.assertIsNone(record["exchange"]["raw_response"])
                    self.assertEqual(record["cost"]["status"], "unavailable")
                else:
                    self.assertEqual(record["exchange"]["raw_response"], outcome.text)

    def test_a_response_from_another_model_is_flagged_and_priced_as_that_model(self):
        record = run_with(response(message_body(call_center_output(), model="claude-opus-4-8")))
        self.assertFalse(record["served_model_matches_request"])
        self.assertEqual(set(record["cost"]["basis"]["prices"]), {"claude-opus-4-8"})

    def test_records_are_written_once_and_never_overwritten(self):
        record = run_with(response(message_body(call_center_output())))
        with TemporaryDirectory() as tmp:
            path = write_record(record, Path(tmp) / "runs", clock=fixed_clock)
            original = path.read_bytes()
            self.assertEqual(read_record(path)["run_id"], record["run_id"])
            with self.assertRaises(FileExistsError):
                write_record(record, Path(tmp) / "runs", clock=fixed_clock)
            self.assertEqual(path.read_bytes(), original)

    def test_a_response_with_malformed_metadata_is_still_recorded_and_written(self):
        body = json.dumps(
            {
                "type": "message",
                "model": ["claude-opus-5-5"],
                "content": [
                    {"type": "fallback", "from": 1},
                    {"type": "text", "text": json.dumps(call_center_output())},
                ],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": [None], "output_tokens": 0},
            }
        )
        record = run_with(response(body))
        self.assertEqual(record["status"], "completed")
        self.assertFalse(record["served_model_matches_request"])
        self.assertEqual(record["cost"]["status"], "unavailable")
        with TemporaryDirectory() as tmp:
            written = read_record(write_record(record, Path(tmp), clock=fixed_clock))
        self.assertEqual(written["exchange"]["raw_response"], body)

    def test_a_response_with_non_standard_json_is_classified_and_recorded(self):
        body = message_body(call_center_output()).replace(
            '"input_tokens": 900', '"input_tokens": NaN'
        )
        record = run_with(response(body))
        self.assertEqual(
            (record["status"], record["diagnostics"]["kind"]),
            ("malformed_output", "unexpected_response"),
        )
        with TemporaryDirectory() as tmp:
            written = read_record(write_record(record, Path(tmp), clock=fixed_clock))
        self.assertEqual(written["exchange"]["raw_response"], body)

    def test_a_record_that_cannot_be_serialized_leaves_no_file(self):
        record = run_with(response(message_body(call_center_output())))
        record["exchange"]["latency_ms"] = float("nan")
        with TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                write_record(record, Path(tmp), clock=fixed_clock)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_a_clock_without_a_utc_offset_is_refused_before_any_call(self):
        transport = ScriptedTransport(response(message_body(call_center_output())))
        with self.assertRaises(TemporalIntegrityError):
            run_model_extraction(
                load_dataset().document("eval-call-center-v1"),
                transport=transport,
                clock=lambda: datetime(2026, 10, 28, 12, 0),
            )
        self.assertEqual(transport.requests, [])

    def test_reading_refuses_a_malformed_record_and_says_why(self):
        record = run_with(response(message_body(call_center_output())))
        exchange, timestamps = record["exchange"], record["timestamps"]
        cases = [
            ("not a model_extraction_run record", [record]),
            ("not a model_extraction_run record", {**record, "record_type": "report"}),
            ("unsupported record_version 2", {**record, "record_version": 2}),
            ("run_id must be a non-empty string", {**record, "run_id": ""}),
            (
                "served_model_matches_request is missing",
                {k: v for k, v in record.items() if k != "served_model_matches_request"},
            ),
            (
                "exchange.latency_ms must be a non-negative number or null",
                {**record, "exchange": {**exchange, "latency_ms": "fast"}},
            ),
            (
                "timestamps.extracted_at must be a timestamp with a UTC offset",
                {**record, "timestamps": {**timestamps, "extracted_at": "2026-10-28T12:00:00"}},
            ),
            ("NaN is not valid JSON", {**record, "exchange": {**exchange, "latency_ms": math.nan}}),
        ]
        with TemporaryDirectory() as tmp:
            for number, (reason, value) in enumerate(cases):
                path = Path(tmp, f"record-{number}.json")
                path.write_text(json.dumps(value), encoding="utf-8")
                with self.subTest(reason=reason, case=number):
                    with self.assertRaisesRegex(ValueError, reason) as caught:
                        read_record(path)
                    self.assertIn(str(path), str(caught.exception))


class Replay(unittest.TestCase):
    def test_replay_reproduces_the_recorded_outcome_without_a_model_call(self):
        record = run_with(response(message_body(call_center_output())))
        replay = replay_record(record, clock=fixed_clock)
        self.assertEqual(
            (replay["origin"], replay["model_calls"], replay["status"]),
            ("replayed_recording", 0, "completed"),
        )
        self.assertTrue(replay["matches_recorded_outcome"])
        self.assertEqual(replay["result"]["unknowns"], record["result"]["unknowns"])
        self.assertEqual(replay["replayed_from"]["run_id"], record["run_id"])

    def test_replay_reproduces_the_diagnostics_of_a_malformed_response(self):
        record = run_with(response(message_body(text="not json")))
        replay = replay_record(record)
        self.assertEqual(
            (replay["status"], replay["diagnostics"]["kind"]), ("malformed_output", "invalid_json")
        )
        self.assertTrue(replay["matches_recorded_outcome"])

    def test_replay_reports_where_the_current_validator_disagrees_with_the_record(self):
        record = run_with(response(message_body(call_center_output())))
        altered = deepcopy(record)
        altered["result"]["claims"].pop()
        replay = replay_record(altered)
        self.assertFalse(replay["matches_recorded_outcome"])
        self.assertEqual(replay["differences"], ["claims"])

    def test_replay_refuses_a_tampered_response_or_document(self):
        record = run_with(response(message_body(call_center_output())))
        tampered_response = deepcopy(record)
        exchange = tampered_response["exchange"]
        exchange["raw_response"] = exchange["raw_response"].replace("9,320", "9,999")
        tampered_document = deepcopy(record)
        tampered_document["document"]["sentences"][1] = "Agents answered 9,999 calls."
        for tampered in (tampered_response, tampered_document):
            with self.subTest(), self.assertRaises(ProvenanceError):
                replay_record(tampered)

    def test_a_run_that_received_no_response_is_reported_as_not_replayable(self):
        record = run_with(
            ProviderCallError("rejected", error_type="BadRequestError", status_code=400)
        )
        replay = replay_record(record)
        self.assertEqual(replay["status"], "not_replayable")
        self.assertEqual(replay["recorded_diagnostics"]["status_code"], 400)

    def test_replay_runs_under_the_recorded_run_id_and_reproduces_the_stored_result(self):
        record = run_with(response(message_body(call_center_output())))
        replay = replay_record(record)
        self.assertEqual(replay["differences"], [])
        for stored, replayed in zip(record["result"]["claims"], replay["result"]["claims"]):
            self.assertEqual(replayed, stored)
        relabeled = {**record, "status": "refused"}
        self.assertEqual(replay_record(relabeled)["differences"], ["status"])

    def test_replay_refuses_a_malformed_record(self):
        record = run_with(response(message_body(call_center_output())))
        with self.assertRaisesRegex(ValueError, "malformed run record: timestamps"):
            replay_record({**record, "timestamps": None})
