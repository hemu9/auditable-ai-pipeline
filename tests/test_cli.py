import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from auditable_ai.cli import main
from auditable_ai.dataset import load_dataset
from auditable_ai.runs import run_model_extraction, write_record
from auditable_ai.scenarios import check_cases, fixture_data
from tests.provider_fixtures import (
    ScriptedTransport,
    call_center_output,
    fixed_clock,
    message_body,
    response,
)


class ModelRunCommands(unittest.TestCase):
    def test_extract_refuses_to_call_the_api_without_live(self):
        errors = StringIO()
        with redirect_stderr(errors), mock.patch("auditable_ai.cli.AnthropicTransport") as sdk:
            self.assertEqual(main(["extract", "--dataset-document", "eval-call-center-v1"]), 2)
        sdk.assert_not_called()
        self.assertIn("--live", errors.getvalue())

    def test_extract_reports_a_missing_sdk_or_credentials_before_sending_anything(self):
        errors = StringIO()
        missing = RuntimeError("the Claude adapter needs the optional SDK")
        with (
            redirect_stderr(errors),
            mock.patch("auditable_ai.cli.AnthropicTransport", side_effect=missing),
        ):
            code = main(["extract", "--dataset-document", "eval-call-center-v1", "--live"])
        self.assertEqual(code, 2)
        self.assertIn("Cannot create the Anthropic client", errors.getvalue())

    def test_extract_records_a_run_labeled_with_its_origin(self):
        transport = ScriptedTransport(response(message_body(call_center_output())))
        output = StringIO()
        with TemporaryDirectory() as tmp, redirect_stdout(output):
            with mock.patch("auditable_ai.cli.AnthropicTransport", return_value=transport):
                code = main(
                    ["extract", "--dataset-document", "eval-call-center-v1", "--live"]
                    + ["--record-dir", tmp]
                )
            records = list(Path(tmp).glob("*.json"))
        self.assertEqual((code, len(records)), (0, 1))
        self.assertIn("SCRIPTED FIXTURE  status=completed", output.getvalue())
        self.assertIn("REJECTED PROVENANCE_VIOLATION", output.getvalue())

    def test_replay_re_validates_a_recorded_run_without_a_model_call(self):
        record = run_model_extraction(
            load_dataset().document("eval-call-center-v1"),
            transport=ScriptedTransport(response(message_body(call_center_output()))),
            clock=fixed_clock,
        )
        output = StringIO()
        with TemporaryDirectory() as tmp, redirect_stdout(output):
            path = write_record(record, Path(tmp))
            code = main(["replay", str(path), "--output", str(Path(tmp) / "replay.json")])
            report = json.loads((Path(tmp) / "replay.json").read_text())
        self.assertEqual(code, 0)
        self.assertIn("REPLAYED RECORDING (no model call)", output.getvalue())
        self.assertIn("matches recorded outcome: yes", output.getvalue())
        self.assertEqual((report["origin"], report["model_calls"]), ("replayed_recording", 0))


class PublicWalkthrough(unittest.TestCase):
    def test_all_scripted_validator_cases_match_their_declared_behavior(self):
        checks = check_cases(fixture_data())
        self.assertEqual(len(checks), 8)
        self.assertTrue(all(check["passed"] for check in checks), checks)

    def test_demo_shows_history_rejection_and_its_accuracy_limitation(self):
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["demo"]), 0)
        text = output.getvalue()
        self.assertIn("OPERATIONAL_REPLAY_AS_OF=[]", text)
        self.assertIn("PROVENANCE_VIOLATION", text)
        self.assertIn("intentionally false", text)
        self.assertIn("COUNTEREVIDENCE: pilot-v1", text)

    def test_report_contains_raw_response_and_run_provenance(self):
        with TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            path = Path(tmp) / "audit.json"
            self.assertEqual(main(["demo", "--output", str(path)]), 0)
            report = json.loads(path.read_text())
            run = report["runs"][0]
            self.assertEqual(run["source_query_mode"], "RESEARCH_AS_OF")
            self.assertEqual(run["model_name"], "scripted-fixture")
            self.assertIn("raw_response", run)
            self.assertEqual(len(run["input_hash"]), 64)
            self.assertEqual(len(run["response_hash"]), 64)
            self.assertEqual(run["accuracy_status"], "NOT_VERIFIED")

    def test_report_path_cannot_be_overwritten(self):
        with TemporaryDirectory() as tmp, redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            path = Path(tmp) / "audit.json"
            self.assertEqual(main(["demo", "--output", str(path)]), 0)
            original = path.read_bytes()
            self.assertEqual(main(["demo", "--output", str(path)]), 2)
            self.assertEqual(path.read_bytes(), original)

    def test_fixture_command_labels_its_results_as_validator_regressions(self):
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["check-fixtures"]), 0)
        self.assertIn("8/8 validator regression scenarios", output.getvalue())
        self.assertIn("not an LLM accuracy score", output.getvalue())
