import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from auditable_ai.cli import main
from auditable_ai.scenarios import check_cases, fixture_data


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
