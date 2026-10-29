"""The README walkthrough is the real example file, and its output is the real output."""

import importlib.util
import runpy
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from auditable_ai.dataset import load_document_file, register

ROOT = Path(__file__).resolve().parent.parent


def mutation_catalog():
    spec = importlib.util.spec_from_file_location(
        "mutation_check", ROOT / "tools" / "mutation_check.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.MUTATIONS


class ReadmeWalkthrough(unittest.TestCase):
    def test_the_readme_shows_the_complete_example_and_its_actual_output(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        script = (ROOT / "examples" / "quickstart.py").read_text(encoding="utf-8")
        self.assertIn(f"```python\n{script}```", readme)
        output = StringIO()
        with redirect_stdout(output):
            runpy.run_path(str(ROOT / "examples" / "quickstart.py"), run_name="__main__")
        self.assertIn(f"```text\n{output.getvalue()}```", readme)

    def test_the_example_document_can_be_registered_for_a_model_run(self):
        document = load_document_file(ROOT / "examples" / "document.json")
        source, claims = register(document)
        self.assertEqual(source.document_version_id, "ferry-terminal-v1")
        self.assertEqual(len(claims.chunks.for_document_version("ferry-terminal-v1")), 5)

    def test_the_documented_number_of_mutations_is_current(self):
        count = len(mutation_catalog())
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        testing = (ROOT / "docs" / "testing.md").read_text(encoding="utf-8")
        self.assertEqual(readme.count(f"{count} stated guarantees"), 2)
        self.assertIn(f"Each of {count} entries", testing)
        self.assertIn(f"against {count} hand-written mutations", testing)
