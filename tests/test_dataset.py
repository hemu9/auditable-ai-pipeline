"""The evaluation dataset and task definition are checked before any scoring relies on them."""

import json
import re
import unittest
from collections import Counter
from dataclasses import replace
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory

from auditable_ai.dataset import (
    DATASETS,
    Trap,
    bundled_document,
    check_dataset,
    document_to_entry,
    load_dataset,
    load_document_file,
    numbers_in,
    register,
)
from auditable_ai.figure_task import (
    CLAIM_TYPE_STATUS,
    TASK_VERSION,
    prompt_sha256,
    render_document,
    response_schema,
    template_sha256,
)

DOCS = Path(__file__).resolve().parent.parent / "docs" / "evaluation.md"


class DatasetIntegrity(unittest.TestCase):
    def setUp(self):
        self.dataset = load_dataset()

    def test_every_document_and_annotation_passes_the_integrity_checks(self):
        self.assertEqual(check_dataset(self.dataset), [])

    def test_the_checker_reports_an_unannotated_number(self):
        document = self.dataset.document("eval-call-center-v1")
        altered = replace(document, sentences=(*document.sentences, "Staff took 12 breaks."))
        problems = check_dataset(replace(self.dataset, documents=(altered,)))
        self.assertEqual(problems, ["eval-call-center-v1: number 12 is not annotated"])

    def test_the_checker_reports_broken_annotations(self):
        document = self.dataset.document("eval-call-center-v1")
        first, second, third, fourth = document.figures
        broken = replace(
            document,
            figures=(
                replace(first, chunks=(document.chunk_id(3),)),
                replace(second, status="estimated"),
                replace(third, counter_chunks=()),
                replace(fourth, chunks=("eval-call-center-v1:s9",)),
            ),
            unknowns=(replace(document.unknowns[0], match_any=(("weather",),)),),
            traps=(Trap(value=77.0, chunks=(document.chunk_id(1),), reason="seeded error"),),
        )
        problems = "\n".join(check_dataset(replace(self.dataset, documents=(broken,))))
        for expected in (
            "F1: value is not stated in eval-call-center-v1:s3",
            "F2: unknown status 'estimated'",
            "F3: a disputed figure needs counter_chunks",
            "F4: unknown chunk eval-call-center-v1:s9",
            "U1: no keyword group appears in its own chunks",
            "trap 77 is not in its chunks",
        ):
            self.assertIn(expected, problems)

    def test_development_and_evaluation_documents_are_disjoint(self):
        dev = {document.document_version_id for document in self.dataset.split("dev")}
        held_out = {document.document_version_id for document in self.dataset.split("eval")}
        self.assertEqual((len(dev), len(held_out)), (4, 8))
        self.assertFalse(dev & held_out)
        with self.assertRaises(ValueError):
            self.dataset.split("test")

    def test_held_out_split_covers_each_phenomenon_and_a_negative_control(self):
        documents = self.dataset.split("eval")
        covered = {phenomenon for document in documents for phenomenon in document.phenomena}
        required = {"correction", "conflict", "missing_information", "unsupported_assertion"}
        self.assertLessEqual(required, covered)
        self.assertTrue(
            any(not {"correction", "conflict"} & set(document.phenomena) for document in documents)
        )

    def test_provenance_says_the_annotations_are_ai_written_and_unreviewed(self):
        provenance = self.dataset.provenance
        self.assertEqual(provenance["review_status"], "unreviewed")
        self.assertEqual(provenance["reviewers"], [])
        self.assertIn("AI-generated", provenance["annotations"])
        self.assertIn("claude-opus-5-5", provenance["conflict_of_interest"])

    def test_numbers_are_parsed_with_separators_and_decimals_but_not_from_identifiers(self):
        text = "Stock is 1,240 cases and 4.2 million gallons, chunk doc-v2:s3, 15 percent."
        self.assertEqual(numbers_in(text), (1240.0, 4.2, 15.0))

    def test_registration_stores_each_sentence_as_a_citable_chunk(self):
        document = self.dataset.document("eval-bridge-inspection-v2")
        source, claims = register(document)
        self.assertEqual(source.raw_content.text, document.text)
        for chunk_id, text in document.chunk_texts.items():
            chunk = claims.chunks.get(chunk_id)
            self.assertEqual(
                (chunk.document_version_id, chunk.raw_text.text), (source.document_version_id, text)
            )

    def test_a_document_file_without_gold_annotations_can_be_loaded(self):
        document = self.dataset.document("eval-solar-install-v1")
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "document.json"
            path.write_text(json.dumps(document_to_entry(document)), encoding="utf-8")
            loaded = load_document_file(path)
        self.assertEqual(loaded.text, document.text)
        self.assertEqual((loaded.figures, loaded.unknowns), ((), ()))


class BundledDatasets(unittest.TestCase):
    def test_every_bundled_dataset_passes_the_integrity_checks(self):
        for name in DATASETS:
            with self.subTest(dataset=name):
                self.assertEqual(check_dataset(load_dataset(name)), [])

    def test_each_dataset_names_itself_and_the_task_it_belongs_to(self):
        for name in DATASETS:
            dataset = load_dataset(name)
            self.assertEqual((dataset.name, dataset.task_version), (name, TASK_VERSION))

    def test_document_ids_are_unique_across_datasets(self):
        ids = [d.document_version_id for name in DATASETS for d in load_dataset(name).documents]
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_dataset_says_its_annotations_are_ai_written_and_unreviewed(self):
        for name in DATASETS:
            provenance = load_dataset(name).provenance
            with self.subTest(dataset=name):
                self.assertEqual(
                    (provenance["review_status"], provenance["reviewers"]), ("unreviewed", [])
                )
                self.assertIn("AI-generated", provenance["annotations"])
                self.assertIn("claude-opus-5-5", provenance["conflict_of_interest"])

    def test_an_unknown_dataset_is_refused(self):
        with self.assertRaisesRegex(ValueError, "unknown dataset"):
            load_dataset("figure-extraction-v9")

    def test_a_bundled_document_is_found_in_whichever_dataset_holds_it(self):
        self.assertEqual(bundled_document("hard-eval-ferry-service-v1").split, "eval")
        self.assertEqual(bundled_document("eval-call-center-v1").source_id, "utility-call-center")
        with self.assertRaises(KeyError):
            bundled_document("eval-unknown-v1")


class HardDataset(unittest.TestCase):
    NAME = "figure-extraction-v1-hard"

    def setUp(self):
        self.dataset = load_dataset(self.NAME)
        self.raw = json.loads(files("auditable_ai").joinpath(DATASETS[self.NAME]).read_text())

    def test_development_and_evaluation_documents_are_disjoint(self):
        dev = {document.document_version_id for document in self.dataset.split("dev")}
        held_out = {document.document_version_id for document in self.dataset.split("eval")}
        self.assertEqual((len(dev), len(held_out)), (4, 16))
        self.assertFalse(dev & held_out)

    def test_every_phenomenon_tag_is_defined_and_used(self):
        used = {tag for document in self.raw["documents"] for tag in document["phenomena"]}
        self.assertEqual(used, set(self.raw["phenomena"]))

    def test_the_documented_phenomenon_counts_are_current(self):
        counts = Counter(tag for d in self.dataset.split("eval") for tag in d.phenomena)
        rows = re.findall(
            r"^\| `(\w+)` \| [^|]+ \| (\d+) \|$", DOCS.read_text(encoding="utf-8"), re.M
        )
        self.assertEqual({tag: int(count) for tag, count in rows}, dict(counts))

    def test_the_construction_note_discloses_the_adversarial_design(self):
        construction = self.dataset.provenance["construction"]
        self.assertIn("adversarial to cue-word rules by design", construction)
        self.assertIn("not run on these documents until the dataset was frozen", construction)


class TaskDefinition(unittest.TestCase):
    def test_the_prompt_lists_every_chunk_under_the_source_id(self):
        document = load_dataset().document("eval-data-center-v1")
        rendered = render_document(document)
        self.assertTrue(rendered.startswith("Source ID: eval-data-center-v1\n"))
        for chunk_id, text in document.chunk_texts.items():
            self.assertIn(f"[{chunk_id}] {text}", rendered)

    def test_the_schema_closes_every_object_and_requires_every_property(self):
        def objects(node):
            if isinstance(node, dict):
                if node.get("type") == "object":
                    yield node
                for value in node.values():
                    yield from objects(value)
            elif isinstance(node, list):
                for value in node:
                    yield from objects(value)

        schema = response_schema()
        found = list(objects(schema))
        self.assertEqual(len({id(node) for node in found}), 3)
        for node in found:
            self.assertIs(node["additionalProperties"], False)
            self.assertEqual(sorted(node["required"]), sorted(node["properties"]))
        claim_types = schema["properties"]["facts"]["items"]["properties"]["claim_type"]["enum"]
        self.assertEqual(claim_types, list(CLAIM_TYPE_STATUS))

    def test_fingerprints_identify_the_template_and_each_document_prompt(self):
        dataset = load_dataset()
        first, second = dataset.split("eval")[:2]
        self.assertNotEqual(prompt_sha256(first), prompt_sha256(second))
        self.assertEqual(prompt_sha256(first), prompt_sha256(first))
        self.assertEqual(len(template_sha256()), 64)
