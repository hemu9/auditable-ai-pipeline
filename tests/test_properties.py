"""Property-based tests of the guarantees stated in the README and docs/architecture.md.

The other test modules check hand-picked examples. Here Hypothesis generates the
inputs and shrinks any counterexample to a minimal one. Inputs include malformed
adapter output, provider response bodies, retry sequences, hand-edited run records
and document version histories. The default profile is derandomized, so CI and
the mutation check see the same examples on every run. Set
HYPOTHESIS_PROFILE=explore for a longer randomized search."""

import io
import json
import os
import unittest
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from auditable_ai.anthropic_adapter import ClaudeSettings, ProviderCallError, sha256_text
from auditable_ai.claims import ClaimCategory
from auditable_ai.cli import main
from auditable_ai.dataset import load_dataset, register
from auditable_ai.documents import (
    InMemorySourceDocumentRepository,
    RawDocumentContent,
    SourceDocument,
)
from auditable_ai.errors import (
    AmbiguousRevisionError,
    ExtractionValidationError,
    ProvenanceError,
    TemporalIntegrityError,
)
from auditable_ai.evaluation import build_report, format_report, record_entries, score_document
from auditable_ai.extraction import EvidenceExtractor, RejectionReason
from auditable_ai.figure_task import CLAIM_TYPE_STATUS
from auditable_ai.runs import read_record, replay_record, run_model_extraction, write_record
from auditable_ai.storage import JsonlSourceDocumentRepository
from auditable_ai.temporal import AsOfMode, timestamp_for_mode
from tests.provider_fixtures import (
    ScriptedTransport,
    call_center_output,
    fixed_clock,
    message_body,
    response,
)

try:
    from hypothesis import HealthCheck, given, settings
    from hypothesis import strategies as st
except ImportError as exc:  # the dev extra is not installed
    raise unittest.SkipTest("property tests need Hypothesis: pip install -e '.[dev]'") from exc

settings.register_profile(
    "repeatable",
    max_examples=150,
    derandomize=True,
    database=None,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "explore", parent=settings.get_profile("repeatable"), max_examples=3000, derandomize=False
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "repeatable"))

DATASET = load_dataset()
DOCUMENT = DATASET.document("eval-call-center-v1")
VERSION = DOCUMENT.document_version_id
CHUNKS = DOCUMENT.chunk_texts
CHUNK_HASHES = {chunk: RawDocumentContent(text).content_hash() for chunk, text in CHUNKS.items()}
DOCUMENT_HASH = RawDocumentContent(DOCUMENT.text).content_hash()
RUN_ID = "property-run"
SYSTEM = {"name": "claude", "origin": "replayed_recording"}
LATER = datetime(2100, 1, 1, tzinfo=timezone.utc)
CLASSIFIED = (ExtractionValidationError, ProvenanceError, TemporalIntegrityError)
RUN_STATUSES = {
    "completed",
    "refused",
    "truncated",
    "malformed_output",
    "validation_error",
    "provenance_error",
    "provider_error",
}

# --- generated adapter output ---------------------------------------------------------

scalars = st.none() | st.booleans() | st.integers() | st.floats() | st.text(max_size=12)
json_values = st.recursive(
    scalars,
    lambda children: (
        st.lists(children, max_size=3) | st.dictionaries(st.text(max_size=6), children, max_size=3)
    ),
    max_leaves=8,
)
not_json = st.sets(st.integers(), max_size=2) | st.binary(max_size=3) | st.just(object())


def or_junk(strategy):
    """Mostly well-typed values, and sometimes any JSON value instead."""
    return st.one_of(strategy, strategy, json_values)


chunk_ids = st.sampled_from(sorted(CHUNKS))
references = st.one_of(
    chunk_ids.map(lambda c: {"source_id": VERSION, "source_chunk_id": c, "locator_note": None}),
    chunk_ids.map(
        lambda c: {
            "source_id": VERSION,
            "source_chunk_id": c,
            "source_content_hash": CHUNK_HASHES[c],
        }
    ),
    st.just({"source_id": VERSION}),
    st.just({"source_id": VERSION, "source_content_hash": DOCUMENT_HASH}),
    chunk_ids.map(
        lambda c: {"source_id": VERSION, "source_chunk_id": c, "source_content_hash": "0"}
    ),
    st.just({"source_id": VERSION, "source_chunk_id": f"{VERSION}:s99"}),
    st.just({"source_id": "undeclared-v1"}),
    chunk_ids.map(lambda c: {"source_id": VERSION, "source_chunk_id": c, "page": 2}),
    json_values,
)
items = st.fixed_dictionaries(
    {"claim_text": or_junk(st.text(min_size=1, max_size=30))},
    optional={
        "claim_type": or_junk(st.sampled_from(sorted(CLAIM_TYPE_STATUS)) | st.text(max_size=8)),
        "supporting_evidence": or_junk(st.lists(references, max_size=3)),
        "counterevidence": or_junk(st.lists(references, max_size=2)),
        "confidence": or_junk(st.none() | st.floats(0, 1)),
        "falsifier": or_junk(st.none() | st.text(max_size=8)),
        "contradicting_claim_ids": or_junk(st.lists(st.text(min_size=1, max_size=6), max_size=2)),
    },
)
claim_items = st.one_of(
    items,
    items,
    items,
    st.builds(lambda item, value: {**item, "unexpected": value}, items, json_values),
    json_values,
)
model_outputs = st.one_of(
    st.fixed_dictionaries(
        {"source_ids": st.just([VERSION])},
        optional={
            "facts": or_junk(st.lists(claim_items, max_size=4)),
            "inferences": or_junk(st.lists(claim_items, max_size=2)),
            "unknowns": or_junk(st.lists(st.text(min_size=1, max_size=20), max_size=2)),
            "contradicting_evidence": or_junk(st.lists(references, max_size=2)),
            "confidence": or_junk(st.none() | st.floats(0, 1)),
        },
    ),
    st.fixed_dictionaries(
        {},
        optional={
            "facts": st.lists(claim_items, max_size=2),
            "source_ids": json_values,
            "unexpected": json_values,
        },
    ),
    json_values,
)
adapter_outputs = model_outputs | not_json | st.lists(not_json, min_size=1, max_size=2)


def extract_once(output):
    """Run the extractor on fresh repositories; only the classified batch errors are caught."""
    source, claims = register(DOCUMENT)
    try:
        result = EvidenceExtractor(claims=claims, extraction_fn=lambda _content: output).extract(
            source,
            source_as_of=DOCUMENT.recorded_at,
            extracted_at=DOCUMENT.recorded_at,
            subject_id=DOCUMENT.source_id,
            run_id=RUN_ID,
            model_name="property",
            model_version="1",
            prompt_version="1",
        )
    except CLASSIFIED as error:
        return claims, None, error
    return claims, result, None


class ExtractionBoundaryProperties(unittest.TestCase):
    @given(adapter_outputs)
    def test_any_adapter_output_is_validated_item_by_item_or_refused_whole(self, output):
        claims, result, _error = extract_once(output)
        stored = claims.research_as_of(DOCUMENT.source_id, LATER)
        if result is None:
            self.assertFalse(claims.has_run(RUN_ID))
            self.assertEqual(stored, ())
            return
        self.assertTrue(claims.has_run(RUN_ID))
        self.assertEqual(
            sorted(claim.claim_id for claim in stored),
            sorted(claim.claim_id for claim in result.claims),
        )
        parsed = json.loads(result.raw_response_json)
        submitted = {bucket: len(parsed.get(bucket, [])) for bucket in ("facts", "inferences")}
        self.assertEqual(len(result.claims) + len(result.rejected_items), sum(submitted.values()))
        for item in result.rejected_items:
            self.assertIsInstance(item.reason, RejectionReason)
            self.assertLess(item.index, submitted[item.bucket])
        for claim in result.claims:
            self.assertEqual(claim.run_id, RUN_ID)
            if claim.category is ClaimCategory.FACT:
                self.assertTrue(claim.supporting_evidence)
            for reference in (*claim.supporting_evidence, *claim.counterevidence):
                self.assert_resolves(reference)
        for reference in result.contradicting_evidence:
            self.assert_resolves(reference)

    def assert_resolves(self, reference):
        self.assertEqual(reference.source_id, VERSION)
        if reference.source_chunk_id is None:
            expected = DOCUMENT_HASH
        else:
            self.assertIn(reference.source_chunk_id, CHUNKS)
            expected = CHUNK_HASHES[reference.source_chunk_id]
        self.assertIn(reference.source_content_hash, (None, expected))

    @given(adapter_outputs)
    def test_the_same_output_always_gets_the_same_verdict(self, output):
        def verdict():
            _claims, result, error = extract_once(output)
            if result is None:
                return type(error).__name__, str(error)
            value = result.to_dict()
            del value["latency_ms"]
            return value

        self.assertEqual(verdict(), verdict())


# --- generated provider responses -----------------------------------------------------

models = st.sampled_from(["claude-opus-5-5", "claude-sonnet-5-5", "claude-unlisted"])
counts = st.integers(0, 10**6)
usages = st.fixed_dictionaries(
    {"input_tokens": or_junk(counts), "output_tokens": or_junk(counts)},
    optional={
        "cache_creation_input_tokens": or_junk(counts),
        "cache_read_input_tokens": or_junk(counts),
        "cache_creation": or_junk(
            st.fixed_dictionaries(
                {"ephemeral_5m_input_tokens": counts, "ephemeral_1h_input_tokens": counts}
            )
        ),
        "service_tier": or_junk(st.sampled_from(["standard", "priority"])),
        "speed": or_junk(st.sampled_from(["standard", "fast"])),
        "inference_geo": or_junk(st.sampled_from(["global", "us"])),
        "iterations": or_junk(
            st.lists(
                st.fixed_dictionaries(
                    {
                        "type": st.sampled_from(["message", "fallback_message"]),
                        "model": or_junk(models),
                        "input_tokens": or_junk(counts),
                        "output_tokens": or_junk(counts),
                    }
                ),
                max_size=2,
            )
        ),
    },
)
model_reference = models.map(lambda model: {"model": model})
blocks = st.one_of(
    model_outputs.map(lambda output: {"type": "text", "text": json.dumps(output)}),
    st.fixed_dictionaries({"type": st.just("text"), "text": or_junk(st.text(max_size=20))}),
    st.fixed_dictionaries(
        {"type": st.just("fallback")},
        optional={"from": or_junk(model_reference), "to": or_junk(model_reference)},
    ),
    st.just({"type": "thinking", "thinking": "", "signature": "scripted-fixture"}),
    json_values,
)
messages = st.fixed_dictionaries(
    {"type": st.just("message"), "content": st.lists(blocks, max_size=3)},
    optional={
        "id": or_junk(st.text(max_size=8)),
        "model": or_junk(models),
        "stop_reason": or_junk(
            st.sampled_from(["end_turn", "refusal", "max_tokens", "pause_turn"])
        ),
        "stop_details": json_values,
        "usage": or_junk(usages),
    },
)
response_bodies = st.one_of(
    model_outputs.map(message_body),
    messages.map(json.dumps),
    json_values.map(json.dumps),
    st.text(max_size=30),
)


def provider_error(kind):
    retryable = kind == "retryable"
    return ProviderCallError(
        "scripted provider error",
        error_type="OverloadedError" if retryable else "BadRequestError",
        status_code=529 if retryable else 400,
        request_id="req_scripted_error",
        retryable=retryable,
    )


provider_outcomes = st.builds(
    lambda errors, final: [provider_error(kind) for kind in errors] + [final],
    st.lists(st.sampled_from(["retryable", "permanent"]), max_size=3),
    st.one_of(response_bodies.map(response), st.just("permanent").map(provider_error)),
)


class ModelRunProperties(unittest.TestCase):
    @given(provider_outcomes)
    def test_every_provider_outcome_is_recorded_replayed_exactly_and_evaluated(self, outcomes):
        transport = ScriptedTransport(*outcomes)
        record = run_model_extraction(
            DOCUMENT,
            ClaudeSettings(max_attempts=3),
            transport=transport,
            clock=fixed_clock,
            sleep=lambda _seconds: None,
        )
        self.assertIn(record["status"], RUN_STATUSES)
        exchange = record["exchange"]
        self.assertEqual(len(exchange["attempts"]), len(transport.requests))
        if exchange["raw_response"] is not None:
            self.assertEqual(exchange["response_sha256"], sha256_text(exchange["raw_response"]))
        with TemporaryDirectory() as directory:
            written = read_record(write_record(record, Path(directory), clock=fixed_clock))
        self.assertEqual(written, {**record, "record_written_at": fixed_clock().isoformat()})
        replay = replay_record(written)
        if exchange["raw_response"] is None:
            self.assertEqual(replay["status"], "not_replayable")
        else:
            self.assertEqual(replay["status"], record["status"])
            self.assertTrue(replay["matches_recorded_outcome"], replay["differences"])
        [entry] = record_entries([written], DATASET)
        self.assertNotEqual(entry["status"], "integrity_error", entry["diagnostics"])
        format_report(build_report(SYSTEM, [entry], DATASET, "eval"))


# --- hand-edited run records ----------------------------------------------------------


def paths(value, prefix=()):
    """Every location inside a JSON value, as a tuple of keys and list indexes."""
    yield prefix
    if isinstance(value, dict):
        children = list(value.items())
    elif isinstance(value, list):
        children = list(enumerate(value))
    else:
        children = []
    for key, child in children:
        yield from paths(child, (*prefix, key))


class _Delete:
    def __repr__(self):
        return "DELETE"


DELETE = _Delete()


def edited(record, path, replacement):
    if not path:
        return deepcopy(record) if replacement is DELETE else replacement
    copy = deepcopy(record)
    container = copy
    for key in path[:-1]:
        container = container[key]
    if replacement is DELETE:
        del container[path[-1]]
    else:
        container[path[-1]] = replacement
    return copy


BASE_RECORD = run_model_extraction(
    DOCUMENT,
    transport=ScriptedTransport(response(message_body(call_center_output()))),
    clock=fixed_clock,
    sleep=lambda _seconds: None,
)
[BASE_ENTRY] = record_entries([BASE_RECORD], DATASET)
BASE_SCORE = score_document(DOCUMENT, BASE_ENTRY["result"])
# The request's response schema is large and never read back; edit only its top levels.
RECORD_PATHS = [p for p in paths(BASE_RECORD) if p[:2] != ("exchange", "request") or len(p) <= 3]
HASHED = {
    ("exchange", "raw_response"),
    ("exchange", "response_sha256"),
    ("document", "sentences"),
    ("document", "content_hash"),
}
edits = st.tuples(st.sampled_from(RECORD_PATHS), st.one_of(st.just(DELETE), json_values))


class EditedRecordProperties(unittest.TestCase):
    def test_the_unedited_record_is_scored(self):
        self.assertEqual(BASE_ENTRY["status"], "completed")
        self.assertEqual(BASE_SCORE["claims"][0]["label"], "SUPPORTED")

    def assert_refused_reported_or_scored_as_the_original(self, path, replacement):
        record = edited(BASE_RECORD, path, replacement)
        try:
            entries = record_entries([record], DATASET)
        except ValueError as refusal:
            self.assertTrue(str(refusal).startswith("malformed run record: "), refusal)
            return
        report = build_report(SYSTEM, entries, DATASET, "eval")
        format_report(report)
        [entry] = entries
        if entry["status"] == "completed":
            self.assertEqual(score_document(DOCUMENT, entry["result"]), BASE_SCORE)
            for key in ("usage", "cost", "served_model"):
                self.assertEqual(entry[key], BASE_ENTRY[key], key)
        elif path[:2] in HASHED and record != BASE_RECORD:
            self.assertNotEqual(entry["status"], "completed")

    def test_deleting_nulling_or_retyping_any_field_is_refused_reported_or_scored(self):
        for path in RECORD_PATHS:
            for replacement in (DELETE, None, ["retyped"]):
                with self.subTest(path=path, replacement=replacement):
                    self.assert_refused_reported_or_scored_as_the_original(path, replacement)

    @given(edits)
    def test_an_edited_record_is_refused_reported_or_scored_exactly_as_the_original(self, edit):
        self.assert_refused_reported_or_scored_as_the_original(*edit)

    @given(edits)
    def test_the_cli_reports_an_edited_record_file_instead_of_crashing(self, edit):
        record = edited(BASE_RECORD, *edit)
        with TemporaryDirectory() as directory:
            Path(directory, "run.json").write_text(json.dumps(record), encoding="utf-8")
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                for command in (
                    ["evaluate", "--system", "recorded", "--records", directory],
                    ["replay", str(Path(directory, "run.json"))],
                ):
                    self.assertIn(main(command), (0, 1, 2))


# --- document version histories -------------------------------------------------------

TIMES = [datetime(2026, 1, day, tzinfo=timezone.utc) for day in range(1, 6)]
times = st.sampled_from(TIMES)


@st.composite
def histories(draw):
    versions = []
    for index in range(draw(st.integers(1, 7))):
        source = draw(st.sampled_from(["alpha", "beta", "gamma"]))
        ingested = draw(times)
        recorded = max(ingested, draw(times))
        content = RawDocumentContent(f"{source} report, version {index}")
        versions.append(
            SourceDocument(
                document_version_id=f"{source}-v{index}",
                source_id=source,
                source_type="report",
                document_location=f"test://{source}/{index}",
                raw_content=content,
                content_hash=content.content_hash(),
                publicly_available_at=draw(times),
                ingested_at=ingested,
                recorded_at=recorded,
            )
        )
    return versions


def expected_current(versions, as_of, mode, source=None):
    """The definition, not the implementation: per source, the one eligible version with no
    eligible sibling at a later timestamp; a tie for the latest timestamp is an error."""
    current = []
    for name in sorted({v.source_id for v in versions if source in (None, v.source_id)}):
        eligible = [
            v for v in versions if v.source_id == name and timestamp_for_mode(v, mode) <= as_of
        ]
        latest = [
            v
            for v in eligible
            if all(timestamp_for_mode(o, mode) <= timestamp_for_mode(v, mode) for o in eligible)
        ]
        if len(latest) > 1:
            return "AmbiguousRevisionError"
        current.extend(latest)
    return sorted(v.document_version_id for v in current)


def observed(query):
    try:
        return sorted(version.document_version_id for version in query())
    except AmbiguousRevisionError:
        return "AmbiguousRevisionError"


class HistoricalQueryProperties(unittest.TestCase):
    @given(histories(), st.sampled_from([*TIMES, LATER]), st.randoms(use_true_random=False))
    def test_every_backend_returns_the_latest_eligible_version_and_never_a_later_one(
        self, versions, as_of, rng
    ):
        shuffled = list(versions)
        rng.shuffle(shuffled)
        with TemporaryDirectory() as directory:
            path = Path(directory, "documents.jsonl")
            backends = {
                "memory": InMemorySourceDocumentRepository(),
                "jsonl": JsonlSourceDocumentRepository(path),
            }
            for version in shuffled:
                for repository in backends.values():
                    repository.add(version)
            backends["jsonl reopened"] = JsonlSourceDocumentRepository(path)
            for name, repository in backends.items():
                for mode, query in (
                    (AsOfMode.RESEARCH, repository.research_as_of),
                    (AsOfMode.OPERATIONAL_REPLAY, repository.operational_replay_as_of),
                ):
                    with self.subTest(backend=name, mode=mode.value):
                        self.assertEqual(
                            observed(lambda: query(as_of)), expected_current(versions, as_of, mode)
                        )
                for source in ("alpha", "beta", "gamma"):
                    for mode, lineage in (
                        (AsOfMode.RESEARCH, repository.research_as_of_lineage),
                        (AsOfMode.OPERATIONAL_REPLAY, repository.operational_replay_as_of_lineage),
                    ):
                        expected = expected_current(versions, as_of, mode, source)
                        found = observed(
                            lambda: [v for v in [lineage(source, as_of)] if v is not None]
                        )
                        with self.subTest(backend=name, mode=mode.value, source=source):
                            self.assertEqual(found, expected)
