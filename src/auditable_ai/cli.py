"""Credential-free walkthrough, validator regression checks, and recorded model runs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from auditable_ai.anthropic_adapter import (
    DEFAULT_EFFORT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    AnthropicTransport,
    ClaudeSettings,
    redact,
)
from auditable_ai.dataset import (
    DATASETS,
    DEFAULT_DATASET,
    DatasetDocument,
    bundled_document,
    load_dataset,
    load_document_file,
)
from auditable_ai.evaluation import (
    build_report,
    format_report,
    offline_entries,
    record_entries,
    settings_of,
)
from auditable_ai.runs import read_record, replay_record, run_model_extraction, write_record
from auditable_ai.scenarios import check_cases, fixture_data, repositories, run_case

EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
ORIGIN_LABELS = {
    "recorded_model_response": "RECORDED MODEL RESPONSE",
    "scripted_fixture": "SCRIPTED FIXTURE",
}


def write_new_report(path: Path, report: dict[str, Any]) -> None:
    """Never replace a prior audit artifact: an existing path is an error.

    The report is serialized first, so a failure leaves no partial file."""
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"{text} is not a positive integer")
    return value


def _positive_seconds(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(f"{text} is not a positive number of seconds")
    return value


def _demo(data: dict[str, Any]) -> dict[str, Any]:
    print("Auditable AI Pipeline | synthetic documents + scripted responses")
    print("No model call. Acceptance checks structure and provenance, not accuracy.\n")
    documents, _claims = repositories(data)
    snapshots = []
    for date in ("2026-04-05T12:00:00+00:00", "2026-04-17T12:00:00+00:00"):
        cutoff = datetime.fromisoformat(date)
        public = [doc.document_version_id for doc in documents.research_as_of(cutoff)]
        recorded = [doc.document_version_id for doc in documents.operational_replay_as_of(cutoff)]
        print(
            f"As of {cutoff.date()}: RESEARCH_AS_OF={public}; OPERATIONAL_REPLAY_AS_OF={recorded}"
        )
        snapshots.append(
            {"as_of": date, "RESEARCH_AS_OF": public, "OPERATIONAL_REPLAY_AS_OF": recorded}
        )
    cases = {case["name"]: case for case in data["cases"]}
    reports = []
    for name in (
        "valid_with_unknown_and_counterevidence",
        "fabricated_citation_with_valid_sibling",
        "false_but_well_formed_claim",
    ):
        result = run_case(data, cases[name])
        print(
            f"\n{name}: accepted={len(result.claims)}, rejected={len(result.rejected_items)}, unknowns={len(result.unknowns)}"
        )
        for claim in result.claims:
            print(f"  {claim.category.value} [accuracy NOT_VERIFIED]: {claim.claim_text}")
        for rejection in result.rejected_items:
            print(f"  REJECTED {rejection.reason.value}: {rejection.detail}")
        for unknown in result.unknowns:
            print(f"  UNKNOWN: {unknown}")
        for reference in result.contradicting_evidence:
            print(f"  COUNTEREVIDENCE: {reference.source_id}")
        reports.append(result.to_dict())
    print(
        "\nThe 900-sensor claim is intentionally false and still accepted: semantic verification is a separate task."
    )
    return {
        "fixture_notice": data["notice"],
        "result_origin": "scripted_fixture",
        "historical_snapshots": snapshots,
        "runs": reports,
    }


def _check_fixtures(data: dict[str, Any]) -> tuple[dict[str, Any], int]:
    checks = check_cases(data)
    for check in checks:
        print(f"{'PASS' if check['passed'] else 'FAIL'} {check['case']}")
    passed = sum(check["passed"] for check in checks)
    print(
        f"\n{passed}/{len(checks)} validator regression scenarios passed. This is not an LLM accuracy score."
    )
    report = {"fixture_notice": data["notice"], "checks": checks}
    return report, 0 if passed == len(checks) else 1


def _print_result(result: dict[str, Any] | None) -> None:
    if result is None:
        return
    for claim in result["claims"]:
        cited = ", ".join(
            reference["source_chunk_id"] or reference["source_id"]
            for reference in claim["supporting_evidence"]
        )
        print(f"  {claim['category']} [{claim['claim_type']}] {claim['claim_text']}  ({cited})")
    for item in result["rejected_items"]:
        print(f"  REJECTED {item['reason']}: {item['detail']}")
    for unknown in result["unknowns"]:
        print(f"  UNKNOWN: {unknown}")
    for reference in result["contradicting_evidence"]:
        print(f"  COUNTEREVIDENCE: {reference['source_id']}")


def _describe_run(record: dict[str, Any]) -> None:
    exchange = record.get("exchange") or {}
    response = record.get("response") or {}
    usage = response.get("usage") or {}
    cost = record["cost"]
    print(
        f"{ORIGIN_LABELS.get(record['origin'], record['origin'])}  status={record['status']}  "
        f"run_id={record['run_id']}"
    )
    print(
        f"model requested={record['settings']['model']} served={response.get('served_model')} "
        f"stop_reason={response.get('stop_reason')} request_id={exchange.get('request_id')}"
    )
    latency = exchange.get("latency_ms")
    print(
        f"attempts={len(exchange.get('attempts', []))} "
        f"latency={'n/a' if latency is None else f'{latency:.0f} ms'} (final attempt) "
        f"tokens in/out={usage.get('input_tokens')}/{usage.get('output_tokens')} "
        + (
            f"est. cost=${cost['usd']:.6f} (list prices {cost['basis']['retrieved_on']})"
            if cost["status"] == "estimated"
            else f"est. cost unavailable: {cost['reason']}"
        )
    )
    _print_result(record.get("result"))
    if record.get("diagnostics"):
        print(f"  DIAGNOSTICS: {record['diagnostics']['message']}")


def _settings(args: argparse.Namespace) -> ClaudeSettings:
    return ClaudeSettings(
        model=args.model,
        effort=args.effort,
        max_tokens=args.max_tokens,
        fallbacks=None if args.fallbacks == "none" else args.fallbacks,
        max_attempts=args.max_attempts,
        timeout_s=args.timeout,
    )


def _extract(args: argparse.Namespace) -> int:
    if not args.live:
        print(
            "extract calls the Anthropic API and may incur cost; re-run with --live to proceed.",
            file=sys.stderr,
        )
        return 2
    try:
        if args.dataset_document:
            document = bundled_document(args.dataset_document)
        else:
            document = load_document_file(args.document)
    except (KeyError, OSError, ValueError) as exc:
        print(f"Cannot load the document: {exc}", file=sys.stderr)
        return 2
    try:
        transport = AnthropicTransport(timeout=args.timeout)
    except Exception as exc:  # missing SDK or credentials: nothing has been sent yet
        print(f"Cannot create the Anthropic client: {redact(str(exc))}", file=sys.stderr)
        return 2
    record = run_model_extraction(document, _settings(args), transport=transport)
    path = write_record(record, args.record_dir)
    _describe_run(record)
    print(f"record: {path}")
    return 0 if record["status"] == "completed" else 1


def _replay(args: argparse.Namespace) -> int:
    try:
        record = read_record(args.record)
        replay = replay_record(record)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Cannot replay {args.record}: {exc}", file=sys.stderr)
        return 2
    print(
        f"REPLAYED RECORDING (no model call) of run {record['run_id']} "
        f"(origin={record['origin']}, recorded status={record['status']})"
    )
    if replay["status"] == "not_replayable":
        print(f"not replayable: {replay['reason']}")
    else:
        matches = replay["matches_recorded_outcome"]
        print(
            f"status={replay['status']}  matches recorded outcome: "
            + ("yes" if matches else "no, differs in " + ", ".join(replay["differences"]))
        )
        _print_result(replay.get("result"))
        if replay.get("diagnostics"):
            print(f"  DIAGNOSTICS: {replay['diagnostics']['message']}")
    exit_code = 0 if replay.get("matches_recorded_outcome") else 1
    return _write_output(args.output, replay, exit_code)


def _live_records(
    args: argparse.Namespace, documents: tuple[DatasetDocument, ...]
) -> list[dict[str, Any]] | None:
    if not args.live:
        print(
            "evaluate --system live calls the Anthropic API for every document and repetition "
            "and may incur cost; re-run with --live to proceed.",
            file=sys.stderr,
        )
        return None
    try:
        transport = AnthropicTransport(timeout=args.timeout)
    except Exception as exc:  # missing SDK or credentials: nothing has been sent yet
        print(f"Cannot create the Anthropic client: {redact(str(exc))}", file=sys.stderr)
        return None
    settings = _settings(args)
    records = []
    for _repetition in range(args.reps):
        for document in documents:
            record = run_model_extraction(document, settings, transport=transport)
            path = write_record(record, args.record_dir)
            print(f"{record['status']:>18}  {document.document_version_id}  {path}")
            records.append(record)
    return records


def _evaluate(args: argparse.Namespace) -> int:
    dataset = load_dataset(args.dataset)
    documents = dataset.split(args.split)
    if args.system == "baseline":
        system: dict[str, Any] = {"name": "rule_baseline", "origin": "rule_based"}
        entries = offline_entries("rule_baseline", documents)
    else:
        if args.system == "live":
            records = _live_records(args, documents)
            if records is None:
                return 2
        elif args.records is None:
            print("evaluate --system recorded needs --records DIR", file=sys.stderr)
            return 2
        else:
            try:
                records = [read_record(path) for path in sorted(args.records.glob("*.json"))]
            except (OSError, ValueError) as exc:
                print(f"Cannot read run records: {exc}", file=sys.stderr)
                return 2
        if not records:
            print("No run records to evaluate.", file=sys.stderr)
            return 2
        try:
            settings = settings_of(records)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 2
        system = {"name": "claude", "origin": "replayed_recording", "settings": settings}
        entries = record_entries(records, dataset)
    report = build_report(system, entries, dataset, args.split)
    print(format_report(report))
    return _write_output(args.output, report, 0 if report["runs"]["scored"] else 1)


def _write_output(path: Path | None, report: dict[str, Any], exit_code: int) -> int:
    if path is None:
        return exit_code
    try:
        write_new_report(path, report)
    except FileExistsError:
        print(
            f"Report already exists: {path}. Choose a new path to preserve audit history.",
            file=sys.stderr,
        )
        return 2
    except OSError as exc:
        print(f"Cannot write report {path}: {exc.strerror or exc}", file=sys.stderr)
        return 2
    return exit_code


def _add_model_options(parser: argparse.ArgumentParser, *, fallbacks: str) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--effort", choices=EFFORT_LEVELS, default=DEFAULT_EFFORT)
    parser.add_argument("--max-tokens", type=_positive_int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--fallbacks",
        choices=("default", "none"),
        default=fallbacks,
        help="server-side refusal fallback; 'none' keeps every response on the requested model",
    )
    parser.add_argument("--max-attempts", type=_positive_int, default=3)
    parser.add_argument(
        "--timeout", type=_positive_seconds, default=120.0, help="seconds per request"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, summary in (
        ("demo", "Walk through scripted fixtures; no model call"),
        ("check-fixtures", "Run the eight scripted validator regression scenarios"),
    ):
        command = commands.add_parser(name, help=summary)
        command.add_argument(
            "--output",
            type=Path,
            help="Write a new JSON report; refuse to overwrite an existing file",
        )
    extract = commands.add_parser(
        "extract", help="Call Claude on one document and write a run record (network; may cost)"
    )
    source = extract.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-document", metavar="ID", help="a bundled dataset document")
    source.add_argument("--document", type=Path, metavar="FILE", help="a document JSON file")
    extract.add_argument(
        "--live", action="store_true", help="confirm that this may call the API and incur cost"
    )
    extract.add_argument("--record-dir", type=Path, default=Path("runs"))
    _add_model_options(extract, fallbacks="default")
    replay = commands.add_parser("replay", help="Re-validate a recorded run without a model call")
    replay.add_argument("record", type=Path)
    replay.add_argument("--output", type=Path, help="Write a new JSON replay report")
    evaluate = commands.add_parser(
        "evaluate", help="Score a system on a figure-extraction-v1 dataset with the frozen rubric"
    )
    evaluate.add_argument("--system", choices=("baseline", "recorded", "live"), required=True)
    evaluate.add_argument(
        "--dataset",
        choices=tuple(DATASETS),
        default=DEFAULT_DATASET,
        help="bundled dataset to score against; results are comparable only within one",
    )
    evaluate.add_argument("--split", choices=("dev", "eval", "all"), default="eval")
    evaluate.add_argument("--records", type=Path, metavar="DIR", help="run records to score")
    evaluate.add_argument(
        "--live", action="store_true", help="confirm that --system live may call the API and cost"
    )
    evaluate.add_argument(
        "--reps", type=_positive_int, default=1, help="repetitions per document (live)"
    )
    evaluate.add_argument("--record-dir", type=Path, default=Path("runs"))
    evaluate.add_argument("--output", type=Path, help="Write a new JSON evaluation report")
    _add_model_options(evaluate, fallbacks="none")
    args = parser.parse_args(argv)
    if args.command == "extract":
        return _extract(args)
    if args.command == "replay":
        return _replay(args)
    if args.command == "evaluate":
        return _evaluate(args)
    data = fixture_data()
    if args.command == "demo":
        report, exit_code = _demo(data), 0
    else:
        report, exit_code = _check_fixtures(data)
    return _write_output(args.output, report, exit_code)
