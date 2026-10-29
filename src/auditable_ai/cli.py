"""Credential-free walkthrough and deterministic validator regression checks."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from auditable_ai.scenarios import check_cases, fixture_data, repositories, run_case


def write_new_report(path: Path, report: dict) -> None:
    """Never replace a prior audit artifact: an existing path is an error."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _demo(data: dict) -> dict:
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
    return {"fixture_notice": data["notice"], "historical_snapshots": snapshots, "runs": reports}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("demo", "check-fixtures"))
    parser.add_argument(
        "--output", type=Path, help="Write a new JSON report; refuse to overwrite an existing file"
    )
    args = parser.parse_args(argv)
    data = fixture_data()
    exit_code = 0
    if args.command == "demo":
        report = _demo(data)
    else:
        checks = check_cases(data)
        for check in checks:
            print(f"{'PASS' if check['passed'] else 'FAIL'} {check['case']}")
        passed = sum(check["passed"] for check in checks)
        print(
            f"\n{passed}/{len(checks)} validator regression scenarios passed. This is not an LLM accuracy score."
        )
        report = {"fixture_notice": data["notice"], "checks": checks}
        exit_code = 0 if passed == len(checks) else 1
    if args.output is not None:
        try:
            write_new_report(args.output, report)
        except FileExistsError:
            print(
                f"Report already exists: {args.output}. Choose a new path to preserve audit history.",
                file=sys.stderr,
            )
            return 2
    return exit_code
