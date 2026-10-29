"""Deterministic scoring for figure-extraction-v1 (rubric in docs/evaluation.md).

Gold annotations were written by an AI model and are unreviewed; every report
carries their provenance. Schema and provenance acceptance (the validator) and
semantic support (this scorer) are reported separately, failed runs are listed
rather than scored, and judgments the rules cannot make are kept as unresolved
instead of being forced into a score."""

from __future__ import annotations

import math
import re
import statistics
import time
from collections import Counter
from typing import Any, Callable, Iterable, Mapping

from auditable_ai.dataset import (
    Dataset,
    DatasetDocument,
    format_number,
    normalize_words,
    numbers_in,
    register,
)
from auditable_ai.extraction import VALIDATOR_VERSION, EvidenceExtractor
from auditable_ai.figure_task import CLAIM_TYPE_STATUS, TASK_VERSION
from auditable_ai.pricing import PRICES_RETRIEVED_ON, estimate_cost
from auditable_ai.runs import check_record, replay_record, validated_outcome

RUBRIC_VERSION = "figure-rubric-v1"
SUPPORTED = "SUPPORTED"
DUPLICATE = "DUPLICATE"
WRONG_STATUS = "WRONG_STATUS"
CITATION_MISMATCH = "CITATION_MISMATCH"
UNSUPPORTED_VALUE = "UNSUPPORTED_VALUE"
TRAP = "TRAP"
OUT_OF_SCOPE = "OUT_OF_SCOPE"
UNRESOLVED = "UNRESOLVED"
LABELS = (
    SUPPORTED,
    DUPLICATE,
    WRONG_STATUS,
    CITATION_MISMATCH,
    UNSUPPORTED_VALUE,
    TRAP,
    OUT_OF_SCOPE,
    UNRESOLVED,
)
_STATUS_CLAIM_TYPE = {status: claim_type for claim_type, status in CLAIM_TYPE_STATUS.items()}
_IDENTIFIERS = re.compile(r"\[[^\]]*\]|[\w.-]+:s\d+")

_ATTRIBUTION = re.compile(
    r"\b(according to|claims?|claimed|says|said|asserts?|asserted|estimates?|estimated|"
    r"projects?|projected|expects?|expected|predicts?|predicted|forecasts?|believes?)\b",
    re.IGNORECASE,
)
_CORRECTION = re.compile(
    r"\b(corrected|revised|replaces?|replaced|instead of|not the|earlier report|original report|"
    r"first reported|previously reported|counted twice|duplicate)\b",
    re.IGNORECASE,
)
_CONFLICT = re.compile(r"\b(while|whereas|but|however|differs?|conflicting)\b", re.IGNORECASE)
_MISSING = re.compile(
    r"\b(not been (measured|recorded|collected|identified|confirmed|determined)|"
    r"not (measured|recorded|collected|available)|did not record|unavailable|missing|pending|"
    r"not yet|unknown)\b",
    re.IGNORECASE,
)


def _reference(document: DatasetDocument, chunk_id: str) -> dict[str, Any]:
    return {
        "source_id": document.document_version_id,
        "source_chunk_id": chunk_id,
        "locator_note": None,
    }


def _response(
    document: DatasetDocument, facts: list[dict[str, Any]], unknowns: list[str]
) -> dict[str, Any]:
    return {
        "facts": facts,
        "inferences": [],
        "unknowns": unknowns,
        "contradicting_evidence": [],
        "source_ids": [document.document_version_id],
    }


def baseline_response(document: DatasetDocument) -> dict[str, Any]:
    """Rule-based extraction from generic cue words: every non-year number becomes a fact.

    Correction sentences treat the first number as current and later ones as
    superseded; attribution and conflict cues set the other claim types; and
    sentences with missing-information cues become unknowns."""
    facts = []
    unknowns = []
    for chunk_id, sentence in document.chunk_texts.items():
        values = [v for v in numbers_in(sentence) if not (v.is_integer() and 1900 <= v <= 2100)]
        for position, value in enumerate(values):
            if _ATTRIBUTION.search(sentence):
                claim_type = "attributed_assertion"
            elif _CORRECTION.search(sentence) and len(values) > 1:
                claim_type = "current_figure" if position == 0 else "superseded_figure"
            elif _CONFLICT.search(sentence) and len(values) > 1:
                claim_type = "disputed_figure"
            else:
                claim_type = "current_figure"
            reference = _reference(document, chunk_id)
            facts.append(
                {
                    "claim_text": f"The report states the figure {format_number(value)}.",
                    "claim_type": claim_type,
                    "supporting_evidence": [reference],
                    "counterevidence": [reference] if claim_type == "disputed_figure" else [],
                }
            )
        if _MISSING.search(sentence):
            unknowns.append(sentence)
    return _response(document, facts, unknowns)


def oracle_response(document: DatasetDocument) -> dict[str, Any]:
    """The gold annotations rendered as output: a harness check that must score fully."""
    facts = [
        {
            "claim_text": f"{figure.quantity}: {format_number(figure.value)} {figure.unit}.",
            "claim_type": _STATUS_CLAIM_TYPE[figure.status],
            "supporting_evidence": [_reference(document, chunk) for chunk in figure.chunks],
            "counterevidence": [_reference(document, chunk) for chunk in figure.counter_chunks],
        }
        for figure in document.figures
        if figure.status != "ambiguous"
    ]
    return _response(document, facts, [unknown.description for unknown in document.unknowns])


def null_response(document: DatasetDocument) -> dict[str, Any]:
    """An empty extraction: a harness check that must cover nothing."""
    return _response(document, [], [])


OFFLINE_SYSTEMS: dict[str, Callable[[DatasetDocument], dict[str, Any]]] = {
    "rule_baseline": baseline_response,
    "oracle": oracle_response,
    "null": null_response,
}


def _matches_unknown(text: str, groups: Iterable[Iterable[str]]) -> bool:
    words = normalize_words(text)
    return any(all(normalize_words(keyword) in words for keyword in group) for group in groups)


def label_claim(
    document: DatasetDocument, claim: Mapping[str, Any], covered: set[str]
) -> dict[str, Any]:
    """Apply the rubric's label rules, in order, to one accepted FACT."""
    text = claim["claim_text"]
    numbers = set(numbers_in(_IDENTIFIERS.sub(" ", text)))
    cited = {reference.get("source_chunk_id") for reference in claim["supporting_evidence"]}
    status = CLAIM_TYPE_STATUS.get(claim["claim_type"])
    entry: dict[str, Any] = {
        "claim_text": text,
        "claim_type": claim["claim_type"],
        "cited": sorted(chunk for chunk in cited if chunk),
        "figures": [],
        "reason": None,
    }
    if not numbers:
        return {**entry, "label": UNRESOLVED, "reason": "no number in claim_text"}
    absent = numbers - set(numbers_in(document.text))
    if absent:
        values = ", ".join(format_number(value) for value in sorted(absent))
        return {**entry, "label": UNSUPPORTED_VALUE, "reason": f"{values} not in the document"}
    if numbers & {trap.value for trap in document.traps}:
        return {**entry, "label": TRAP, "reason": "reports a number the document marks as a trap"}
    in_cited = [f for f in document.figures if f.value in numbers and cited & set(f.chunks)]
    if in_cited:
        ids = [figure.id for figure in in_cited]
        if any(figure.status == "ambiguous" for figure in in_cited):
            reason = "the matching figure's status is annotated as ambiguous"
            return {**entry, "label": UNRESOLVED, "figures": ids, "reason": reason}
        matching = [figure for figure in in_cited if figure.status == status]
        if not matching:
            statuses = ", ".join(figure.status for figure in in_cited)
            reason = f"claim type {claim['claim_type']} but annotated status {statuses}"
            return {**entry, "label": WRONG_STATUS, "figures": ids, "reason": reason}
        new = [figure.id for figure in matching if figure.id not in covered]
        covered.update(new)
        words = normalize_words(text)
        confirmed = any(
            normalize_words(keyword) in words
            for figure in matching
            for keyword in figure.description_keywords
        )
        return {
            **entry,
            "label": SUPPORTED if new else DUPLICATE,
            "figures": [figure.id for figure in matching],
            "description_confirmed": confirmed,
        }
    elsewhere = [figure.id for figure in document.figures if figure.value in numbers]
    if elsewhere:
        reason = "no cited sentence states the matched figure"
        return {**entry, "label": CITATION_MISMATCH, "figures": elsewhere, "reason": reason}
    return {**entry, "label": OUT_OF_SCOPE, "reason": "only dates or identifiers"}


def score_document(document: DatasetDocument, result: Mapping[str, Any]) -> dict[str, Any]:
    """Score one completed run's validated result against the document's annotations."""
    covered: set[str] = set()
    facts = [claim for claim in result["claims"] if claim["category"] == "FACT"]
    claims = [label_claim(document, claim, covered) for claim in facts]
    figures = []
    for figure in document.figures:
        if figure.id in covered:
            omission = None
        elif any(c["label"] == WRONG_STATUS and figure.id in c["figures"] for c in claims):
            omission = "wrong_status"
        elif any(c["label"] == CITATION_MISMATCH and figure.id in c["figures"] for c in claims):
            omission = "citation_mismatch"
        else:
            omission = "not_found"
        figures.append(
            {
                "id": figure.id,
                "status": figure.status,
                "covered": figure.id in covered,
                "omission": omission if figure.status != "ambiguous" else None,
            }
        )
    disputed = {figure.id: figure for figure in document.figures if figure.status == "disputed"}
    disputed_claims = 0
    with_counterevidence = 0
    for labeled, claim in zip(claims, facts):
        credited = [disputed[i] for i in labeled["figures"] if i in disputed]
        if labeled["label"] in (SUPPORTED, DUPLICATE) and credited:
            disputed_claims += 1
            counter = {reference.get("source_chunk_id") for reference in claim["counterevidence"]}
            with_counterevidence += any(counter & set(f.counter_chunks) for f in credited)
    unknowns = list(result["unknowns"])
    matched = [
        u.id for u in document.unknowns if any(_matches_unknown(t, u.match_any) for t in unknowns)
    ]
    unmatched = [
        text
        for text in unknowns
        if not any(_matches_unknown(text, unknown.match_any) for unknown in document.unknowns)
    ]
    rejected = Counter(item["reason"] for item in result["rejected_items"])
    return {
        "document_version_id": document.document_version_id,
        "claims": claims,
        "figures": figures,
        "disputed": {
            "supported_claims": disputed_claims,
            "with_counterevidence": with_counterevidence,
        },
        "unknowns": {
            "gold": [unknown.id for unknown in document.unknowns],
            "matched": matched,
            "unmatched_output": unmatched,
        },
        "items_submitted": len(result["claims"]) + len(result["rejected_items"]),
        "rejected": dict(rejected),
        "out_of_task_inferences": len(result["claims"]) - len(facts),
    }


def ratio(numerator: int, denominator: int) -> dict[str, Any]:
    rate = round(numerator / denominator, 3) if denominator else None
    return {"numerator": numerator, "denominator": denominator, "rate": rate}


def summarize(scores: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter({label: 0 for label in LABELS})
    labels.update(claim["label"] for score in scores for claim in score["claims"])
    resolved = sum(labels.values()) - labels[UNRESOLVED]
    supported = labels[SUPPORTED] + labels[DUPLICATE]
    located = supported + labels[WRONG_STATUS]
    figures = [figure for score in scores for figure in score["figures"]]
    scorable = [figure for figure in figures if figure["status"] != "ambiguous"]
    confirmed = sum(
        bool(claim.get("description_confirmed"))
        for score in scores
        for claim in score["claims"]
        if claim["label"] in (SUPPORTED, DUPLICATE)
    )
    submitted = sum(score["items_submitted"] for score in scores)
    provenance = sum(score["rejected"].get("PROVENANCE_VIOLATION", 0) for score in scores)
    return {
        "labels": dict(labels),
        "claim_support": ratio(supported, resolved),
        "unsupported_claims": {
            **ratio(labels[WRONG_STATUS] + labels[UNSUPPORTED_VALUE] + labels[TRAP], resolved),
            "wrong_status": labels[WRONG_STATUS],
            "unsupported_value": labels[UNSUPPORTED_VALUE],
            "trap": labels[TRAP],
        },
        "figure_coverage": ratio(sum(f["covered"] for f in scorable), len(scorable)),
        "coverage_by_status": {
            status: ratio(
                sum(f["covered"] for f in scorable if f["status"] == status),
                sum(1 for f in scorable if f["status"] == status),
            )
            for status in CLAIM_TYPE_STATUS.values()
        },
        "omissions": dict(Counter(f["omission"] for f in scorable if not f["covered"])),
        "ambiguous_figures": len(figures) - len(scorable),
        "citation_validity": {
            "provenance_rejections": ratio(provenance, submitted),
            "semantic": ratio(located, located + labels[CITATION_MISMATCH]),
        },
        "description_check": ratio(confirmed, supported),
        "disputed_counterevidence": ratio(
            sum(score["disputed"]["with_counterevidence"] for score in scores),
            sum(score["disputed"]["supported_claims"] for score in scores),
        ),
        "unknown_coverage": ratio(
            sum(len(score["unknowns"]["matched"]) for score in scores),
            sum(len(score["unknowns"]["gold"]) for score in scores),
        ),
        "items": {
            "submitted": submitted,
            "rejected": sum(sum(score["rejected"].values()) for score in scores),
        },
        "out_of_task_inferences": sum(score["out_of_task_inferences"] for score in scores),
        "unresolved": [
            {
                "document_version_id": score["document_version_id"],
                "claim_text": claim["claim_text"],
                "reason": claim["reason"],
            }
            for score in scores
            for claim in score["claims"]
            if claim["label"] == UNRESOLVED
        ],
        "unmatched_output_unknowns": [
            {"document_version_id": score["document_version_id"], "unknown": text}
            for score in scores
            for text in score["unknowns"]["unmatched_output"]
        ],
    }


def offline_entries(system: str, documents: Iterable[DatasetDocument]) -> list[dict[str, Any]]:
    """Run a non-model system through the same validator the model output goes through."""
    respond = OFFLINE_SYSTEMS[system]
    entries = []
    for document in documents:
        source, claims = register(document)
        began = time.perf_counter()
        output = respond(document)
        elapsed = (time.perf_counter() - began) * 1000
        status, result, diagnostics = validated_outcome(
            lambda: EvidenceExtractor(claims=claims, extraction_fn=lambda _content: output).extract(
                source,
                source_as_of=document.recorded_at,
                extracted_at=document.recorded_at,
                subject_id=document.source_id,
                run_id=f"{system}--{document.document_version_id}",
                model_name=system,
                model_version=RUBRIC_VERSION,
                prompt_version="not-applicable",
            )
        )
        entries.append(
            {
                "document_version_id": document.document_version_id,
                "rep": 1,
                "status": status,
                "result": result.to_dict() if result else None,
                "diagnostics": diagnostics,
                "latency_ms": elapsed,
                "usage": None,
                "cost": None,
            }
        )
    return entries


def _integrity_problem(
    record: Mapping[str, Any], replay: Mapping[str, Any], served_as_requested: bool | None
) -> str | None:
    """Why a well-formed record contradicts its own hash-checked content, if it does."""
    if replay["status"] == "not_replayable":
        if record["status"] == "provider_error":
            return None
        return f"the record reports status {record['status']!r} but holds no provider response"
    if record["validator_version"] == VALIDATOR_VERSION and replay["differences"]:
        stored = ", ".join(replay["differences"])
        return f"replay under validator {VALIDATOR_VERSION} does not reproduce the stored {stored}"
    if record["served_model_matches_request"] != served_as_requested:
        return "served_model_matches_request disagrees with the recorded response"
    return None


def record_entries(records: Iterable[Mapping[str, Any]], dataset: Dataset) -> list[dict[str, Any]]:
    """Replay recorded model runs (no model calls) and turn them into scoring entries.

    A malformed record raises ValueError. For the rest, the replayed hash-checked
    response supplies the result, token usage, served model and cost; latency is
    a measurement, so it comes from the record as written. A record whose content
    fails its hashes, or whose stored outcome or summary contradicts that content,
    is an integrity failure and is never scored."""
    records = list(records)
    for record in records:
        check_record(record)
    known = {document.document_version_id: document for document in dataset.documents}
    entries = []
    reps: Counter[str] = Counter()
    for record in sorted(records, key=lambda item: item["run_id"]):
        document_id = record["document"]["document_version_id"]
        reps[document_id] += 1
        entry: dict[str, Any] = {
            "document_version_id": document_id,
            "rep": reps[document_id],
            "run_id": record["run_id"],
            "origin": record["origin"],
            "latency_ms": (record["exchange"] or {}).get("latency_ms"),
            "usage": None,
            "cost": None,
            "served_model": None,
        }
        try:
            replay = replay_record(record)
        except ValueError as exc:  # ProvenanceError or TemporalIntegrityError: altered content
            message = f"{type(exc).__name__}: {exc}"
            entries.append(
                {
                    **entry,
                    "status": "integrity_error",
                    "result": None,
                    "diagnostics": {"message": message},
                }
            )
            continue
        response = replay.get("response")
        served_as_requested = None
        if response:
            served_as_requested = response["served_model"] == record["settings"]["model"]
            entry["usage"] = response["usage"]
            entry["served_model"] = response["served_model"]
            entry["cost"] = estimate_cost(
                response["usage"], response["served_model"], stop_reason=response["stop_reason"]
            )
        diagnostics = replay.get("diagnostics") or record.get("diagnostics")
        problem = _integrity_problem(record, replay, served_as_requested)
        if problem:
            status, diagnostics = "integrity_error", {"message": problem}
        elif replay["status"] == "not_replayable":
            status = record["status"]
        elif document_id not in known:
            status = "unknown_document"
        elif known[document_id].text != " ".join(record["document"]["sentences"]):
            status = "document_mismatch"
        elif record["task"]["version"] != TASK_VERSION:
            status = "task_version_mismatch"
        elif replay["status"] == "completed" and served_as_requested is False:
            status = "served_model_mismatch"
        else:
            status = replay["status"]
        entries.append(
            {
                **entry,
                "status": status,
                "result": replay.get("result") if status == "completed" else None,
                "diagnostics": diagnostics,
                "replay_matches_record": replay.get("matches_recorded_outcome"),
            }
        )
    return entries


def settings_of(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """The shared settings of a set of records; mixed settings are evaluated separately."""
    keys = ("provider", "model", "effort", "max_tokens", "fallbacks")
    found = {tuple(record["settings"].get(key) for key in keys) for record in records}
    if len(found) != 1:
        raise ValueError(
            f"records mix settings {sorted(found, key=str)}; evaluate each set separately"
        )
    return dict(zip(keys, found.pop()))


def _performance(scored: list[dict[str, Any]], *, model: bool) -> dict[str, Any]:
    latencies = [entry["latency_ms"] for entry in scored if entry["latency_ms"] is not None]
    timing = {
        "basis": "final API attempt per run" if model else "local computation, not API latency",
        "runs": len(latencies),
        "median_ms": round(statistics.median(latencies), 3) if latencies else None,
        "max_ms": round(max(latencies), 3) if latencies else None,
    }
    if not model:
        return {"latency": timing, "tokens": None, "cost": "not applicable: no API call"}
    counted = [counts for counts in (_token_counts(entry["usage"]) for entry in scored) if counts]
    costs = [entry["cost"] for entry in scored if entry["cost"]]
    estimated = [cost["usd"] for cost in costs if cost["status"] == "estimated"]
    return {
        "latency": timing,
        "tokens": {
            "input": sum(counts[0] for counts in counted),
            "output": sum(counts[1] for counts in counted),
            "cache_read": sum(counts[2] for counts in counted),
            "runs": len(counted),
        },
        "cost": {
            "estimated_usd": round(sum(estimated), 6),
            "runs_estimated": len(estimated),
            "runs_unavailable": len(costs) - len(estimated),
            "basis": "API-reported tokens x published list prices; not an invoice",
            "prices_retrieved_on": PRICES_RETRIEVED_ON,
        },
    }


def _token_counts(usage: Any) -> tuple[int, int, int] | None:
    """Input, output and cache-read tokens, or None for usage that is absent or malformed."""
    if not isinstance(usage, Mapping):
        return None
    names = ("input_tokens", "output_tokens", "cache_read_input_tokens")
    values = [usage.get(name) or 0 for name in names]
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        return None
    if min(values) < 0:
        return None
    return values[0], values[1], values[2]


def build_report(
    system: dict[str, Any],
    entries: list[dict[str, Any]],
    dataset: Dataset,
    split: str,
) -> dict[str, Any]:
    documents = {document.document_version_id: document for document in dataset.split(split)}
    selected = [entry for entry in entries if entry["document_version_id"] in documents]
    scored = [entry for entry in selected if entry["status"] == "completed"]
    scores = [score_document(documents[e["document_version_id"]], e["result"]) for e in scored]
    model = system["origin"] == "replayed_recording"
    scored_documents = {entry["document_version_id"] for entry in scored}
    caveats = [
        "Gold annotations were written by an AI model and have not been reviewed by a person; "
        "see annotation_provenance. Treat these numbers as provisional.",
        f"{len(documents)} documents: with documents as the unit, differences smaller than about "
        f"{round(100 / math.sqrt(len(documents)))} percentage points are within noise for one "
        "repetition.",
        "The scorer checks numbers, cited sentences and claim types. It cannot confirm that a "
        "supported claim describes the right quantity; description_check is a keyword proxy.",
        "Schema and provenance acceptance is not semantic accuracy; see the claim labels.",
    ]
    if system["origin"] == "rule_based":
        caveats.append(
            "This is a rule-based baseline, not a model. Its latency is local computation and is "
            "not comparable with API latency."
        )
    origins = Counter(entry["origin"] for entry in selected if entry.get("origin"))
    if model and set(origins) != {"recorded_model_response"}:
        caveats.append(
            f"Some responses did not come from a real API call: {dict(origins)}. Do not report "
            "these numbers as measured model performance."
        )
    return {
        "report_type": "figure_extraction_evaluation",
        "task_version": TASK_VERSION,
        "rubric_version": RUBRIC_VERSION,
        "dataset": dataset.name,
        "dataset_sha256": dataset.sha256,
        "split": split,
        "system": system,
        "response_origins": dict(origins),
        "annotation_provenance": dataset.provenance,
        "caveats": caveats,
        "runs": {
            "attempted": len(selected),
            "scored": len(scored),
            "failed": dict(Counter(e["status"] for e in selected if e["status"] != "completed")),
            "documents": len(documents),
            "documents_without_a_scored_run": sorted(set(documents) - scored_documents),
            "ignored_entries_outside_split": len(entries) - len(selected),
        },
        "summary": summarize(scores),
        "performance": _performance(scored, model=model),
        "failures": [
            {
                "document_version_id": entry["document_version_id"],
                "rep": entry["rep"],
                "status": entry["status"],
                "diagnostics": entry.get("diagnostics"),
            }
            for entry in selected
            if entry["status"] != "completed"
        ],
        "documents": scores,
    }


def _ratio_text(value: Mapping[str, Any]) -> str:
    rate = "n/a" if value["rate"] is None else f"{value['rate']:.3f}"
    return f"{value['numerator']}/{value['denominator']} ({rate})"


def markdown_summary(report: Mapping[str, Any]) -> str:
    """The headline numbers as a Markdown table; docs/evaluation.md embeds it verbatim."""
    summary = report["summary"]
    unsupported = summary["unsupported_claims"]
    by_status = ", ".join(
        f"{status} {value['numerator']}/{value['denominator']}"
        for status, value in summary["coverage_by_status"].items()
    )
    rows = [
        ("Runs scored", f"{report['runs']['scored']} of {report['runs']['attempted']}"),
        ("Figure coverage", f"{_ratio_text(summary['figure_coverage'])}: {by_status}"),
        ("Claim support", _ratio_text(summary["claim_support"])),
        (
            "Unsupported claims",
            f"{_ratio_text(unsupported)}: wrong status {unsupported['wrong_status']}, "
            f"absent value {unsupported['unsupported_value']}, trap {unsupported['trap']}",
        ),
        ("Out-of-scope claims (dates, identifiers)", str(summary["labels"][OUT_OF_SCOPE])),
        ("Unresolved, left for human review", str(summary["labels"][UNRESOLVED])),
        (
            "Citation validity",
            f"provenance rejections {_ratio_text(summary['citation_validity']['provenance_rejections'])}; "
            f"semantic {_ratio_text(summary['citation_validity']['semantic'])}",
        ),
        ("Disputed figures with counterevidence", _ratio_text(summary["disputed_counterevidence"])),
        ("Unknown coverage", _ratio_text(summary["unknown_coverage"])),
        ("Description check (keyword proxy)", _ratio_text(summary["description_check"])),
    ]
    lines = ["| Measure | Result |", "|---|---|"]
    lines.extend(f"| {name} | {value} |" for name, value in rows)
    return "\n".join(lines)


def format_report(report: Mapping[str, Any]) -> str:
    system = report["system"]
    runs = report["runs"]
    performance = report["performance"]
    lines = [
        f"{report['task_version']} | split={report['split']} ({runs['documents']} documents) | "
        f"system={system['name']} ({system['origin']})",
        "Annotations: AI-written, unreviewed. Scores are provisional.",
        f"Runs: {runs['attempted']} attempted, {runs['scored']} scored, failed {runs['failed'] or 0}",
        markdown_summary(report),
    ]
    latency = performance["latency"]
    lines.append(
        f"Latency ({latency['basis']}): median {latency['median_ms']} ms, max {latency['max_ms']} ms"
    )
    cost = performance["cost"]
    lines.append(
        f"Cost: {cost}"
        if isinstance(cost, str)
        else f"Estimated cost: ${cost['estimated_usd']:.6f} over {cost['runs_estimated']} runs "
        f"({cost['runs_unavailable']} unavailable); {cost['basis']}"
    )
    for item in report["summary"]["unresolved"]:
        lines.append(
            f"UNRESOLVED {item['document_version_id']}: {item['claim_text']} ({item['reason']})"
        )
    for failure in report["failures"]:
        lines.append(
            f"FAILED {failure['document_version_id']} rep {failure['rep']}: {failure['status']}"
        )
    return "\n".join(lines)
