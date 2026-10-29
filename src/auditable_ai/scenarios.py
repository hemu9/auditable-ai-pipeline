"""Synthetic validator regression cases, never a model-accuracy benchmark."""

from __future__ import annotations

import json
from datetime import datetime
from importlib.resources import files
from typing import Any

from auditable_ai.claims import ClaimRepository
from auditable_ai.documents import (
    InMemorySourceChunkRepository,
    InMemorySourceDocumentRepository,
    RawDocumentContent,
    SourceDocument,
)
from auditable_ai.extraction import EvidenceExtractor, ExtractionResult


def fixture_data() -> dict[str, Any]:
    return json.loads(
        files("auditable_ai").joinpath("fixtures/scenarios.json").read_text(encoding="utf-8")
    )


def repositories(data: dict[str, Any]) -> tuple[InMemorySourceDocumentRepository, ClaimRepository]:
    documents = InMemorySourceDocumentRepository()
    for entry in data["documents"]:
        content = RawDocumentContent(entry["text"])
        recorded = datetime.fromisoformat(entry["recorded_at"])
        documents.add(
            SourceDocument(
                document_version_id=entry["document_version_id"],
                source_id=entry["source_id"],
                source_type="synthetic_operational_report",
                document_location="synthetic://sensor-pilot",
                raw_content=content,
                content_hash=content.content_hash(),
                publicly_available_at=datetime.fromisoformat(entry["publicly_available_at"]),
                ingested_at=recorded,
                recorded_at=recorded,
                metadata={"synthetic": True},
            )
        )
    return documents, ClaimRepository(documents, InMemorySourceChunkRepository(documents))


def run_case(data: dict[str, Any], case: dict[str, Any]) -> ExtractionResult:
    documents, claims = repositories(data)
    document = documents.get_version(case["document_version_id"])
    if document is None:
        raise ValueError("scenario names an unknown input document")
    extractor = EvidenceExtractor(claims=claims, extraction_fn=lambda _content: case["response"])
    return extractor.extract(
        document,
        source_as_of=datetime.fromisoformat(case.get("source_as_of", "2026-04-21T12:00:00+00:00")),
        extracted_at=datetime.fromisoformat("2026-04-21T12:00:00+00:00"),
        subject_id="sensor-pilot",
        run_id=case["name"],
        model_name="scripted-fixture",
        model_version="1",
        prompt_version="not-applicable",
    )


def check_cases(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Check declared validator behavior; do not score an LLM or investment rule."""
    checks = []
    for case in data["cases"]:
        expected = case["expected"]
        try:
            result = run_case(data, case)
        except (ValueError, TypeError) as exc:
            passed = type(exc).__name__ == expected.get("error")
            observed = {"error": type(exc).__name__}
        else:
            observed = {"accepted": len(result.claims), "rejected": len(result.rejected_items)}
            passed = "error" not in expected and all(
                observed[key] == expected[key] for key in observed
            )
            if "unknowns" in expected:
                passed = passed and len(result.unknowns) == expected["unknowns"]
            if "reason" in expected:
                passed = passed and any(
                    item.reason.value == expected["reason"] for item in result.rejected_items
                )
            if "accuracy_status" in expected:
                passed = (
                    passed and result.to_dict()["accuracy_status"] == expected["accuracy_status"]
                )
        checks.append({"case": case["name"], "passed": passed, "observed": observed})
    return checks
