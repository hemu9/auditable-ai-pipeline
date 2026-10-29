"""Register a document version, then validate untrusted extraction output against it.

Runs as-is after `pip install -e .`, with no credentials and no network.
`scripted_model` stands in for a model call; its output is treated as untrusted.
"""

from datetime import datetime

from auditable_ai import (
    ClaimRepository,
    EvidenceExtractor,
    InMemorySourceChunkRepository,
    InMemorySourceDocumentRepository,
    RawDocumentContent,
    SourceDocument,
)

content = RawDocumentContent("Depot 4 inspected 62 buses; 57 passed. Brake wear was not measured.")
recorded = datetime.fromisoformat("2026-06-15T09:00:00+00:00")
report = SourceDocument(
    document_version_id="depot-report-v1",
    source_id="depot-report",
    source_type="synthetic_report",
    document_location="example://depot-report",
    raw_content=content,
    content_hash=content.content_hash(),
    publicly_available_at=datetime.fromisoformat("2026-06-14T17:00:00+00:00"),
    ingested_at=recorded,
    recorded_at=recorded,
)
documents = InMemorySourceDocumentRepository()
documents.add(report)
claims = ClaimRepository(documents, InMemorySourceChunkRepository(documents))


def scripted_model(document: RawDocumentContent) -> dict:
    return {
        "facts": [
            {
                "claim_text": "57 of the 62 inspected buses passed.",
                "supporting_evidence": [{"source_id": "depot-report-v1"}],
            },
            {
                "claim_text": "A regional audit confirmed the brakes are safe.",
                "supporting_evidence": [{"source_id": "regional-audit"}],
            },
        ],
        "unknowns": ["Brake wear"],
        "source_ids": ["depot-report-v1", "regional-audit"],
    }


result = EvidenceExtractor(claims=claims, extraction_fn=scripted_model).extract(
    report,
    source_as_of=recorded,
    extracted_at=recorded,
    subject_id="depot-report",
    run_id="quickstart-1",
    model_name="scripted-example",
    model_version="1",
    prompt_version="none",
)
for claim in result.claims:
    print(f"ACCEPTED  {claim.claim_text} (accuracy NOT_VERIFIED)")
for item in result.rejected_items:
    print(f"REJECTED  {item.reason.value}: {item.detail}")
for unknown in result.unknowns:
    print(f"UNKNOWN   {unknown}")
print(
    f"Audit record: response sha256 {result.response_hash[:12]}, validator v{result.validator_version}"
)
