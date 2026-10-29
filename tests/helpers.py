from copy import deepcopy
from datetime import datetime

from auditable_ai.documents import RawDocumentContent, SourceChunk
from auditable_ai.extraction import EvidenceExtractor
from auditable_ai.scenarios import fixture_data, repositories


def at(date: str) -> datetime:
    return datetime.fromisoformat(f"2026-04-{date}T12:00:00+00:00")


def setup():
    data = fixture_data()
    documents, claims = repositories(data)
    return data, documents, claims


def chunk(chunk_id, document_version_id, text, *, public, recorded):
    raw = RawDocumentContent(text)
    return SourceChunk(
        chunk_id,
        document_version_id,
        "sentence 1",
        raw,
        raw.content_hash(),
        at(public),
        at(recorded),
        at(recorded),
    )


def extract(
    response,
    *,
    document_id="pilot-v2",
    source_as_of=None,
    extracted_at=None,
    run_id="test-run",
    subject_id="sensor-pilot",
    bundle=None,
    calls=None,
):
    data, documents, claims = setup() if bundle is None else bundle
    document = documents.get_version(document_id)

    def adapter(raw):
        if calls is not None:
            calls.append(raw)
        return response

    result = EvidenceExtractor(claims=claims, extraction_fn=adapter).extract(
        document,
        source_as_of=source_as_of or at("21"),
        extracted_at=extracted_at or at("21"),
        subject_id=subject_id,
        run_id=run_id,
        model_name="scripted-fixture",
        model_version="1",
        prompt_version="not-applicable",
    )
    return result, documents, claims


def valid_response():
    return deepcopy(fixture_data()["cases"][0]["response"])
