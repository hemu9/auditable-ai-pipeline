"""Mutation check: show that the test suite catches the loss of each stated guarantee.

Each entry in MUTATIONS disables one guarantee from the README or
docs/architecture.md by replacing exact source text. The check copies the
repository once per mutation, applies that mutation to the copy and runs the
whole test suite against it. A mutation that leaves the suite passing has
survived: no test protects that guarantee. The check passes only when every
mutation is caught.

    python tools/mutation_check.py              run every mutation
    python tools/mutation_check.py --only C9    run selected mutations
    python tools/mutation_check.py --list       print the catalog without running it

When a change moves or rewrites code that a mutation targets, the check stops
and names the stale entry; update its find and replace text to match."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = Path("src", "auditable_ai")
COPIED = ("src", "tests", "docs", "examples", "tools", "README.md")
SUITE_TIMEOUT_S = 600


@dataclass(frozen=True)
class Mutation:
    id: str
    file: str  # relative to src/auditable_ai
    find: str  # must occur exactly once in that file
    replace: str
    guarantee: str


MUTATIONS = (
    # Historical queries
    Mutation(
        "T1",
        "temporal.py",
        "timestamp_for_mode(v, mode) <= as_of]",
        "timestamp_for_mode(v, mode) < as_of]",
        "an as-of cutoff includes versions stamped exactly at the cutoff",
    ),
    Mutation(
        "T2",
        "temporal.py",
        "    if len(winners) > 1:",
        "    if False:",
        "a tie for the latest version raises AmbiguousRevisionError instead of picking one",
    ),
    Mutation(
        "T3",
        "temporal.py",
        "    if mode is AsOfMode.RESEARCH:\n        return record.publicly_available_at",
        "    if mode is AsOfMode.RESEARCH:\n        return record.recorded_at",
        "RESEARCH_AS_OF selects versions by when they became public",
    ),
    Mutation(
        "T4",
        "temporal.py",
        "    max_ts = max((timestamp_fn(v) for v in candidates))",
        "    max_ts = min((timestamp_fn(v) for v in candidates))",
        "the latest eligible version is selected",
    ),
    Mutation(
        "T5",
        "temporal.py",
        "    if dependent_recorded_at < dependency_recorded_at:",
        "    if False:",
        "nothing is recorded before the evidence it depends on",
    ),
    Mutation(
        "T6",
        "temporal.py",
        "    if dependent_publicly_available_at < dependency_publicly_available_at:",
        "    if False:",
        "nothing is public before the evidence it depends on",
    ),
    Mutation(
        "T7",
        "temporal.py",
        "    if superseded_in_system_at is not None and superseded_in_system_at < recorded_at:",
        "    if False:",
        "a version cannot be superseded in the system before it was recorded",
    ),
    Mutation(
        "T8",
        "temporal.py",
        "    if value is not None and value.tzinfo is None:",
        "    if False:",
        "every timestamp must carry a UTC offset",
    ),
    Mutation(
        "T9",
        "temporal.py",
        "        scoped = [r for r in self._records if predicate(r)]",
        "        scoped = list(self._records)",
        "a scoped query resolves versions within its scope only",
    ),
    Mutation(
        "T10",
        "temporal.py",
        "    if value > now:",
        "    if False:",
        "a recording time later than the writer's clock is refused",
    ),
    # Documents and chunks
    Mutation(
        "D1",
        "documents.py",
        "        if self.content_hash != self.raw_content.content_hash():",
        "        if False:",
        "a document's text must match its content hash",
    ),
    Mutation(
        "D2",
        "documents.py",
        '        object.__setattr__(self, "metadata", deep_freeze(self.metadata))',
        '        object.__setattr__(self, "metadata", self.metadata)',
        "document metadata is deep-frozen",
    ),
    Mutation(
        "D3",
        "documents.py",
        "        if self.content_hash != self.raw_text.content_hash():",
        "        if False:",
        "a chunk's text must match its content hash",
    ),
    Mutation(
        "D4",
        "documents.py",
        "        if document.document_version_id in self._by_version:",
        "        if False:",
        "a document version identity is never reused (in memory)",
    ),
    Mutation(
        "D5",
        "documents.py",
        "        return self._lineage(source_id, as_of, AsOfMode.OPERATIONAL_REPLAY)",
        "        return self._lineage(source_id, as_of, AsOfMode.RESEARCH)",
        "replaying one lineage selects by recording time (in memory)",
    ),
    Mutation(
        "D6",
        "documents.py",
        "        if chunk.chunk_id in self._chunks:",
        "        if False:",
        "a chunk identity is never reused (in memory)",
    ),
    Mutation(
        "D7",
        "documents.py",
        "    require_dependent_not_before_dependency(\n        dependent_publicly_available_at=chunk.publicly_available_at,",
        "    (lambda **kw: None)(\n        dependent_publicly_available_at=chunk.publicly_available_at,",
        "a chunk cannot be public or recorded before its document",
    ),
    Mutation(
        "D8",
        "documents.py",
        "    if not chunk.raw_text.text or chunk.raw_text.text not in document.raw_content.text:",
        "    if False:",
        "a chunk must quote text from the version it names",
    ),
    Mutation(
        "D9",
        "documents.py",
        "    if not chunk.raw_text.text or chunk.raw_text.text not in document.raw_content.text:",
        "    if chunk.raw_text.text not in document.raw_content.text:",
        "an empty chunk is refused",
    ),
    Mutation(
        "D10",
        "documents.py",
        '    require_not_in_future(f"SourceChunk {chunk.chunk_id!r} recorded_at", chunk.recorded_at, clock)',
        "    pass",
        "a chunk recorded after the clock is refused",
    ),
    Mutation(
        "D11",
        "documents.py",
        '        require_not_in_future(\n            f"document version {document.document_version_id!r} recorded_at",\n            document.recorded_at,\n            self._clock,\n        )\n        self._store.add(document)',
        "        self._store.add(document)",
        "a document recorded after the clock is refused (in memory)",
    ),
    Mutation(
        "D12",
        "documents.py",
        '            raise ProvenanceError(\n                f"content_hash {self.content_hash!r} does not match the actual hash of raw_content',
        '            raise ValueError(\n                f"content_hash {self.content_hash!r} does not match the actual hash of raw_content',
        "a document hash mismatch raises ProvenanceError",
    ),
    Mutation(
        "D13",
        "documents.py",
        "        validate_chunk_registration(\n            chunk, self._documents.get_version(chunk.document_version_id), self._clock\n        )\n        self._chunks[chunk.chunk_id] = chunk",
        "        self._chunks[chunk.chunk_id] = chunk",
        "the in-memory chunk store applies the shared registration checks",
    ),
    # JSONL storage
    Mutation(
        "S1",
        "storage.py",
        "            [d for d in self._read_all() if d.source_id == source_id],",
        "            self._read_all(),",
        "a lineage query reads only its own source (JSONL)",
    ),
    Mutation(
        "S2",
        "storage.py",
        "        return self._lineage(source_id, as_of, AsOfMode.OPERATIONAL_REPLAY)",
        "        return self._lineage(source_id, as_of, AsOfMode.RESEARCH)",
        "replaying one lineage selects by recording time (JSONL)",
    ),
    Mutation(
        "S3",
        "storage.py",
        "        if any((d.document_version_id == document.document_version_id for d in self._read_all())):",
        "        if False:",
        "a document version identity is never reused (JSONL)",
    ),
    Mutation(
        "S4",
        "storage.py",
        "        if any((c.chunk_id == chunk.chunk_id for c in self._read_all())):",
        "        if False:",
        "a chunk identity is never reused (JSONL)",
    ),
    Mutation(
        "S5",
        "storage.py",
        '        "metadata": to_plain(doc.metadata),',
        '        "metadata": {},',
        "a JSONL round trip preserves every field",
    ),
    Mutation(
        "S6",
        "storage.py",
        "        validate_chunk_registration(\n            chunk, self._documents.get_version(chunk.document_version_id), self._clock\n        )\n        _append(self._path, _serialize_chunk(chunk))",
        "        _append(self._path, _serialize_chunk(chunk))",
        "the JSONL chunk store applies the shared registration checks",
    ),
    Mutation(
        "S7",
        "storage.py",
        '    with path.open("a", encoding="utf-8") as f:',
        '    with path.open("w", encoding="utf-8") as f:',
        "JSONL writes append and never truncate earlier rows",
    ),
    Mutation(
        "S8",
        "storage.py",
        '        require_not_in_future(\n            f"document version {document.document_version_id!r} recorded_at",\n            document.recorded_at,\n            self._clock,\n        )\n        _append(self._path, _serialize(document))',
        "        _append(self._path, _serialize(document))",
        "a document recorded after the clock is refused (JSONL)",
    ),
    # Claims
    Mutation(
        "C1",
        "claims.py",
        "            if chunk is None or chunk.document_version_id != reference.source_id:",
        "            if chunk is None:",
        "a cited chunk belongs to the cited document version",
    ),
    Mutation(
        "C2",
        "claims.py",
        "                dependency_publicly_available_at=chunk.publicly_available_at,\n                dependency_recorded_at=chunk.recorded_at,",
        "                dependency_publicly_available_at=document.publicly_available_at,\n                dependency_recorded_at=document.recorded_at,",
        "a cited chunk's own timing is checked",
    ),
    Mutation(
        "C3",
        "claims.py",
        '            raise ProvenanceError("supplied evidence hash does not match the referenced content")',
        "            pass",
        "a supplied evidence hash must match the cited content",
    ),
    Mutation(
        "C4",
        "claims.py",
        "            if claim.claim_id in self._by_id or claim.claim_id in seen:",
        "            if False:",
        "a claim identity is never overwritten",
    ),
    Mutation(
        "C5",
        "claims.py",
        "        for reference in (*claim.supporting_evidence, *claim.counterevidence):",
        "        for reference in claim.supporting_evidence:",
        "counterevidence is checked like supporting evidence",
    ),
    Mutation(
        "C6",
        "claims.py",
        "            self._store.as_of_matching(as_of, mode, lambda claim: claim.subject_id == subject_id)",
        "            self._store.as_of_matching(as_of, mode, lambda claim: True)",
        "claim queries are scoped to their subject",
    ),
    Mutation(
        "C7",
        "claims.py",
        "        return self._for(subject_id, as_of, AsOfMode.RESEARCH)",
        "        return self._for(subject_id, as_of, AsOfMode.OPERATIONAL_REPLAY)",
        "claim research and operational-replay queries use different times",
    ),
    Mutation(
        "C8",
        "claims.py",
        "        if self.category is ClaimCategory.FACT and not self.supporting_evidence:",
        "        if False:",
        "a FACT cannot be constructed without a source reference",
    ),
    Mutation(
        "C9",
        "claims.py",
        "        if claim.category is ClaimCategory.FACT and not claim.supporting_evidence:",
        "        if False:",
        "a FACT is never stored without a supporting reference",
    ),
    Mutation(
        "C10",
        "claims.py",
        '            raise ProvenanceError("claim evidence must be a tuple of EvidenceReference values")',
        "            pass",
        "stored evidence must be a tuple of references",
    ),
    Mutation(
        "C11",
        "claims.py",
        "object.__setattr__(self, name, _normalized(name, getattr(self, name), item_type))",
        "object.__setattr__(self, name, tuple(getattr(self, name)))",
        "claim collections are materialized before they are validated",
    ),
    Mutation(
        "C12",
        "claims.py",
        "    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Iterable):",
        "    if not isinstance(value, Iterable):",
        "a string is not accepted as a collection",
    ),
    Mutation(
        "C13",
        "claims.py",
        "    if not all(isinstance(item, item_type) for item in items):",
        "    if False:",
        "claim collections contain only values of the declared type",
    ),
    Mutation(
        "C14",
        "claims.py",
        "        if self.has_run(run_id):",
        "        if False:",
        "a run identity is committed at most once",
    ),
    Mutation(
        "C15",
        "claims.py",
        "            self.validate_evidence(claim)\n        for claim in claims:\n            self._store.add(claim)\n            self._by_id[claim.claim_id] = claim",
        "            self.validate_evidence(claim)\n            self._store.add(claim)\n            self._by_id[claim.claim_id] = claim\n        for claim in ():\n            pass",
        "a run's claims are committed together or not at all",
    ),
    Mutation(
        "C16",
        "claims.py",
        '            require_not_in_future(\n                f"claim {claim.claim_id!r} recorded_at", claim.recorded_at, self.clock\n            )',
        "            pass",
        "a claim recorded after the clock is refused",
    ),
    Mutation(
        "C17",
        "claims.py",
        "        self._run_ids.add(run_id)",
        "        pass",
        "a committed run identity is remembered",
    ),
    Mutation(
        "C18",
        "claims.py",
        "            if claim.run_id != run_id:",
        "            if False:",
        "every claim in a run commit belongs to that run",
    ),
    Mutation(
        "C19",
        "claims.py",
        "            if claim.claim_id in self._by_id or claim.claim_id in seen:",
        "            if claim.claim_id in self._by_id:",
        "a run commit cannot repeat a claim identity",
    ),
    # Extraction boundary
    Mutation(
        "E1",
        "extraction.py",
        "        if registered != document:",
        "        if False:",
        "the extraction input must be the registered document version",
    ),
    Mutation(
        "E2",
        "extraction.py",
        "        if document.publicly_available_at > source_as_of:",
        "        if False:",
        "the input must be public at source_as_of",
    ),
    Mutation(
        "E3",
        "extraction.py",
        "        if extracted_at < document.recorded_at or extracted_at < document.publicly_available_at:",
        "        if False:",
        "an extraction cannot precede the input's recording or availability",
    ),
    Mutation(
        "E4",
        "extraction.py",
        "        if not isinstance(parsed, dict) or set(parsed) - TOP_FIELDS:",
        "        if not isinstance(parsed, dict):",
        "an unknown top-level output field fails the whole batch",
    ),
    Mutation(
        "E5",
        "extraction.py",
        "                    if set(item) - ITEM_FIELDS:",
        "                    if False:",
        "an unknown item field rejects the item",
    ),
    Mutation(
        "E6",
        "extraction.py",
        "    if set(raw) - REF_FIELDS:",
        "    if False:",
        "an unknown citation field rejects the item",
    ),
    Mutation(
        "E7",
        "extraction.py",
        "    if not isinstance(source_id, str) or not source_id or source_id not in declared:",
        "    if not isinstance(source_id, str) or not source_id:",
        "a citation must name a declared source",
    ),
    Mutation(
        "E8",
        "extraction.py",
        "        for reference in contradicting:",
        "        for reference in ():",
        "top-level counterevidence is checked against the repositories",
    ),
    Mutation(
        "E9",
        "extraction.py",
        "                    if category is ClaimCategory.FACT and not supporting:",
        "                    if False:",
        "a FACT without a citation is attributed to the input document",
    ),
    Mutation(
        "E10",
        "extraction.py",
        "            parsed = json.loads(raw_json)",
        "            parsed = raw",
        "validation reads the canonical JSON snapshot, not the adapter's objects",
    ),
    Mutation(
        "E11",
        "extraction.py",
        "                        counterevidence=counterevidence,",
        "                        counterevidence=(),",
        "valid per-claim counterevidence is kept",
    ),
    Mutation(
        "E12",
        "extraction.py",
        "                        confidence=item_confidence,",
        "                        confidence=None,",
        "a valid confidence is kept",
    ),
    Mutation(
        "E13",
        "extraction.py",
        "        if self.claims.has_run(run_id):",
        "        if False:",
        "a reused run identity is refused before the adapter runs",
    ),
    Mutation(
        "E14",
        "extraction.py",
        '        require_not_in_future("extracted_at", extracted_at, self.claims.clock)',
        "        pass",
        "a future extraction time is refused before the adapter runs",
    ),
    Mutation(
        "E15",
        "extraction.py",
        "        except (TypeError, ValueError, RecursionError) as exc:",
        "        except (TypeError, ValueError) as exc:",
        "output nested too deeply to snapshot fails the batch instead of crashing",
    ),
    Mutation(
        "E16",
        "extraction.py",
        '            parsed = json.loads(raw_json)\n            response_hash = canonical_hash(parsed)\n        except (TypeError, ValueError, RecursionError) as exc:\n            raise ExtractionValidationError("adapter output must be finite JSON data") from exc\n',
        '            parsed = json.loads(raw_json)\n        except (TypeError, ValueError, RecursionError) as exc:\n            raise ExtractionValidationError("adapter output must be finite JSON data") from exc\n        response_hash = canonical_hash(parsed)\n',
        "the response fingerprint is computed inside the guarded snapshot",
    ),
    Mutation(
        "E17",
        "extraction.py",
        "        self.claims.add_run(run_id, accepted)",
        "        pass",
        "accepted claims are committed",
    ),
    Mutation(
        "E18",
        "extraction.py",
        "                except ExtractionValidationError as exc:",
        "                except (ExtractionValidationError, ValueError) as exc:",
        "a storage failure propagates instead of becoming an item rejection",
    ),
    Mutation(
        "E19",
        "extraction.py",
        "                except _UnknownFieldError as exc:",
        "                except ArithmeticError as exc:",
        "an unknown item field is classified UNKNOWN_FIELD",
    ),
    Mutation(
        "E20",
        "extraction.py",
        '        raise _UnknownFieldError("evidence reference contains unknown fields")',
        '        raise ExtractionValidationError("evidence reference contains unknown fields")',
        "an unknown citation field is classified UNKNOWN_FIELD",
    ),
    # Claude adapter
    Mutation(
        "A1",
        "anthropic_adapter.py",
        '    if interpretation.stop_reason == "refusal":',
        "    if False:",
        "a refusal is classified, not parsed",
    ),
    Mutation(
        "A2",
        "anthropic_adapter.py",
        '    if interpretation.stop_reason == "max_tokens":',
        "    if False:",
        "a truncated response is classified, not parsed",
    ),
    Mutation(
        "A3",
        "anthropic_adapter.py",
        "                if not exc.retryable or attempt == self.settings.max_attempts:",
        "                if True:",
        "retryable provider errors are retried",
    ),
    Mutation(
        "A4",
        "anthropic_adapter.py",
        "                self._sleep(min(max(backoff, exc.retry_after_s or 0.0), 60.0))",
        "                self._sleep(min(backoff, 60.0))",
        "a provider's retry-after is respected",
    ),
    Mutation(
        "A5",
        "anthropic_adapter.py",
        '    return _SECRETS.sub("[REDACTED]", text)',
        "    return text",
        "credentials are redacted from recorded messages",
    ),
    Mutation(
        "A6",
        "anthropic_adapter.py",
        "    if settings.fallbacks:",
        "    if True:",
        "server-side fallbacks can be turned off",
    ),
    Mutation(
        "A7",
        "anthropic_adapter.py",
        '    return _text(side.get("model")) if isinstance(side, dict) else None',
        '    return side.get("model")',
        "a fallback block of an unexpected shape reads as absent instead of raising",
    ),
    Mutation(
        "A8",
        "anthropic_adapter.py",
        '        served_model=_text(body.get("model")),',
        '        served_model=body.get("model"),',
        "a served model that is not a string reads as absent",
    ),
    Mutation(
        "A9",
        "anthropic_adapter.py",
        '    except (ValueError, RecursionError) as exc:\n        raise ModelOutputError("unexpected_response"',
        '    except ValueError as exc:\n        raise ModelOutputError("unexpected_response"',
        "a response body nested too deeply to parse is classified, not raised",
    ),
    Mutation(
        "A10",
        "anthropic_adapter.py",
        '    except (ValueError, RecursionError) as exc:\n        raise ModelOutputError("invalid_json"',
        '    except ValueError as exc:\n        raise ModelOutputError("invalid_json"',
        "output text nested too deeply to parse is classified, not raised",
    ),
    Mutation(
        "A11",
        "_immutable.py",
        "    return json.loads(text, parse_constant=_refuse_constant)",
        "    return json.loads(text)",
        "responses and record files must be standard JSON: NaN and Infinity are refused",
    ),
    Mutation(
        "A12",
        "anthropic_adapter.py",
        "            if isinstance(value, bool) or not isinstance(value, int) or value < 1:",
        "            if False:",
        "token limits and attempt counts must be positive integers",
    ),
    Mutation(
        "A13",
        "anthropic_adapter.py",
        '            raise ValueError("timeout_s must be a positive number of seconds")',
        "            pass",
        "the request timeout must be a positive number of seconds",
    ),
    # Run records and replay
    Mutation(
        "R1",
        "runs.py",
        '    if sha256_text(raw_response) != exchange.get("response_sha256"):',
        "    if False:",
        "replay refuses a response that does not match its recorded hash",
    ),
    Mutation(
        "R2",
        "runs.py",
        '    if RawDocumentContent(document.text).content_hash() != record["document"]["content_hash"]:',
        "    if False:",
        "replay refuses a document that does not match its recorded hash",
    ),
    Mutation(
        "R3",
        "runs.py",
        '    with path.open("x", encoding="utf-8") as stream:',
        '    with path.open("w", encoding="utf-8") as stream:',
        "a run record is never overwritten",
    ),
    Mutation(
        "R4",
        "runs.py",
        "interpretation.served_model == settings.model if interpretation else None",
        "True if interpretation else None",
        "a response from a model other than the one requested is flagged",
    ),
    Mutation(
        "R5",
        "runs.py",
        '    text = json.dumps(written, indent=2, allow_nan=False) + "\\n"\n    with path.open("x", encoding="utf-8") as stream:\n        stream.write(text)',
        '    with path.open("x", encoding="utf-8") as stream:\n        stream.write(json.dumps(written, indent=2, allow_nan=False) + "\\n")',
        "a record that cannot be serialized leaves no partial file",
    ),
    Mutation(
        "R6",
        "runs.py",
        "        elif not _KINDS[expected](value[name]):",
        "        elif False:",
        "reading a record checks the type of every field that replay and evaluation use",
    ),
    Mutation(
        "R7",
        "runs.py",
        "        if name not in value:",
        "        if False:",
        "reading a record refuses one with a field missing",
    ),
    Mutation(
        "R8",
        "runs.py",
        '    if record.get("exchange") is not None:\n        _check_fields(',
        "    if False:\n        _check_fields(",
        "reading a record checks the provider exchange it holds",
    ),
    Mutation(
        "R9",
        "runs.py",
        '    if not isinstance(record, dict) or record.get("record_type") != RECORD_TYPE:',
        "    if not isinstance(record, dict):",
        "a file that is not a run record is refused",
    ),
    Mutation(
        "R10",
        "runs.py",
        '    if record.get("record_version") != RECORD_VERSION:',
        "    if False:",
        "a record of an unsupported version is refused",
    ),
    Mutation(
        "R11",
        "runs.py",
        '        record: dict[str, Any] = strict_loads(path.read_text(encoding="utf-8"))',
        '        record: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))',
        "a record file must be standard JSON",
    ),
    Mutation(
        "R12",
        "runs.py",
        '            run_id=record["run_id"],',
        "            run_id=f\"{record['run_id']}--replay\",",
        "replay runs under the recorded run identity",
    ),
    Mutation(
        "R13",
        "runs.py",
        '    return {key: value for key, value in result.items() if key != "latency_ms"}',
        "    return result",
        "replay must reproduce the stored result exactly, apart from latency",
    ),
    Mutation(
        "R14",
        "runs.py",
        '    if record["status"] != status:',
        "    if False:",
        "replay reports a status that differs from the stored one",
    ),
    Mutation(
        "R15",
        "runs.py",
        '    check_record(record)\n    document = document_from_entry(record["document"])',
        '    document = document_from_entry(record["document"])',
        "replay checks a record's shape itself",
    ),
    Mutation(
        "R16",
        "runs.py",
        '    require_timezone_aware("clock()", started)',
        "    pass",
        "a run is refused before any call when its clock has no UTC offset",
    ),
    # Cost estimates
    Mutation(
        "P1",
        "pricing.py",
        '    if stop_reason == "refusal":',
        "    if False:",
        "a refusal's cost is reported as unavailable",
    ),
    Mutation(
        "P2",
        "pricing.py",
        '    attempts = usage.get("iterations") or [dict(usage, model=served_model)]',
        "    attempts = [dict(usage, model=served_model)]",
        "each attempt is priced at the model that ran it",
    ),
    Mutation(
        "P3",
        "pricing.py",
        "    if isinstance(value, bool) or not isinstance(value, int) or value < 0:",
        "    if False:",
        "a token count must be a non-negative integer",
    ),
    Mutation(
        "P4",
        "pricing.py",
        "    if not isinstance(attempts, list) or not all(isinstance(a, Mapping) for a in attempts):",
        "    if False:",
        "usage iterations must be a list of objects",
    ),
    Mutation(
        "P5",
        "pricing.py",
        "    except (_MalformedUsage, OverflowError) as exc:",
        "    except _MalformedUsage as exc:",
        "a count too large to convert makes the estimate unavailable instead of raising",
    ),
    Mutation(
        "P6",
        "pricing.py",
        "    if not math.isfinite(total):",
        "    if False:",
        "a total too large to represent makes the estimate unavailable",
    ),
    Mutation(
        "P7",
        "pricing.py",
        "            prices = PRICES_PER_MTOK.get(model) if isinstance(model, str) else None",
        "            prices = PRICES_PER_MTOK.get(model)",
        "only a model name that is a string can be priced",
    ),
    Mutation(
        "P8",
        "pricing.py",
        "            if not isinstance(durations, Mapping):",
        "            if False:",
        "cache-write durations must be an object",
    ),
    # Evaluation
    Mutation(
        "V1",
        "evaluation.py",
        '        elif replay["status"] == "completed" and served_as_requested is False:',
        "        elif False:",
        "a response from another model is listed but never scored",
    ),
    Mutation(
        "V2",
        "evaluation.py",
        "        except ValueError as exc:  # ProvenanceError or TemporalIntegrityError: altered content",
        "        except ArithmeticError as exc:",
        "a record whose content fails its hashes becomes an integrity failure",
    ),
    Mutation(
        "V3",
        "evaluation.py",
        "    if numbers & {trap.value for trap in document.traps}:",
        "    if False:",
        "a claim reporting a trap number is labeled TRAP",
    ),
    Mutation(
        "V4",
        "evaluation.py",
        "        matching = [figure for figure in in_cited if figure.status == status]",
        "        matching = in_cited",
        "a claim with the wrong status is labeled WRONG_STATUS",
    ),
    Mutation(
        "V5",
        "evaluation.py",
        '        if any(figure.status == "ambiguous" for figure in in_cited):',
        "        if False:",
        "a figure annotated as ambiguous stays unresolved",
    ),
    Mutation(
        "V6",
        "evaluation.py",
        "    if absent:",
        "    if False:",
        "a number absent from the document is labeled UNSUPPORTED_VALUE",
    ),
    Mutation(
        "V7",
        "evaluation.py",
        "    elsewhere = [figure.id for figure in document.figures if figure.value in numbers]",
        "    elsewhere = []",
        "a figure cited from the wrong sentence is labeled CITATION_MISMATCH",
    ),
    Mutation(
        "V8",
        "evaluation.py",
        '            "label": SUPPORTED if new else DUPLICATE,',
        '            "label": SUPPORTED,',
        "a repeated figure is labeled DUPLICATE",
    ),
    Mutation(
        "V9",
        "evaluation.py",
        '    if model and set(origins) != {"recorded_model_response"}:',
        "    if False:",
        "a report says when its responses did not come from a real API call",
    ),
    Mutation(
        "V10",
        "evaluation.py",
        "        values = [v for v in numbers_in(sentence) if not (v.is_integer() and 1900 <= v <= 2100)]",
        "        values = list(numbers_in(sentence))",
        "the published baseline table matches the scorer",
    ),
    Mutation(
        "V11",
        "evaluation.py",
        "    for record in records:\n        check_record(record)",
        "    for record in records:\n        pass",
        "evaluation refuses a malformed record instead of scoring it",
    ),
    Mutation(
        "V12",
        "evaluation.py",
        "        return f\"the record reports status {record['status']!r} but holds no provider response\"",
        "        return None",
        "a record that claims an outcome without a provider response is an integrity failure",
    ),
    Mutation(
        "V13",
        "evaluation.py",
        '        if record["status"] == "provider_error":\n            return None',
        "        if False:\n            return None",
        "a run that never received a response is listed as a provider error",
    ),
    Mutation(
        "V14",
        "evaluation.py",
        '    if record["validator_version"] == VALIDATOR_VERSION and replay["differences"]:',
        "    if False:",
        "a stored outcome the same validator does not reproduce is an integrity failure",
    ),
    Mutation(
        "V15",
        "evaluation.py",
        '    if record["served_model_matches_request"] != served_as_requested:',
        "    if False:",
        "a served-model flag that contradicts the response is an integrity failure",
    ),
    Mutation(
        "V16",
        "evaluation.py",
        '            entry["usage"] = response["usage"]',
        '            entry["usage"] = (record.get("response") or {}).get("usage")',
        "token usage is read from the replayed response",
    ),
    Mutation(
        "V17",
        "evaluation.py",
        '            entry["served_model"] = response["served_model"]',
        '            entry["served_model"] = (record.get("response") or {}).get("served_model")',
        "the served model is read from the replayed response",
    ),
    Mutation(
        "V18",
        "evaluation.py",
        '            entry["cost"] = estimate_cost(\n                response["usage"], response["served_model"], stop_reason=response["stop_reason"]\n            )',
        '            entry["cost"] = record.get("cost")',
        "cost is recomputed from the replayed response",
    ),
    Mutation(
        "V19",
        "evaluation.py",
        "    if not all(isinstance(value, int) and not isinstance(value, bool) for value in values):",
        "    if False:",
        "token totals count only integer usage",
    ),
    Mutation(
        "V20",
        "evaluation.py",
        "    if min(values) < 0:",
        "    if False:",
        "token totals exclude negative counts",
    ),
    Mutation(
        "V21",
        "evaluation.py",
        "        f\"{report['dataset']} | split={report['split']} ({runs['documents']} documents) | \"",
        "        f\"{report['task_version']} | split={report['split']} ({runs['documents']} documents) | \"",
        "a report names the dataset it scored",
    ),
    # Dataset
    Mutation(
        "Q1",
        "dataset.py",
        "            if number not in annotated:",
        "            if False:",
        "every number in a dataset document is annotated",
    ),
    Mutation(
        "Q2",
        "dataset.py",
        "            if chunk_id in texts and figure.value not in numbers_in(texts[chunk_id]):",
        "            if False:",
        "a gold value appears in the sentence it cites",
    ),
    Mutation(
        "Q3",
        "dataset.py",
        "    if name not in DATASETS:\n        raise ValueError(",
        "    if False:\n        raise ValueError(",
        "an unknown dataset name is refused",
    ),
    Mutation(
        "Q4",
        "dataset.py",
        "    for name in DATASETS:\n        try:\n            return load_dataset(name).document(document_version_id)",
        "    for name in (DEFAULT_DATASET,):\n        try:\n            return load_dataset(name).document(document_version_id)",
        "a bundled document is found in whichever dataset holds it",
    ),
    # Command line
    Mutation(
        "L1",
        "cli.py",
        '    with path.open("x", encoding="utf-8") as stream:',
        '    with path.open("w", encoding="utf-8") as stream:',
        "a report file is never overwritten",
    ),
    Mutation(
        "L2",
        "cli.py",
        "    return report, 0 if passed == len(checks) else 1",
        "    return report, 0",
        "check-fixtures exits 1 when a scenario regresses",
    ),
    Mutation(
        "L3",
        "scenarios.py",
        '            passed = type(exc).__name__ == expected.get("error")',
        "            passed = True",
        "an unexpected exception in a scenario counts as a failure",
    ),
    Mutation(
        "L4",
        "cli.py",
        '    if not args.live:\n        print(\n            "extract calls',
        '    if False:\n        print(\n            "extract calls',
        "extract calls the API only with --live",
    ),
    Mutation(
        "L5",
        "cli.py",
        '    if not args.live:\n        print(\n            "evaluate --system live',
        '    if False:\n        print(\n            "evaluate --system live',
        "a live evaluation calls the API only with --live",
    ),
    Mutation(
        "L6",
        "cli.py",
        '    _add_model_options(evaluate, fallbacks="none")',
        '    _add_model_options(evaluate, fallbacks="default")',
        "evaluation keeps server-side fallbacks off by default",
    ),
    Mutation(
        "L7",
        "cli.py",
        "    except OSError as exc:",
        "    except ArithmeticError as exc:",
        "an unwritable report path is a clear error",
    ),
    Mutation(
        "L8",
        "cli.py",
        "    if value < 1:",
        "    if False:",
        "count options must be positive integers",
    ),
    Mutation(
        "L9",
        "cli.py",
        "    if not math.isfinite(value) or value <= 0:",
        "    if value <= 0:",
        "the timeout option must be a finite positive number",
    ),
    Mutation(
        "L10",
        "cli.py",
        '    text = json.dumps(report, indent=2, allow_nan=False) + "\\n"\n    with path.open("x", encoding="utf-8") as stream:\n        stream.write(text)',
        '    with path.open("x", encoding="utf-8") as stream:\n        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\\n")',
        "a report that cannot be serialized leaves no partial file",
    ),
    Mutation(
        "L11",
        "cli.py",
        "    dataset = load_dataset(args.dataset)",
        "    dataset = load_dataset()",
        "evaluate scores the dataset it was asked for",
    ),
)


def stale(mutations: tuple[Mutation, ...] | list[Mutation]) -> list[str]:
    """Entries whose find text no longer occurs exactly once in the current source."""
    problems = []
    for mutation in mutations:
        count = (ROOT / PACKAGE / mutation.file).read_text(encoding="utf-8").count(mutation.find)
        if count != 1:
            problems.append(f"{mutation.id}: find text occurs {count} times in {mutation.file}")
    return problems


def _copy(destination: Path) -> None:
    for name in COPIED:
        source = ROOT / name
        if source.is_dir():
            ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info")
            shutil.copytree(source, destination / name, ignore=ignore)
        else:
            shutil.copy2(source, destination / name)


def _environment(copy: Path) -> dict[str, str]:
    # The copy's sources must win over any installed or editable auditable_ai.
    environment = {**os.environ, "PYTHONPATH": str(copy / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    environment.pop("HYPOTHESIS_PROFILE", None)
    return environment


def _run_suite(copy: Path, *, failfast: bool) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-t", "."]
    if failfast:
        command.append("--failfast")
    return subprocess.run(
        command,
        cwd=copy,
        env=_environment(copy),
        capture_output=True,
        text=True,
        timeout=SUITE_TIMEOUT_S,
    )


def _imports_from(copy: Path) -> Path:
    found = subprocess.run(
        [sys.executable, "-B", "-c", "import auditable_ai; print(auditable_ai.__file__)"],
        cwd=copy,
        env=_environment(copy),
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(found.stdout.strip()).resolve()


def _outcome(mutation: Mutation, scratch: Path) -> str:
    copy = scratch / mutation.id
    _copy(copy)
    target = copy / PACKAGE / mutation.file
    source = target.read_text(encoding="utf-8")
    target.write_text(source.replace(mutation.find, mutation.replace), encoding="utf-8")
    try:
        passed = _run_suite(copy, failfast=True).returncode == 0
    except subprocess.TimeoutExpired:
        return "caught (timed out)"
    finally:
        shutil.rmtree(copy, ignore_errors=True)
    return "SURVIVED" if passed else "caught"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--only", nargs="+", metavar="ID", help="run only these mutations")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--list", action="store_true", help="print the catalog and exit")
    args = parser.parse_args(argv)
    selected = [m for m in MUTATIONS if not args.only or m.id in args.only]
    unknown = sorted(set(args.only or ()) - {m.id for m in MUTATIONS})
    if unknown:
        print(f"unknown mutation ids: {', '.join(unknown)}", file=sys.stderr)
        return 2
    problems = stale(selected)
    if problems:
        print("stale mutation entries; update tools/mutation_check.py:", file=sys.stderr)
        print("\n".join(f"  {problem}" for problem in problems), file=sys.stderr)
        return 2
    if args.list:
        for mutation in selected:
            print(f"{mutation.id:4} {mutation.file:21} {mutation.guarantee}")
        return 0
    with tempfile.TemporaryDirectory(prefix="mutation-check-") as directory:
        scratch = Path(directory)
        baseline = scratch / "baseline"
        _copy(baseline)
        if not _imports_from(baseline).is_relative_to(baseline.resolve()):
            print("the copied suite would not import the copied sources", file=sys.stderr)
            return 2
        unmutated = _run_suite(baseline, failfast=False)
        if unmutated.returncode != 0:
            print("the unmutated suite fails; fix it before checking mutations", file=sys.stderr)
            print(unmutated.stderr[-4000:], file=sys.stderr)
            return 2
        print(f"unmutated suite: {unmutated.stderr.strip().splitlines()[-3]}; passes")
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            outcomes = list(pool.map(lambda mutation: _outcome(mutation, scratch), selected))
    survivors = 0
    for mutation, outcome in zip(selected, outcomes):
        survivors += outcome == "SURVIVED"
        print(f"{outcome:18} {mutation.id:4} {mutation.guarantee}")
    print(f"\n{len(selected) - survivors} of {len(selected)} mutations caught.")
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main())
