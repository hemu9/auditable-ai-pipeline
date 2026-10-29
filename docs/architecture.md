# Architecture and contracts

## Data flow and trust

`SourceDocument` stores the raw text, content hash, permanent version identity and timing assertions. Construction validates the hash and temporal field ordering. Metadata is deep-frozen so later mutation of the caller's dictionary cannot change a stored document. `SourceChunk` anchors an exact document version and carries its own hash and timestamps.

`EvidenceExtractor` passes only the raw-content wrapper to an injected callable. The callable is executable application code, not a security sandbox. Returning well-formed data cannot grant tools or execute instructions through this library: the schema has no tool-dispatch path. This does not establish resistance to semantic prompt injection; malicious or false prose can still satisfy the schema.

The returned value is snapshotted as finite JSON before validation. Unknown top-level fields or invalid batch shapes raise `ExtractionValidationError`. Valid batches are processed item by item. Unknown item fields, malformed values and failed provenance checks become explicit rejections; valid siblings can still be stored. A run is not a database transaction.

## Evidence checks

Each accepted supporting or opposing source reference must resolve an existing immutable document version. A chunk must exist and belong to that version. If a content hash is supplied, it must match the chunk hash when a chunk is cited, otherwise the document hash. Neither the cited source nor chunk may become available or be recorded after the claim's respective timestamps.

Top-level `contradicting_evidence` is checked before storing any claims. Its provenance failure raises an exception for the batch. `declared_source_ids` is an untrusted declaration retained for inspection; declaration alone is not validation of a source. References used by accepted claims are independently resolved.

`locator_note`, `contradicting_claim_ids` and claim text are not semantically verified. In particular, contradiction IDs are retained pointers and may refer to records not loaded here; the library does not infer or adjudicate disagreement. A FACT without an explicit reference is attributed to the current input document. An INFERENCE may have no references. Both remain accuracy-unverified.

## Two historical questions

Document repositories group versions by logical `source_id` and select the latest eligible version:

| Query | Timestamp used for eligibility and revision order | Meaning |
|---|---|---|
| `RESEARCH_AS_OF(T)` | `publicly_available_at <= T` | What source version was publicly knowable? |
| `OPERATIONAL_REPLAY_AS_OF(T)` | `recorded_at <= T` | What source version had this system recorded? |

A later backfill can change the result of a newly executed research query. It cannot change a previously written report. Operational replay uses system recording time, so late ingestion never makes data appear recorded earlier. The two questions are intentionally different.

`superseded_publicly_at` and `superseded_in_system_at` are validated annotations retained from the extracted record shape. The resolver selects versions from eligible timestamps; it does **not** independently filter on those annotations. Equal winning timestamps are ambiguous and raise an error. Scoped queries restrict the complete logical lineage before resolving revisions so a tie in an unrelated source does not block the query.

Extraction uses **RESEARCH_AS_OF(source_as_of)** to check the explicitly supplied version's public availability, not to replace it with the latest version. Historical or superseded documents can be deliberately analyzed by exact version. Every cited source is also checked against the input source's public timestamp and the run's recording time; this one-document contract intentionally refuses later evidence in an older source-attributed claim. The caller must not interpret this as general multi-document synthesis.

Claims retain the input source's availability and the extraction's own recording time separately. `ClaimRepository.research_as_of` means **source-eligible claims**, not proof the model result existed at the historical cutoff. `operational_replay_as_of` means recorded model outputs. Never use the first to claim historical model performance.

## Audit and reproducibility

Each completed result includes raw response JSON, canonical response hash, source-content hash, exact input version, source cutoff, recording time, model/prompt labels, validator version, accepted claims, rejected items, unknowns and opposing evidence. The adapter's latency is measured locally; the demo's latency is for a scripted callable, not inference. A custom adapter must retain provider responses, prompts and settings itself; the Claude workflow below does this.

The CLI writes reports with exclusive file creation. It refuses a path that already exists. This prevents accidental overwrite through that command; it does not provide tamper-proof storage, authentication or encryption. In library use, malformed batches and adapter exceptions propagate to the caller. The model-run workflow below turns every outcome, including failures, into a run record.

## Claude adapter, run records and replay

`ClaudeExtractionAdapter` (`src/auditable_ai/anthropic_adapter.py`) is an extraction callable like any other: `EvidenceExtractor` invokes it, and its output passes through the same validator. It sends one Messages API request per document through the official `anthropic` SDK, an optional dependency installed with the `claude` extra. The request:

- asks for structured output (`output_config.format`) using the schema in `figure_task.py`;
- sets `effort` explicitly (`medium` by default) and sends no sampling parameters;
- leaves thinking at the model's adaptive default.

The schema leaves source and chunk IDs as free strings, so an invented citation reaches the provenance checks instead of being made impossible by decoding.

The transport returns the exact HTTP response body and the request ID. Connection errors, 408, 409, 429 and 5xx responses are retried with jittered exponential backoff, respecting `retry-after` and capped at the configured number of attempts. Every attempt is recorded, and the reported latency is that of the final attempt only. Interpreting and parsing a response body use plain JSON, not SDK types, so a recording can be replayed without the SDK.

`run_model_extraction` (`src/auditable_ai/runs.py`) produces one record per run:

- the embedded document and its content hash;
- the task version, plus template and prompt fingerprints;
- the settings and SDK version;
- timestamps;
- the exchange: the full request, every attempt, the request ID, the raw response body and its SHA-256;
- the interpreted response: served model, stop reason, stop details, usage and fallback switches;
- whether the served model matches the request;
- a cost estimate;
- the validator version;
- the validated result, or diagnostics.

Outcomes are classified as:

| Outcome | Meaning |
|---|---|
| `completed` | The response was validated and a result was produced |
| `refused` | `stop_reason` was `refusal` |
| `truncated` | `stop_reason` was `max_tokens` |
| `malformed_output` | No text block, or the text is not JSON |
| `validation_error` | The batch shape was invalid |
| `provenance_error` | The top-level counterevidence did not resolve |
| `provider_error` | No response was received |

Records are written with exclusive creation and never overwritten.

Each record's `origin` is `recorded_model_response` for a real API call or `scripted_fixture` for a test double; replay output is labeled `replayed_recording`.

Request headers are never recorded, and anything shaped like an API key or bearer token is redacted from error messages. `extract` and `evaluate --system live` send nothing without `--live`, and they report missing SDKs or credentials before any request is made.

`replay_record` re-validates a recording under the current validator without calling a model. It refuses a recording whose embedded document or response body no longer matches its hash, reproduces the diagnostics of failed runs, and reports which parts of the outcome (status, claims, rejections, unknowns) differ from the recorded ones. A run that received no response is reported as not replayable.

`extract` opts into server-side refusal fallbacks (`fallbacks: "default"` under the `server-side-fallback-2026-07-01` beta), so the served model can differ from the requested one; the record keeps both. `evaluate` disables fallbacks and scores only responses served by the requested model, listing mismatches as failures.

Cost estimates multiply the per-attempt token usage the API reports by the list prices in `pricing.py`, which records its source URL and retrieval date. The estimate is reported as unavailable, not guessed, in these cases:

- an unknown model;
- a refusal;
- fast mode;
- a non-default inference geography or service tier;
- cache writes of unknown duration.

## Storage and scope

The in-memory and JSONL source stores share the same resolver. JSONL reconstructs validated records on read and preserves earlier row contents on additions, but reads and rewrites the entire file. It is appropriate for this small single-process demo, not concurrent or large production workloads. Claims themselves use an in-memory repository.

The core package depends only on Python's standard library. The optional `claude` extra adds the official Anthropic SDK, which only the live transport imports. No private system imports, data providers or domain decision components are needed. Exact public-availability timestamps are upstream assertions; an uncertain date must not be silently represented as precise. Until uncertainty envelopes exist, only run strict historical queries on sources with established exact timestamps.
