# Public export scope

This is a standalone educational engineering package adapted from selected mechanisms in a personal research system. It has fresh Git history and no runtime dependency on that system.

## Included

- Immutable document versions and chunk anchors, content hashes and frozen metadata.
- Distinct public-availability and system-recording query modes.
- Shared temporal selection across in-memory and reference JSONL backends.
- A generic claim representation with deterministic extraction-output validation.
- New synthetic operational reports, scripted responses, a runnable walkthrough, tests and a failure analysis.

## Adaptation choices

The temporal and source-store mechanisms preserve the selected original algorithms, including explicit revision ambiguity, scoped resolution and the documented informational status of supersession fields. Public module names, comments and docstrings are rewritten for a standalone audience.

The claim/extractor API is adapted to a generic `subject_id` and removes private-system context. A small FACT/INFERENCE enum replaces a dependency on the original metric package; FACT remains source attribution rather than truth verification. Domain-specific action denylists are omitted: this library has no action-routing path and makes no claim to recognize arbitrary malicious prose. It accepts explicit claim types as descriptive strings.

The public adapter snapshots the complete JSON response and reports its hash. Its claim repository checks supplied evidence hashes and chunk timing, and validates top-level opposing references. These are public-package checks, not claimed repairs or behavior changes in the private system. The public result records its validator version, rather than an investment methodology version. The private system is unchanged.

## Excluded

No investment selection, research ranking, valuation, survival, validation-weighting, portfolio or trading rules are included. No live research, holdings, decision journals, hidden evaluation cases, provider datasets, third-party document text, original specifications, internal design history or private repository history are copied.

All bundled document text and responses are newly written synthetic examples. There is no claim of live model accuracy or production deployment. The code and documentation were developed with AI assistance; the tests and visible limitation case make the implemented behavior inspectable.
