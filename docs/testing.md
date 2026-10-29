# Testing notes

The README and [architecture notes](architecture.md) state guarantees. This page explains how the test suite checks them, what that checking found, and how to extend it.

## Four layers

- **Example tests.** Hand-picked cases cover each documented behavior and every failure class. Provider calls use scripted transports, so nothing calls a model.
- **Property-based tests** ([`tests/test_properties.py`](../tests/test_properties.py)). Hypothesis generates the inputs and shrinks any counterexample to a minimal one. Each property restates a guarantee:
  - **Extraction boundary.** Any adapter output, including malformed JSON-like values and values that are not JSON at all, is either validated item by item or refused as a whole batch. A refused batch stores nothing. An accepted batch stores exactly its accepted claims. Every accepted citation resolves to a registered version or chunk with a matching hash. Every submitted item is either accepted or rejected with a reason. The same output always gets the same verdict.
  - **Model runs.** Any sequence of provider errors, ending in any response body, produces a run record. The record can be written and read back unchanged, and replay reproduces its outcome. A record written by this code is never flagged as an integrity failure, and an evaluation report can be built from it.
  - **Hand-edited records.** Generated edits, plus a deterministic sweep that deletes, nulls or retypes every field of a record, all end one of three ways: the record is refused as malformed, reported as a failure, or scored exactly like the original, with the same score, token counts, cost and served model. Edited hashed content is never scored, and the CLI never crashes on an edited file.
  - **Historical queries.** Random version histories are inserted in random order into the in-memory store, a JSONL file and the reopened file. For each source, every query returns the one eligible version with no later eligible sibling, raises an error on a tie, and never returns a version from after the cutoff.
- **Mutation check** ([`tools/mutation_check.py`](../tools/mutation_check.py)). Each of 139 entries disables one stated guarantee by replacing exact source text, for example by turning a check into `if False:`. The check copies the repository once per entry, applies that one change and runs the whole suite. It fails if the suite still passes for any entry, because then no test protects that guarantee.
- **Strict type checking.** CI runs mypy in strict mode against the installed SDK.

## What these checks found

Each bug below now has a regression test that fails on the code before the fix. The inputs are the minimal counterexamples Hypothesis reported, or the smallest case that reproduces a bug found by reading the code.

| Found by | Input | Effect before the fix | Fix |
|---|---|---|---|
| Property test | A response whose `usage.input_tokens` is `[null]`, or whose `model` is a list | Cost estimation raised after the call returned, so the record of a possibly billed run was never written | Malformed usage makes the estimate unavailable; a non-string model reads as absent |
| Property test | A fallback block whose `from` is a number | Interpreting the response raised after the call returned; no record was written | A field of an unexpected type reads as absent |
| Code review, confirmed by a failing test | A response body containing `NaN` | Python's parser accepted it. The record then could not be serialized and left an empty record file behind | Responses and record files are parsed as standard JSON; records and reports are serialized before their file is created |
| Property test | A hand-edited record: `cost` set to `true`, `run_id` deleted, or a file holding a JSON array | `evaluate --system recorded` and `replay` crashed with `TypeError`, `KeyError` or `AttributeError` | Reading a record checks every field that replay, evaluation and the CLI use, and names the bad field |
| Property test, then code review | Edited summary fields: `response.usage`, `cost`, `served_model_matches_request` | Reports used them, so token counts and cost could be misreported, and a flipped served-model flag was trusted | Evaluation re-derives these from the replayed response; a contradiction is an integrity failure |
| Code review, confirmed by a failing test | A completed record with its `exchange` removed | Evaluation counted it as completed with no result, and building the report crashed | A record that claims an outcome without a response is an integrity failure |
| Randomized property search | A record missing `served_model_matches_request` or `exchange` | The new shape check read a missing field as null and accepted the record, and evaluation crashed on the missing key | Every checked field must be present; the deterministic sweep now covers every field |
| Mutation check, run locally on Python 3.14 | Removing the handlers for response bodies and output text nested too deeply to parse | No test failed. Python 3.14's JSON parser accepts deeper input than 3.12's, so the deep-input tests never reached those handlers | A test now raises the parser's recursion error directly |
| Mutation check, run in CI on Python 3.12 | Moving the response fingerprint outside the guarded snapshot | No test failed. On 3.12 the JSON step rejects the deep test input before fingerprinting, so the test never reached the fingerprint guard | A test now makes fingerprinting raise the recursion error directly |

Reading the code during this work also led to smaller hardening, each with a test. Settings are checked when they are created, and numeric CLI options are checked before anything is sent. A run with a clock that has no UTC offset is refused before any call. Replay now runs under the recorded run ID, so it can compare whole results rather than a summary. Because the same response can now be classified differently, `validator_version` went from 2 to 3.

## Running the checks

```bash
python -m pip install -e '.[dev,claude]'
python -m unittest discover -s tests -t . -v        # property tests skip without the dev extra
HYPOTHESIS_PROFILE=explore python -m unittest tests.test_properties   # randomized, 3,000 examples each
mypy
python tools/mutation_check.py                      # every mutation
python tools/mutation_check.py --only C9 R12        # selected mutations
python tools/mutation_check.py --list               # the catalog, without running it
```

The full mutation check took about a minute and a half on a 16-core machine.

## Adding a guarantee

1. Write a test that fails without it.
2. Add an entry to `MUTATIONS` in `tools/mutation_check.py` that disables it, with a one-line statement of the guarantee.
3. Run the mutation check. A surviving entry means no test protects the guarantee yet.

If a change moves or rewrites code that an entry targets, the check stops, names the stale entry and runs nothing. Update that entry's find and replace text to match the new code.

## Limits

- The mutation check measures the suite against 139 hand-written mutations, each chosen to disable a stated guarantee. It says nothing about bugs no one has listed.
- In CI, property tests run with a fixed seed, so the same inputs run every time. The `explore` profile searches further. Neither proves that no counterexample exists.
- None of these checks measures model accuracy. See the [evaluation notes](evaluation.md).
