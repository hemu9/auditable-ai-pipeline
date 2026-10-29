# Evaluation notes

## What is measured now

`auditable-ai check-fixtures` checks eight fixed, synthetic response scenarios against declared validator behavior. These are contract regressions, not a statistical evaluation of an AI model. No LLM was called and the response fixtures are scripted, not captured from a provider.

The cases cover valid attribution with opposing evidence, a fabricated citation with a valid sibling, an unexpected command field, a supplied content-hash mismatch, a well-formed false claim, future input, invalid confidence and unresolved negative evidence.

The false-claim case is expected to be accepted with `accuracy_status=NOT_VERIFIED`. It tests an explicit limitation: existence and timing of a citation do not prove the claim is supported. Counting that case as "a hallucination prevented" would be incorrect.

Unit and integration tests additionally exercise out-of-order revisions, both query modes, backend parity, ambiguity, immutable identities, source/chunk temporal dependencies, conflicting claims, response snapshots, hashes and report overwrite refusal. Their counts describe software checks, not model quality or investment performance.

## What an actual model evaluation would require

Before making model-performance claims, add a separately reviewed dataset with permitted source text and source-grounded annotations. Freeze the evaluation cases and rubric before examining candidate outputs. Keep development examples separate from held-out evaluation. Reviewers should label factual support, valid attribution, omissions, unsupported claims, preserved uncertainty and relevant contradictions independently of mere schema acceptance.

Run real adapters while recording the actual prompt, model version, sampling settings, input versions, raw responses, provider token usage, actual cost and latency. Report denominators, failure categories and uncertainty alongside aggregate measurements. Keep a straightforward extraction baseline and document any exposure of held-out cases. Do not convert a small synthetic regression suite into a claimed hallucination rate.

This public release provides a starting interface and deterministic regression cases. It includes no independent human annotation study, real-model comparison, reliability SLA or production usage measurement.
