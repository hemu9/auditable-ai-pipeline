# Failure analysis: a correction arriving before its original

## The defect

A tempting implementation keeps one mutable "current version" pointer. Whenever a new document arrives, it supersedes whichever record was current and makes the arriving record current.

That works only if ingestion order matches historical revision order. Backfills violate that assumption.

Consider two versions of one synthetic document:

| Version | Public availability | This system recorded it | Stated sensor count |
|---|---|---|---|
| original | April 1 | April 10 | 36 |
| correction | April 15 | April 20 | 24 |

A migration or replay may load the correction first and the original second. Updating "current" on insertion would make the older original win. Historical results would then depend on the loader's traversal order, not on what was knowable or recorded.

## The correction

Store immutable versions. At query time:

1. Select the appropriate timestamp dimension for the question.
2. Keep versions eligible by the cutoff.
3. Group by stable logical source identity.
4. Select the uniquely latest eligible timestamp in each group.
5. Raise an ambiguity if more than one version shares that winning timestamp.

Both storage backends call this resolver. A source-specific query filters to the entire logical lineage before resolving revisions, so an unrelated ambiguous source cannot stop it.

## Evidence from the regression suite

The tests load the original, correction and another source in all six possible orders. They check both historical modes before ingestion, between public correction and correction ingestion, and after both recordings. They repeat the queries through the in-memory backend, JSONL backend and reopened JSONL file. Other tests verify that equal timestamps are refused and earlier versions remain individually retrievable.

These tests establish correctness for the cases exercised. They do not establish upstream timestamp accuracy, correct revision grouping, concurrent-write safety or throughput at production scale. Those are separate responsibilities.

## Why this matters beyond this example

The same ordering problem can arise with amended filings, corrected clinical reports, policy revisions, support incidents, or late-arriving business records. Separating public knowledge from system recording history makes backfills explainable without pretending the application had information before it actually recorded it.
