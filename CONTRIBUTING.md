# Contributing

Use Python 3.10+ and install with `python -m pip install -e '.[dev,claude]'`. Run `ruff check .`, `ruff format --check .`, `mypy`, `python -m unittest discover -s tests -t . -v` and `auditable-ai check-fixtures` before proposing a change. `python -m build` checks distribution packaging. Development-tool versions are pinned in `pyproject.toml`.

When a change adds or alters a guarantee, write a test that fails without it and add an entry to `MUTATIONS` in `tools/mutation_check.py`, then run `python tools/mutation_check.py`. CI fails if any mutation survives, or if a change leaves an entry's find text stale. [Testing notes](docs/testing.md) describe the property-based tests and the mutation check.

Preserve the distinction between public source availability and system recording history. Never overwrite version identities or resolve revision ties by insertion order. New backends must share the temporal resolver and `validate_chunk_registration`, and subclass the repository contract tests in `tests/test_temporal.py` and `tests/test_documents.py`.

Keep examples synthetic or explicitly redistribution-cleared. Never add credentials, real private documents or reports containing personal information. Raw model output remains untrusted; a real citation never automatically establishes factual correctness. Add failure cases for changes to validation behavior and update the documented contract when behavior changes.

Tests never call a model. Provider tests use scripted transports. When the optional SDK is installed (`python -m pip install -e '.[dev,claude]'`), they also exercise the real SDK over a mocked HTTP layer. Changing the prompt, schema or rubric of `figure-extraction-v1` requires a new task version. Bundled datasets are frozen: add new documents as a new dataset rather than editing an existing one. Published model results must come from recorded API runs, never from scripted fixtures, and must state who produced and reviewed the annotations.

Work on a feature branch and submit a pull request. Keep the package small; provider integrations should remain behind the injected adapter boundary.
