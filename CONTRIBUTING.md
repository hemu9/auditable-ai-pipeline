# Contributing

Use Python 3.10+ and install with `python -m pip install -e '.[dev]'`. Run `ruff check .`, `ruff format --check .`, `python -m unittest discover -s tests -t . -v` and `auditable-ai check-fixtures` before proposing a change. `python -m build` checks distribution packaging. Development-tool versions are pinned in `pyproject.toml`.

Preserve the distinction between public source availability and system recording history. Never overwrite version identities or resolve revision ties by insertion order. New backends must share the temporal resolver and pass the historical-query tests.

Keep examples synthetic or explicitly redistribution-cleared. Never add credentials, real private documents or reports containing personal information. Raw model output remains untrusted; a real citation never automatically establishes factual correctness. Add failure cases for changes to validation behavior and update the documented contract when behavior changes.

Tests never call a model. Provider tests use scripted transports. When the optional SDK is installed (`python -m pip install -e '.[dev,claude]'`), they also exercise the real SDK over a mocked HTTP layer. Changing the prompt, schema, dataset, annotations or rubric of `figure-extraction-v1` requires a new task version. Published model results must come from recorded API runs, never from scripted fixtures, and must state who produced and reviewed the annotations.

Work on a feature branch and submit a pull request. Keep the package small; provider integrations should remain behind the injected adapter boundary.
