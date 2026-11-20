# CLAUDE.md

`kb` (a `uv run --script` file) plus `kbcore/` (config, scan, chunker, embed, store) and `schema.sql`. Tests: `tests/run.sh`.

- The deny rules in `kbcore/scan.py` are the privacy boundary. Never add a config switch that removes a built-in rule; `deny_dirs` only adds.
- `include` globs cannot override a deny rule; keep a test per rule in `tests/test_scan.py`.
- Dependencies are pinned in the `kb` script header and `tests/run.sh`; bump both together, to current stable versions.
- Changing how chunks are batched changes every stored vector: it needs a re-embed note in CHANGELOG.
- The store tests need PostgreSQL 18 with pgvector; CI sets `KB_REQUIRE_PG=1` so a missing database fails the run.
- Index of documents: [docs/INDEX.md](docs/INDEX.md).
