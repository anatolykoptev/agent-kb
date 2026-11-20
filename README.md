# agent-kb

Search an agent's memory notes and your markdown wikis by meaning, from the shell. Russian and English, local
bge-m3 embeddings (default ranker `vec`: cosine over chunk vectors, collapsed to one hit per file), with
PostgreSQL full-text legs available as reciprocal-rank-fusion partners. The command is `kb`.

The markdown files stay the truth; the index is derived and rebuildable (`kb sync --full`). A hit is a
pointer to a file and a line range, so an agent can find the one note that answers a question without loading
every note into its context, then read only that file.

Nothing leaves the machine. Embeddings run in process on CPU (onnxruntime, never a GPU). The only network call
is the one-time download of a public model, `kb fetch-model`, which also runs once by itself when the model is
missing from the cache (set `KB_NO_AUTOFETCH=1` to turn that off and fail with `ModelMissing` instead).

## Requirements

- [`uv`](https://docs.astral.sh/uv/) and Python 3.13 or 3.14: `kb` is a `uv run --script` file with pinned dependencies.
- PostgreSQL 18 (the schema uses `uuidv7()`) with the [pgvector](https://github.com/pgvector/pgvector) extension, reachable on
  the host and port in your config, with `trust` or otherwise passwordless access for the config's role.
- About 570 MB of disk for the model ([gpahal/bge-m3-onnx-int8](https://huggingface.co/gpahal/bge-m3-onnx-int8), an int8 ONNX export
  of BAAI/bge-m3), in the Hugging Face cache.

## Run

```
cp sources.example.toml ~/.config/agent-kb/sources.toml   # edit the roots
kb init                      # once: role kb_owner (NOSUPERUSER), database, pgvector, schema (run as the cluster owner)
kb fetch-model               # once: download the model
kb sync [--source X] [--full]
kb sync --path FILE [--path FILE ...]   # one file: same scope and deny rules as a full sync; a path outside every source is a no-op (exit 0); a deleted file leaves the index
kb search "<question>" [-k 5] [--source memory,wiki] [--layer wiki,memory] [--mode vec|h0|h0w25|h0w50|h1|h1w25|h1w50|fts0|fts1] [--json] [--no-sync]
kb status                    # files/chunks per source, deny count (non-zero exit if > 0), disk-vs-index drift
kb bench [-n 200]            # embed N chunks and extrapolate the full build time on this machine
```

Put `kb` on your `PATH` (`ln -s "$PWD/kb" ~/.local/bin/kb`). `search` first runs a quick sync (a `stat` per file; changed files are
re-chunked and only new chunks are embedded); `--no-sync` skips it. The config is `--config FILE`, else `KB_CONFIG`, else
`~/.config/agent-kb/sources.toml`. `KB_SESSION` is recorded in `kb.search_log`. Exit codes: `search` 3 = index unavailable (use grep);
`status` 2 = a denied path is indexed.

To keep the index fresh while an agent edits notes, call `kb sync --path FILE` from your agent's post-edit hook.

## Config

See [`sources.example.toml`](sources.example.toml). One `[[source]]` per root; a root is a directory or a single `.md` file. Each source has a
`layer` (`memory`, `wiki`, `rules`, `plans`, or any name), `include` / `exclude` globs and `private = true` to tag its hits `[private]` (a
reminder for prompts, not access control). Symlinked directories are not followed. `[layer_weights]` multiplies a file's score by its
layer's weight (default 1.0) in every mode, in SQL before `LIMIT k`; each ranking leg still keeps its own top 40 chunks first, so a
weight reorders candidates and cannot pull in a file no leg ranked. A weight <= 0 or a layer no source uses is refused at load.

## Rankers

`kb search --mode M` (aliases: `--vec-only` = `vec`, `--fts-only` = `fts0`, `hybrid` = `h0`). Default `vec`.

| mode | what it is |
|---|---|
| `vec` | vectors only (default) |
| `h0` | RRF (k=60) of vectors + a `russian` full-text leg (tsvector, OR query) |
| `h1` | RRF of vectors + a dual full-text leg: `russian` and a stoplisted unstemmed `simple` tsvector, chunks matching all terms first, then OR |
| `h0w25` `h0w50` `h1w25` `h1w50` | the same with the full-text leg weighted 0.25 / 0.5: `w/(60+rank)` |
| `fts0` `fts1` | the full-text legs alone |

The default is `vec` because it was the best of these on the author's own corpus; that evaluation set is private and is not shipped, so
no recall figure is claimed here. Measure on your own notes before changing it. Full-text rank is PostgreSQL `ts_rank_cd`, not BM25, and
RRF uses ranks, so score magnitudes are comparable only within one query. A database without the `tsv_simple` column still searches:
`h1*` falls back to the `h0` leg and prints a notice (run `kb init` once to add the column; no re-embedding).

## Scope and deny rules

Sources are opt-in: only the roots in your config are read. Deny rules are applied in code (`kbcore/scan.py`) to every candidate file,
by path component, NFC-normalised and case-folded, on the relative path, the absolute path and the symlink-resolved path, whatever
`include` says:

- any `raw/` folder, and `raw/interviews`, are never indexed (built in);
- `deny_dirs = ["name", ...]` in the config adds folder names to the list. The config can add, never remove the built-ins.

Tests prove each rule, including case variants, Unicode normalisation and symlinks into a denied folder.

## What the output can and cannot be trusted for

- A hit is a pointer to a file and a line range, not a fact. The snippet is 200 characters of a chunk. Open the file before stating
  anything from it. `STALE` = the file changed since indexing, `GONE` = it is missing.
- "No hit" is not proof of absence: five results are a ranking, not a filter. Grep before concluding a fact is not recorded.
- An identifier typed inside a URL or file path (`.../article/414690/...`) is one PostgreSQL token, so full-text cannot match the
  fragment; grep exact strings.
- Chunks of more than 512 tokens are cut at `MAX_TOKENS` before embedding, so their tails are invisible to `vec` search.
- Vectors depend on how chunks are batched (the int8 model quantises activations per batch tensor): the same text embedded in another
  batch composition differs slightly. Changing the batching in `kbcore/embed.py` means re-embedding everything.
- `private` is a tag, not access control. The database is a normal PostgreSQL database: anyone who can connect can read it, as they
  can read the markdown. Do not dump or back up the index to a place the notes themselves should not go.
- Not built, on purpose: no HNSW (the exact scan stays exact; an approximate index changes top-k), no watcher or cron, no reranker,
  no daemon.
- The thermal guard in `kb sync` reads `pmset` and only acts on macOS; elsewhere it is a no-op.

## Tests

`tests/run.sh` (needs `uv`). The store tests create and drop their own `kb_test_*` database and need PostgreSQL 18 with pgvector on
`127.0.0.1:5432` and a passwordless connection (`PGUSER` for the admin connection); they skip when it is down unless `KB_REQUIRE_PG=1`.
The real-embedder test runs only when the model is in the local Hugging Face cache; CI does not download it, so CI covers the scanner, chunker,
store, rankers and fetch logic with a deterministic fake embedder, not the model's retrieval quality.

## Credits

`kbcore/chunker.py` and `kbcore/embed.py` are adapted from [zilliztech/memsearch](https://github.com/zilliztech/memsearch) (MIT, commit
3e296a3), see [`LICENSE-memsearch`](LICENSE-memsearch). Model: `gpahal/bge-m3-onnx-int8` (MIT), an int8 ONNX export of `BAAI/bge-m3` (MIT).

## License

Apache-2.0, see [LICENSE](LICENSE); third-party notice in `LICENSE-memsearch`.
