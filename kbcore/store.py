"""Postgres side: bootstrap, incremental sync, one-statement hybrid search, status."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from .chunker import _HTML_COMMENT_RE, chunk_markdown
from .config import HERE, Config
from .scan import NOT_IN_SCOPE, Found, Source, denied, doc_fields, locate, scan, scope_verdict, sha256, split_frontmatter

EMBED_GROUP = 192  # chunks embedded + committed per group: bounded memory, checkpointed progress


def connect(cfg: Config, dbname: str | None = None, user: str | None = None):
    import psycopg

    return psycopg.connect(cfg.dsn(dbname, user), autocommit=False)


# ---------------------------------------------------------------- bootstrap
def init(cfg: Config) -> None:
    """Create role kb_owner (NOSUPERUSER), database, extension, schema. Run as the cluster owner."""
    import psycopg

    with psycopg.connect(f"host={cfg.host} port={cfg.port} dbname=postgres", autocommit=True) as c:
        if not c.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (cfg.user,)).fetchone():
            c.execute(f'CREATE ROLE "{cfg.user}" LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE')
        if not c.execute("SELECT 1 FROM pg_database WHERE datname=%s", (cfg.database,)).fetchone():
            c.execute(f'CREATE DATABASE "{cfg.database}" OWNER "{cfg.user}"')
    sql = (HERE / "schema.sql").read_text(encoding="utf-8")
    with psycopg.connect(f"host={cfg.host} port={cfg.port} dbname={cfg.database}", autocommit=True) as c:
        c.execute("CREATE EXTENSION IF NOT EXISTS vector")  # needs superuser; the rest runs as owner
        c.execute(f'ALTER DATABASE "{cfg.database}" OWNER TO "{cfg.user}"')
    with connect(cfg) as c:
        c.execute(sql)
        c.commit()


# ---------------------------------------------------------------- thermal guard
def thermal_ok() -> bool:
    """False when macOS reports a CPU speed limit below 100 (thermal throttling)."""
    try:
        out = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return True
    m = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", out)
    return not (m and int(m.group(1)) < 100)


def wait_thermal(log=print, limit_s: int = 600) -> None:
    waited = 0
    while not thermal_ok():
        if waited >= limit_s:
            raise RuntimeError("thermal throttling for >10 min; stopping the build")
        log("thermal limit reported; pausing 30 s")
        time.sleep(30)
        waited += 30


# ---------------------------------------------------------------- chunk preparation
def _words(stem: str) -> str:
    return re.sub(r"[-_]+", " ", stem)


def prepare(src: Source, f, data: bytes) -> tuple[dict, list[dict]]:
    text = data.decode("utf-8", errors="replace")
    fm, body = split_frontmatter(text)
    stem = Path(f.rel).stem
    fields = doc_fields(fm, body, stem)
    chunks = []
    for ix, c in enumerate(chunk_markdown(body, f.rel)):
        label = f"{fields['title']} | {_words(stem)}"
        if c.heading_path:
            label += f" | {c.heading_path}"
        if ix == 0 and fields["description"]:
            label += f" | {fields['description']}"
        text_c = _HTML_COMMENT_RE.sub("", c.content).strip()
        chunks.append(
            {
                "ix": ix,
                "heading_path": c.heading_path,
                "start": c.start_line,
                "end": c.end_line,
                "label": label,
                "body": text_c,
                "sha": sha256((label + "\0" + text_c).encode()),
            }
        )
    return fields | {"frontmatter": fm}, chunks


def _vec(v) -> str:
    return "[" + ",".join(f"{x:.6g}" for x in v) + "]"


def git_head(root: Path) -> str | None:
    if not (root / ".git").exists():
        return None
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


# ---------------------------------------------------------------- sync
class SyncStats(dict):
    def __missing__(self, k):
        return 0


def sync(cfg: Config, conn, embedder, sources=None, full=False, log=lambda *_: None) -> SyncStats:
    """Bring the index in line with the files. Only changed files are read, only new chunks embedded."""
    st = SyncStats()
    for src in cfg.sources:
        if sources and src.name not in sources:
            continue
        refused: list = []
        found = {f.rel: f for f in scan(src, refused)}
        st["refused"] += len(refused)
        have = {
            r[0]: r
            for r in conn.execute(
                "SELECT path, doc_id, content_sha256, size, extract(epoch from mtime) FROM kb.doc WHERE source=%s", (src.name,)
            )
        }
        # vanished files
        for path in set(have) - set(found):
            conn.execute("DELETE FROM kb.doc WHERE doc_id=%s", (have[path][1],))
            st["deleted"] += 1
        todo: list = []
        for rel, f in found.items():
            t = _plan_file(conn, f, have.get(rel), full, st)
            if t:
                todo.append(t)
        conn.commit()
        head = git_head(src.root)
        # groups of docs bounded by chunk count
        group, gchunks = [], 0

        def flush():
            nonlocal group, gchunks
            if group:
                _write_group(conn, embedder, src, group, head, st, log)
                group, gchunks = [], 0

        for f, data, doc_id in todo:
            fields, chunks = prepare(src, f, data)
            group.append((f, data, doc_id, fields, chunks))
            gchunks += len(chunks)
            if gchunks >= EMBED_GROUP:
                flush()
        flush()
    conn.commit()
    return st


def _plan_file(conn, f: Found, h, full: bool, st):
    """(f, data, doc_id) when the file must be (re)indexed; None when the index row is already current."""
    if h and not full and h[3] == f.size and abs(float(h[4]) - f.mtime) < 0.002:
        st["unchanged"] += 1
        return None
    data = Path(f.abs).read_bytes()
    if h and not full and bytes(h[2]) == sha256(data):
        conn.execute("UPDATE kb.doc SET mtime=to_timestamp(%s), size=%s WHERE doc_id=%s", (f.mtime, f.size, h[1]))
        st["touched"] += 1
        return None
    return (f, data, h[1] if h else None)


def sync_path(cfg: Config, conn, embedder, path, log=lambda *_: None) -> SyncStats:
    """Index (or drop) ONE file. Same scope and deny verdict as the full scan; a path outside every source is a no-op.

    A file that no longer exists is removed from the index. A file the scope refuses is never indexed
    (and any row it somehow has is removed).
    """
    st = SyncStats()
    loc = locate(cfg.sources, path)
    if loc is None:
        st["out_of_scope"] += 1
        return st
    src, rel, ap = loc
    row = conn.execute(
        "SELECT path, doc_id, content_sha256, size, extract(epoch from mtime) FROM kb.doc WHERE source=%s AND path=%s",
        (src.name, rel),
    ).fetchone()
    verdict = scope_verdict(src, rel, ap)
    if verdict and verdict != NOT_IN_SCOPE:
        st["refused"] += 1
    if verdict or not os.path.isfile(ap):
        if row:
            conn.execute("DELETE FROM kb.doc WHERE doc_id=%s", (row[1],))
            st["deleted"] += 1
        elif verdict == NOT_IN_SCOPE:
            st["out_of_scope"] += 1
        conn.commit()
        return st
    s = os.stat(ap)
    t = _plan_file(conn, Found(rel, ap, s.st_size, s.st_mtime), row, False, st)
    if t:
        f, data, doc_id = t
        fields, chunks = prepare(src, f, data)
        _write_group(conn, embedder, src, [(f, data, doc_id, fields, chunks)], git_head(src.root), st, log)
    conn.commit()
    return st


def _write_group(conn, embedder, src, group, head, st, log) -> None:
    # reuse embeddings of chunks that kept their hash
    reuse: dict[bytes, str] = {}
    for _f, _d, doc_id, _fields, _ch in group:
        if doc_id:
            for sha, emb in conn.execute("SELECT chunk_sha256, embedding::text FROM kb.chunk WHERE doc_id=%s AND embedding IS NOT NULL AND embed_model=%s", (doc_id, embedder.model)):
                reuse[bytes(sha)] = emb
    need: dict[bytes, str] = {}
    for _f, _d, _id, _fields, chunks in group:
        for c in chunks:
            if c["sha"] not in reuse and c["sha"] not in need:
                need[c["sha"]] = c["label"] + "\n" + c["body"]
    emb_new: dict[bytes, str] = {}
    if need and embedder is not None:
        wait_thermal(log)
        keys = list(need)
        vecs = embedder.embed([need[k] for k in keys])
        emb_new = {k: _vec(v) for k, v in zip(keys, vecs)}
        st["embedded"] += len(keys)
    for f, data, doc_id, fields, chunks in group:
        import json

        if doc_id:
            conn.execute("DELETE FROM kb.doc WHERE doc_id=%s", (doc_id,))
        row = conn.execute(
            """INSERT INTO kb.doc (source, layer, path, content_sha256, size, git_sha, title, type, status, tags, sources,
                                   doc_date, frontmatter, mtime)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,to_timestamp(%s)) RETURNING doc_id""",
            (src.name, src.layer, f.rel, sha256(data), f.size, head, fields["title"], fields["type"], fields["status"],
             fields["tags"], fields["sources"], fields["doc_date"], json.dumps(fields["frontmatter"], default=str), f.mtime),
        ).fetchone()
        for c in chunks:
            emb = reuse.get(c["sha"]) or emb_new.get(c["sha"])
            conn.execute(
                """INSERT INTO kb.chunk (doc_id, chunk_ix, heading_path, start_line, end_line, label, body, chunk_sha256,
                                         embedding, embed_model)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::vector,%s)""",
                (row[0], c["ix"], c["heading_path"], c["start"], c["end"], c["label"], c["body"], c["sha"], emb,
                 embedder.model if emb and embedder else None),
            )
        st["indexed"] += 1
        st["chunks"] += len(chunks)
    conn.commit()
    log(f"{src.name}: +{len(group)} files, {st['embedded']} embedded so far")


# ---------------------------------------------------------------- search
# Rankers. Each mode is (use the vector leg, which full-text leg, weight of the full-text leg in RRF). The fused score is
# sum(weight / (60 + rank)) over the legs a chunk appears in, then the best chunk per file. The file's score is then
# multiplied by its layer's weight (sources.toml [layer_weights], default 1.0) before the final ORDER BY ... LIMIT k, so a
# down-weighted layer ranks lower in every mode. Each leg keeps its own top 40 first: the weight reorders those candidates.
#   fts "legacy": OR query on the `russian` tsvector only (the original leg).
#   fts "dual":   AND-then-OR query on the `russian` and the stoplisted `simple` tsvectors; chunks matching ALL query
#                 terms (in either config) come first, ts_rank_cd of the two OR matches orders within each group.
MODES = {
    "vec": {"vec": True, "fts": None, "w": 1.0},
    "h0": {"vec": True, "fts": "legacy", "w": 1.0},  # the hybrid kb shipped with
    "h0w25": {"vec": True, "fts": "legacy", "w": 0.25},
    "h0w50": {"vec": True, "fts": "legacy", "w": 0.5},
    "h1": {"vec": True, "fts": "dual", "w": 1.0},
    "h1w25": {"vec": True, "fts": "dual", "w": 0.25},
    "h1w50": {"vec": True, "fts": "dual", "w": 0.5},
    "fts0": {"vec": False, "fts": "legacy", "w": 1.0},
    "fts1": {"vec": False, "fts": "dual", "w": 1.0},
}
ALIASES = {"hybrid": "h0", "fts": "fts0"}  # names the first eval and the tests use
DEFAULT_MODE = "vec"  # best of the rankers on the author's own notes; measure on yours before changing it

_FTS_LEGACY = """
qt AS (SELECT replace(websearch_to_tsquery('russian', %(q)s)::text, '&', '|')::tsquery AS ru_or),
fts AS (SELECT c.chunk_id, row_number() OVER (ORDER BY ts_rank_cd(c.tsv, qt.ru_or) DESC) AS r
          FROM kb.chunk c JOIN kb.doc d USING (doc_id), qt
         WHERE c.tsv @@ qt.ru_or AND d.source = ANY(%(src)s) AND d.layer = ANY(%(layer)s)
         ORDER BY r LIMIT 40),"""
_FTS_DUAL = """
qt AS (SELECT replace(websearch_to_tsquery('russian', %(q)s)::text, '&', '|')::tsquery AS ru_or,
              websearch_to_tsquery('russian', %(q)s) AS ru_and,
              replace(websearch_to_tsquery('kb.simple_sw', %(q)s)::text, '&', '|')::tsquery AS s_or,
              websearch_to_tsquery('kb.simple_sw', %(q)s) AS s_and),
fts AS (SELECT c.chunk_id,
               row_number() OVER (ORDER BY (c.tsv @@ qt.ru_and OR c.tsv_simple @@ qt.s_and) DESC,
                                           ts_rank_cd(c.tsv, qt.ru_or) + ts_rank_cd(c.tsv_simple, qt.s_or) DESC) AS r
          FROM kb.chunk c JOIN kb.doc d USING (doc_id), qt
         WHERE (c.tsv @@ qt.ru_or OR c.tsv_simple @@ qt.s_or) AND d.source = ANY(%(src)s) AND d.layer = ANY(%(layer)s)
         ORDER BY r LIMIT 40),"""
_NO_FTS = "\nfts AS (SELECT NULL::uuid AS chunk_id, NULL::bigint AS r WHERE false),"
_HEAD = "WITH"
_TAIL = """
vec AS (SELECT c.chunk_id, row_number() OVER (ORDER BY c.embedding <=> %(emb)s::vector) AS r
          FROM kb.chunk c JOIN kb.doc d USING (doc_id)
         WHERE %(use_vec)s AND c.embedding IS NOT NULL AND d.source = ANY(%(src)s) AND d.layer = ANY(%(layer)s)
         ORDER BY c.embedding <=> %(emb)s::vector LIMIT 40),
fused AS (SELECT chunk_id, sum(wt / (60 + r)) AS score
            FROM (SELECT chunk_id, r, %(w)s::float8 AS wt FROM fts UNION ALL SELECT chunk_id, r, 1.0 FROM vec) u GROUP BY chunk_id),
best AS (SELECT DISTINCT ON (c.doc_id) c.doc_id, c.chunk_id, f.score
           FROM fused f JOIN kb.chunk c USING (chunk_id) ORDER BY c.doc_id, f.score DESC)
SELECT d.source, d.path, c.heading_path, c.start_line, c.end_line, (b.score * lw.w)::float8, left(c.body, 200),
       d.content_sha256, d.layer
  FROM best b JOIN kb.chunk c USING (chunk_id) JOIN kb.doc d ON d.doc_id = c.doc_id
       CROSS JOIN LATERAL (SELECT coalesce((SELECT t.w FROM unnest(%(lay)s::text[], %(lw)s::float8[]) AS t(l, w) WHERE t.l = d.layer), 1.0) AS w) lw
 ORDER BY b.score * lw.w DESC LIMIT %(k)s
"""
_FTS_SQL = {None: _NO_FTS, "legacy": _FTS_LEGACY, "dual": _FTS_DUAL}


def search_sql(fts: str | None) -> str:
    return _HEAD + _FTS_SQL[fts] + _TAIL


def resolve_mode(mode: str) -> str:
    name = ALIASES.get(mode, mode)
    if name not in MODES:
        raise ValueError(f"unknown search mode {mode!r}; choose one of: {', '.join(MODES)}")
    return name


def has_dual_leg(conn) -> bool:
    """True when the database carries the tsv_simple migration (run `kb init`, idempotent, to add it)."""
    return conn.execute(
        "SELECT 1 FROM information_schema.columns WHERE table_schema='kb' AND table_name='chunk' AND column_name='tsv_simple'"
    ).fetchone() is not None


def search(cfg: Config, conn, embedder, query: str, k=5, sources=None, layers=None, mode=DEFAULT_MODE) -> list[dict]:
    name = resolve_mode(mode)
    spec = dict(MODES[name])
    if spec["fts"] == "dual" and not has_dual_leg(conn):
        print("kb: this database lacks the tsv_simple column; run `kb init` (idempotent) -- using the legacy full-text leg", file=sys.stderr)
        spec["fts"] = "legacy"
    names = [s.name for s in cfg.sources]
    emb = None
    if spec["vec"] and embedder is not None:
        emb = _vec(embedder.embed([query])[0])
    if spec["vec"] and not spec["fts"] and emb is None:
        raise RuntimeError("vector mode needs the embedding model")
    rows = conn.execute(
        search_sql(spec["fts"]),
        {"q": query, "emb": emb, "use_vec": emb is not None and spec["vec"], "w": spec["w"],
         "lay": list(cfg.layer_weights), "lw": list(cfg.layer_weights.values()),
         "src": sources or names, "layer": layers or sorted({s.layer for s in cfg.sources}), "k": k},
    ).fetchall()
    by = {s.name: s for s in cfg.sources}
    out = []
    for source, path, hp, a, b, score, snip, sha, layer in rows:
        flag = ""
        s = by.get(source)
        if s is not None:
            fp = s.abs_of(path)
            if not fp.exists():
                flag = "GONE"
            elif sha256(fp.read_bytes()) != bytes(sha):
                flag = "STALE"
        out.append({"source": source, "path": path, "heading": hp, "start": a, "end": b, "score": score,
                    "snippet": snip.replace("\n", " "), "flag": flag, "private": bool(s and s.private)})
    return out


# ---------------------------------------------------------------- status
def status(cfg: Config, conn) -> dict:
    per = {}
    for name, files, chunks, emb, last in conn.execute(
        """SELECT d.source, count(DISTINCT d.doc_id), count(c.chunk_id), count(c.embedding), max(d.indexed_at)
             FROM kb.doc d LEFT JOIN kb.chunk c USING (doc_id) GROUP BY d.source ORDER BY 1"""
    ):
        per[name] = {"files": files, "chunks": chunks, "embedded": emb, "last_sync": last.isoformat() if last else None}
    models = [r[0] for r in conn.execute("SELECT DISTINCT embed_model FROM kb.chunk WHERE embed_model IS NOT NULL")]
    # S3: authoritative deny count = python rule over every stored path, + a SQL pre-check
    deny = 0
    bad = []
    for src, path in conn.execute("SELECT source, path FROM kb.doc"):
        if denied(unicodedata.normalize("NFC", path)):
            deny += 1
            bad.append(f"{src}:{path}")
    # fitness 2: files on disk in scope == files indexed
    drift = 0
    for s in cfg.sources:
        disk = {f.rel for f in scan(s)}
        db = {r[0] for r in conn.execute("SELECT path FROM kb.doc WHERE source=%s", (s.name,))}
        drift += len(disk ^ db)
    mem = None
    for s in cfg.sources:
        mp = s.root / "MEMORY.md"
        if s.layer == "memory" and mp.exists():
            b = mp.read_bytes()
            mem = {"lines": b.count(b"\n"), "bytes": len(b)}
    return {"per_source": per, "models": models, "deny_count": deny, "deny_paths": bad, "drift": drift,
            "memory_md": mem}
