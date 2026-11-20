import hashlib
import os
import subprocess
import sys
import textwrap
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kbcore import config as kbconfig  # noqa: E402
from kbcore import store  # noqa: E402
from kbcore.chunker import chunk_markdown  # noqa: E402
from kbcore.scan import split_frontmatter  # noqa: E402


def test_heading_path_and_lines():
    md = "# A\nintro\n## B\nbody b\n### C\nbody c\n## D\nbody d\n"
    ch = chunk_markdown(md, "x.md")
    paths = [c.heading_path for c in ch]
    assert paths == ["A", "A > B", "A > B > C", "A > D"]
    assert ch[1].start_line == 3


def test_frontmatter_blanked_keeps_line_numbers():
    text = "---\nname: n\ntype: feedback\n---\n# H\nbody\n"
    fm, body = split_frontmatter(text)
    assert fm["name"] == "n"
    ch = chunk_markdown(body, "x")
    assert ch[0].start_line == 5 and "name:" not in ch[0].content


class FakeEmbedder:
    """Deterministic 1024-d vectors from words: shared words -> shared dimensions."""

    model = "fake"

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        out = []
        for t in texts:
            self.calls += 1
            v = [0.0] * 1024
            for w in t.lower().split():
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 1024] += 1.0
            n = sum(x * x for x in v) ** 0.5 or 1.0
            out.append([x / n for x in v])
        return out


def pg_up() -> bool:
    try:
        import psycopg

        psycopg.connect("host=127.0.0.1 port=5432 dbname=postgres", connect_timeout=2).close()
        return True
    except Exception:
        return False


@pytest.fixture
def kbdb(tmp_path: Path):
    if not pg_up():
        if os.environ.get("KB_REQUIRE_PG"):  # CI sets this so a missing database fails instead of skipping
            pytest.fail("KB_REQUIRE_PG is set but postgres is not reachable on 127.0.0.1:5432")
        pytest.skip("local postgres not reachable")
    db = "kb_test_" + uuid.uuid4().hex[:8]
    mem = tmp_path / "mem"
    wiki = tmp_path / "w"
    (wiki / "wiki").mkdir(parents=True)
    (wiki / "raw").mkdir()
    mem.mkdir()
    toml = tmp_path / "c.toml"
    toml.write_text(textwrap.dedent(f"""
        database = "{db}"
        host = "127.0.0.1"
        port = 5432
        user = "kb_owner"
        deny_dirs = ["journal", "coaching", "casework"]
        [[source]]
        name = "memory"
        root = "{mem}"
        layer = "memory"
        include = ["*.md"]
        exclude = ["MEMORY.md"]
        [[source]]
        name = "w"
        root = "{wiki}"
        layer = "wiki"
        include = ["**/*.md"]
    """))
    cfg = kbconfig.load(toml)
    store.init(cfg)
    conn = store.connect(cfg)
    yield cfg, conn, mem, wiki
    conn.close()
    import psycopg

    with psycopg.connect("host=127.0.0.1 port=5432 dbname=postgres", autocommit=True) as c:
        c.execute(f'DROP DATABASE "{db}" WITH (FORCE)')


def test_sync_search_incremental_and_denied(kbdb):
    cfg, conn, mem, wiki = kbdb
    (mem / "tiering.md").write_text("---\nname: Model tiering\ndescription: which model for which task\n---\nResearch goes to haiku, code to sonnet.\n")
    (mem / "MEMORY.md").write_text("# index\n")
    (wiki / "wiki" / "borsch.md").write_text("# Борщ\nСвёкла, капуста, картофель и мясо варятся долго.\n")
    (wiki / "raw" / "secret.md").write_text("# Secret\nборщ secret raw text\n")
    emb = FakeEmbedder()
    st = store.sync(cfg, conn, emb)
    assert st["indexed"] == 2 and st["refused"] == 1 and st["embedded"] >= 2  # raw/ pruned and reported
    paths = {r[0] for r in conn.execute("SELECT path FROM kb.doc")}
    assert paths == {"tiering.md", "wiki/borsch.md"}  # raw/ and MEMORY.md never indexed

    hits = store.search(cfg, conn, emb, "haiku sonnet", mode="fts")
    assert hits[0]["path"] == "tiering.md"
    # Russian stemming: "свёклу" finds "Свёкла"
    hits = store.search(cfg, conn, emb, "варить свёклу", mode="fts")
    assert hits[0]["path"] == "wiki/borsch.md"
    hits = store.search(cfg, conn, emb, "model for haiku", mode="hybrid")
    assert hits and hits[0]["path"] == "tiering.md" and hits[0]["flag"] == ""

    # nothing changed: nothing embedded, nothing re-read
    calls = emb.calls
    st2 = store.sync(cfg, conn, emb)
    assert st2["unchanged"] == 2 and emb.calls == calls

    # edit one chunk of one file: only the new chunk is embedded
    (mem / "tiering.md").write_text("---\nname: Model tiering\ndescription: which model for which task\n---\nResearch goes to haiku, code to opus.\n")
    st3 = store.sync(cfg, conn, emb)
    assert st3["indexed"] == 1 and st3["embedded"] == 1

    # STALE / GONE flags
    (mem / "tiering.md").write_text("changed again after indexing, same words haiku\n")
    assert store.search(cfg, conn, emb, "haiku", mode="fts")[0]["flag"] == "STALE"
    (mem / "tiering.md").unlink()
    assert store.search(cfg, conn, emb, "haiku", mode="fts")[0]["flag"] == "GONE"
    st4 = store.sync(cfg, conn, emb)
    assert st4["deleted"] == 1
    assert store.status(cfg, conn)["deny_count"] == 0


# ---------------------------------------------------------------- rankers
def _docs(mem: Path, texts: dict[str, str]) -> None:
    for name, body in texts.items():
        (mem / f"{name}.md").write_text(f"---\nname: {name}\n---\n{body}\n")


def _rank(cfg, conn, emb, q, mode, k=50):
    return store.search(cfg, conn, emb, q, k=k, mode=mode)


def test_unknown_mode_is_an_error_and_aliases_resolve():
    with pytest.raises(ValueError):
        store.resolve_mode("hybrd")
    assert store.resolve_mode("hybrid") == "h0" and store.resolve_mode("fts") == "fts0"
    assert store.DEFAULT_MODE in store.MODES


@pytest.mark.parametrize("mode,w,leg", [("h0", 1.0, "fts0"), ("h0w25", 0.25, "fts0"), ("h0w50", 0.5, "fts0"),
                                         ("h1", 1.0, "fts1"), ("h1w25", 0.25, "fts1"), ("h1w50", 0.5, "fts1")])
def test_weighted_rrf_scores_follow_the_formula(kbdb, mode, w, leg):
    """Each hybrid score is w/(60+rank_fts) + 1/(60+rank_vec), ranks taken from the single-leg modes (one chunk per file)."""
    cfg, conn, mem, _ = kbdb
    _docs(mem, {
        "a": "alpha beta gamma delta", "b": "alpha beta filler words", "c": "alpha only here and more filler",
        "d": "gamma delta epsilon zeta", "e": "beta zeta eta theta filler", "f": "iota kappa lambda mu",
        "g": "alpha alpha alpha unrelated", "h": "delta delta words about nothing",
    })
    emb = FakeEmbedder()
    store.sync(cfg, conn, emb)
    q = "alpha beta gamma delta"
    fr = {h["path"]: i for i, h in enumerate(_rank(cfg, conn, emb, q, leg), 1)}
    vr = {h["path"]: i for i, h in enumerate(_rank(cfg, conn, emb, q, "vec"), 1)}
    assert fr and vr
    expect = {p: (w / (60 + fr[p]) if p in fr else 0.0) + (1 / (60 + vr[p]) if p in vr else 0.0) for p in set(fr) | set(vr)}
    got = _rank(cfg, conn, emb, q, mode)
    for h in got:
        assert h["score"] == pytest.approx(expect[h["path"]], rel=1e-9)
    ranked = [expect[h["path"]] for h in got]
    assert ranked == sorted(ranked, reverse=True)  # ordered by the formula (ties may come in either order)
    assert sorted(ranked, reverse=True) == pytest.approx(sorted(expect.values(), reverse=True)[: len(got)], rel=1e-9)


def test_weight_changes_scores_and_modes_are_not_ignored(kbdb):
    cfg, conn, mem, _ = kbdb
    _docs(mem, {"a": "alpha beta gamma", "b": "alpha filler words", "c": "gamma delta words"})
    emb = FakeEmbedder()
    store.sync(cfg, conn, emb)
    scores = {m: [h["score"] for h in _rank(cfg, conn, emb, "alpha beta gamma", m)] for m in ("h0", "h0w25", "h0w50", "h1w25", "h1w50")}
    # the weight reaches the SQL: w=1, 0.5, 0.25 give three distinct score vectors (h0wNN and h1wNN share one when both legs rank alike)
    assert len({tuple(round(x, 9) for x in scores[m]) for m in ("h0", "h0w25", "h0w50")}) == 3


def _two_layers(mem, wiki):
    """wiki doc matches the query exactly (ranks first unweighted); memory doc matches partly."""
    _docs(mem, {"m": "alpha beta filler words"})
    (wiki / "wiki" / "w.md").write_text("---\nname: w\n---\nalpha beta gamma delta\n")


@pytest.mark.parametrize("mode", ["vec", "h0", "h1", "fts0", "fts1"])
def test_layer_weight_reorders_every_mode_and_multiplies_the_score(kbdb, mode):
    import dataclasses

    cfg, conn, mem, wiki = kbdb
    _two_layers(mem, wiki)
    emb = FakeEmbedder()
    store.sync(cfg, conn, emb)
    q = "alpha beta gamma delta"
    base = _rank(cfg, conn, emb, q, mode)
    assert [h["path"] for h in base] == ["wiki/w.md", "m.md"]
    heavy = dataclasses.replace(cfg, layer_weights={"memory": 3.0})
    got = _rank(heavy, conn, emb, q, mode)
    assert [h["path"] for h in got] == ["m.md", "wiki/w.md"]
    by = {h["path"]: h["score"] for h in base}
    assert got[0]["score"] == pytest.approx(by["m.md"] * 3.0, rel=1e-9)
    assert got[1]["score"] == pytest.approx(by["wiki/w.md"], rel=1e-9)


def test_unlisted_layer_weighs_exactly_one(kbdb):
    import dataclasses

    cfg, conn, mem, wiki = kbdb
    _two_layers(mem, wiki)
    emb = FakeEmbedder()
    store.sync(cfg, conn, emb)
    q = "alpha beta gamma delta"
    base = {h["path"]: h["score"] for h in _rank(cfg, conn, emb, q, "h1")}
    for lw in ({}, {"wiki": 1.0}, {"wiki": 2.0}):
        got = {h["path"]: h["score"] for h in _rank(dataclasses.replace(cfg, layer_weights=lw), conn, emb, q, "h1")}
        assert got["m.md"] == pytest.approx(base["m.md"], rel=1e-12)  # memory is never listed: weight 1.0
        assert got["wiki/w.md"] == pytest.approx(base["wiki/w.md"] * lw.get("wiki", 1.0), rel=1e-12)


def test_layer_weights_config_parses_and_rejects_typos(tmp_path):
    base = 'database = "x"\n[[source]]\nname = "a"\nroot = "/tmp"\nlayer = "rules"\n'
    ok = tmp_path / "ok.toml"
    ok.write_text(base + "[layer_weights]\nrules = 1.25\n")
    assert kbconfig.load(ok).layer_weights == {"rules": 1.25}
    none = tmp_path / "none.toml"
    none.write_text(base)
    assert kbconfig.load(none).layer_weights == {}
    for bad in ("rulez = 1.2", "rules = 0", "rules = -1"):  # unknown layer would silently weigh nothing
        f = tmp_path / "bad.toml"
        f.write_text(base + f"[layer_weights]\n{bad}\n")
        with pytest.raises(ValueError):
            kbconfig.load(f)


def test_dual_leg_puts_all_terms_matches_first(kbdb):
    cfg, conn, mem, _ = kbdb
    _docs(mem, {"repeat": " ".join(["alpha"] * 12) + " filler", "both": "alpha then many filler words and later beta"})
    emb = FakeEmbedder()
    store.sync(cfg, conn, emb)
    assert _rank(cfg, conn, emb, "alpha beta", "fts1")[0]["path"] == "both.md"


def test_dual_leg_matches_unstemmed_token_exactly(kbdb):
    cfg, conn, mem, _ = kbdb
    _docs(mem, {"exact": "the universe is large", **{f"decoy{i}": f"the university {i} is large" for i in range(5)}})
    emb = FakeEmbedder()
    store.sync(cfg, conn, emb)
    # the stemming config conflates universe/university; the stoplisted simple config does not
    assert conn.execute("SELECT count(*) FROM kb.chunk WHERE tsv @@ websearch_to_tsquery('russian','universe')").fetchone()[0] == 6
    assert conn.execute("SELECT count(*) FROM kb.chunk WHERE tsv_simple @@ websearch_to_tsquery('kb.simple_sw','universe')").fetchone()[0] == 1
    assert _rank(cfg, conn, emb, "universe", "fts1")[0]["path"] == "exact.md"


def test_stoplist_keeps_noise_words_out_of_the_simple_vector(kbdb):
    cfg, conn, mem, _ = kbdb
    _docs(mem, {"x": "the quick brown fox и собака", "y": "MAX_RETRIES 4821 sessions.db-wal"})
    store.sync(cfg, conn, FakeEmbedder())
    lex = {r[0] for r in conn.execute("SELECT unnest(tsvector_to_array(tsv_simple)) FROM kb.chunk")}
    assert "the" not in lex and "и" not in lex and "4821" in lex and "собака" in lex and "quick" in lex


def test_schema_migration_is_idempotent_and_backfills_without_reembedding(kbdb):
    cfg, conn, mem, _ = kbdb
    _docs(mem, {"a": "alpha beta", "b": "gamma delta"})
    store.sync(cfg, conn, FakeEmbedder())
    sql = (store.HERE / "schema.sql").read_text(encoding="utf-8")
    conn.execute(sql)  # second application: no error, nothing to do
    conn.execute("DROP INDEX kb.chunk_tsv_simple_gin")
    conn.execute("ALTER TABLE kb.chunk DROP COLUMN tsv_simple")  # an index built before this change
    conn.commit()
    assert not store.has_dual_leg(conn)
    conn.execute(sql)
    conn.commit()
    assert store.has_dual_leg(conn)
    n, filled, emb = conn.execute("SELECT count(*), count(*) FILTER (WHERE tsv_simple <> ''::tsvector), count(embedding) FROM kb.chunk").fetchone()
    assert n == 2 and filled == 2 and emb == 2


def test_old_index_without_tsv_simple_falls_back_to_legacy_leg_with_a_warning(kbdb, capsys):
    cfg, conn, mem, _ = kbdb
    _docs(mem, {"a": "alpha beta gamma", "b": "alpha filler"})
    emb = FakeEmbedder()
    store.sync(cfg, conn, emb)
    want = _rank(cfg, conn, emb, "alpha beta", "h1w50")
    conn.execute("DROP INDEX kb.chunk_tsv_simple_gin")
    conn.execute("ALTER TABLE kb.chunk DROP COLUMN tsv_simple")
    conn.commit()
    got = _rank(cfg, conn, emb, "alpha beta", "h1w50")
    assert [h["path"] for h in got] == [h["path"] for h in want]
    assert "tsv_simple" in capsys.readouterr().err
    assert [h["score"] for h in got] == [h["score"] for h in _rank(cfg, conn, emb, "alpha beta", "h0w50")]


def test_cli_mode_resolution():
    import argparse
    import importlib.machinery
    import importlib.util

    path = str(Path(__file__).resolve().parent.parent / "kb")
    loader = importlib.machinery.SourceFileLoader("kbcli", path)
    spec = importlib.util.spec_from_loader("kbcli", loader)
    cli = importlib.util.module_from_spec(spec)
    loader.exec_module(cli)
    ns = lambda **k: argparse.Namespace(**{"mode": None, "fts_only": False, "vec_only": False, **k})  # noqa: E731
    assert cli.resolve_mode(ns()) == (store.DEFAULT_MODE, False)
    assert cli.resolve_mode(ns(mode="h1w25")) == ("h1w25", True)
    assert cli.resolve_mode(ns(fts_only=True)) == ("fts0", True)
    assert cli.resolve_mode(ns(vec_only=True)) == ("vec", True)
    with pytest.raises(SystemExit):
        cli.resolve_mode(ns(mode="h1", vec_only=True))


def _paths(conn):
    return {r[0] for r in conn.execute("SELECT path FROM kb.doc")}


def test_sync_path_indexes_updates_and_drops_one_file(kbdb):
    cfg, conn, mem, wiki = kbdb
    emb = FakeEmbedder()
    note = mem / "probe.md"
    note.write_text("---\nname: probe\n---\nalpha unique phrase zebra\n")
    (mem / "other.md").write_text("# other\nnot touched by the path sync\n")
    st = store.sync_path(cfg, conn, emb, note)
    assert st["indexed"] == 1 and _paths(conn) == {"probe.md"}  # the sibling stays unindexed
    st = store.sync_path(cfg, conn, emb, note)
    assert st["indexed"] == 0 and st["unchanged"] == 1
    calls = emb.calls
    note.write_text("---\nname: probe\n---\nbeta changed phrase giraffe\n")
    st = store.sync_path(cfg, conn, emb, str(note))
    assert st["indexed"] == 1 and emb.calls > calls
    assert "giraffe" in conn.execute("SELECT string_agg(body, ' ') FROM kb.chunk").fetchone()[0]
    note.unlink()
    st = store.sync_path(cfg, conn, emb, note)
    assert st["deleted"] == 1 and _paths(conn) == set()


def test_sync_path_scope_and_deny_are_the_full_scans_verdict(kbdb, tmp_path):
    cfg, conn, mem, wiki = kbdb
    emb = FakeEmbedder()
    (mem / "MEMORY.md").write_text("# index\nlink list\n")  # excluded by the source config
    outside = tmp_path / "elsewhere.md"
    outside.write_text("# nope\nbody\n")
    denied_files = [
        wiki / "raw" / "book.md",
        wiki / "wiki" / "journal" / "diary.md",
        wiki / "wiki" / "coaching" / "j.md",
        wiki / "wiki" / "casework" / "c.md",
    ]
    for p in denied_files:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# private\nsome body text\n")
    (wiki / "wiki" / "notmd.txt").write_text("text")
    for p in [mem / "MEMORY.md", outside, wiki / "wiki" / "notmd.txt", tmp_path / "nonexistent.md", *denied_files]:
        st = store.sync_path(cfg, conn, emb, p)  # outside every source or refused: a no-op, never an error
        assert st["indexed"] == 0
    assert _paths(conn) == set()
    assert emb.calls == 0
    # the same verdict as a full scan: nothing of the above, and drift stays zero
    store.sync(cfg, conn, emb)
    assert _paths(conn) == set()
    assert store.status(cfg, conn)["drift"] == 0


def test_sync_path_removes_a_row_whose_file_became_refused(kbdb):
    cfg, conn, mem, wiki = kbdb
    emb = FakeEmbedder()
    ok = wiki / "wiki" / "page.md"
    ok.parent.mkdir(parents=True, exist_ok=True)
    ok.write_text("# page\nbody text\n")
    store.sync_path(cfg, conn, emb, ok)
    assert _paths(conn) == {"wiki/page.md"}
    conn.execute("UPDATE kb.doc SET path = 'wiki/coaching/page.md'")  # simulate a pre-existing bad row
    conn.commit()
    bad = wiki / "wiki" / "coaching" / "page.md"
    bad.parent.mkdir(parents=True)
    bad.write_text("# page\nbody text\n")
    store.sync_path(cfg, conn, emb, bad)
    assert _paths(conn) == set()


def test_file_root_source_syncs_searches_by_default_layers_and_drops_with_the_file(kbdb, tmp_path):
    import dataclasses

    from kbcore.scan import Source

    cfg, conn, mem, wiki = kbdb
    emb = FakeEmbedder()
    rules = tmp_path / "CLAUDE.md"
    rules.write_text("# Home layout\nwhere the API token lives: vault items\n")
    cfg = dataclasses.replace(cfg, sources=cfg.sources + (Source("home-md", rules, "rules", include=("**/*.md",)),))
    store.sync(cfg, conn, emb)
    assert _paths(conn) == {"CLAUDE.md"}
    # no --layer: the default must cover every layer in the config, not only memory + wiki
    hits = store.search(cfg, conn, emb, "api token vault", k=3)
    assert [(h["source"], h["path"], h["flag"]) for h in hits] == [("home-md", "CLAUDE.md", "")]
    rules.write_text("# Home layout\nchanged after indexing\n")
    assert store.search(cfg, conn, emb, "api token vault", k=3)[0]["flag"] == "STALE"
    store.sync_path(cfg, conn, emb, rules)
    assert store.status(cfg, conn)["drift"] == 0
    rules.unlink()
    assert store.search(cfg, conn, emb, "changed after indexing", k=3)[0]["flag"] == "GONE"
    st = store.sync_path(cfg, conn, emb, rules)
    assert st["deleted"] == 1 and _paths(conn) == set()
