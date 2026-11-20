"""The deny rules live in code. Each excluded path must be refused even when the config lists it."""
import os
import sys
import unicodedata
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kbcore.scan import Source, deny_reason, scan, set_deny_dirs  # noqa: E402


@pytest.fixture(autouse=True)
def configured_deny_dirs():
    """Stand-ins for the config's `deny_dirs`; the built-in raw/ rule needs no config."""
    set_deny_dirs(["journal", "coaching", "casework"])
    yield
    set_deny_dirs([])

DENIED = [
    "raw/book.md",
    "journal/diary.md",
    "coaching/client/journal.md",
    "casework/case.md",
    "raw/interviews/talk.md",
    "wiki/raw/nested.md",  # raw anywhere in the path
    "wiki/coaching/inside-wiki.md",
]


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    for rel in DENIED + ["wiki/ok.md", "wiki/sub/ok2.md", "top.md"]:
        p = tmp_path / unicodedata.normalize("NFC", rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# t\nbody text here\n", encoding="utf-8")
    return tmp_path


def names(found) -> set[str]:
    return {f.rel for f in found}


@pytest.mark.parametrize("rel", DENIED)
def test_each_denied_path_is_refused_even_if_config_includes_it(tree: Path, rel: str):
    # the config lists exactly the denied file and a catch-all
    src = Source("t", tree, "wiki", include=(rel, "**/*.md"))
    refused: list = []
    got = names(scan(src, refused))
    assert unicodedata.normalize("NFC", rel) not in got
    assert not any(g.startswith(("raw/", "journal/", "coaching/", "casework/")) for g in got)
    assert got == {"wiki/ok.md", "wiki/sub/ok2.md", "top.md"}
    assert refused, "refusals must be reported"


@pytest.mark.parametrize("d", ["raw", "journal", "coaching", "casework", "raw/interviews"])
def test_root_inside_denied_folder_yields_nothing(tree: Path, d: str):
    src = Source("t", tree / d, "wiki", include=("**/*.md",))
    assert scan(src) == []


def test_nfc_normalisation_applies(tmp_path: Path, monkeypatch):
    # none of the configured names has a decomposable letter, so prove the normalisation with a stand-in
    import kbcore.scan as sc

    monkeypatch.setattr(sc, "DENY_DIRS", frozenset({unicodedata.normalize("NFC", "Йод").casefold()}))
    nfd = unicodedata.normalize("NFD", "Йод")
    assert nfd != "Йод"
    p = tmp_path / nfd / "j.md"
    p.parent.mkdir()
    p.write_text("# j\nprivate\n", encoding="utf-8")
    assert sc.deny_reason(nfd + "/j.md")
    assert sc.scan(sc.Source("t", tmp_path, "wiki", include=("**/*.md",))) == []


def test_case_variants_are_denied(tmp_path: Path):
    for d in ("RAW", "COACHING"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "x.md").write_text("# x\nbody\n")
    assert scan(Source("t", tmp_path, "wiki", include=("**/*.md",))) == []


def test_symlink_into_denied_folder_is_refused(tree: Path):
    link = tree / "wiki" / "innocent.md"
    link.symlink_to(tree / "coaching" / "client" / "journal.md")
    got = names(scan(Source("t", tree, "wiki", include=("wiki/**/*.md",))))
    assert "wiki/innocent.md" not in got
    assert "wiki/ok.md" in got


def test_symlinked_dir_is_not_followed(tree: Path):
    (tree / "wiki" / "via").symlink_to(tree / "raw", target_is_directory=True)
    got = names(scan(Source("t", tree, "wiki", include=("wiki/**/*.md",))))
    assert not any(g.startswith("wiki/via/") for g in got)


def test_normal_paths_pass():
    assert deny_reason("wiki/people/ivan.md") is None
    assert deny_reason("raw.md") is None  # a file named raw.md is not the raw layer
    assert deny_reason("wiki/drawing/x.md") is None


def test_locate_resolves_symlinked_roots_and_keeps_deleted_paths(tmp_path: Path):
    from kbcore.scan import locate

    real = tmp_path / "real"
    (real / "wiki").mkdir(parents=True)
    link = tmp_path / "link"
    link.symlink_to(real)
    src = Source("t", real, "wiki", include=("wiki/**/*.md",))
    got = locate([src], link / "wiki" / "gone.md")  # via the symlink, file does not exist
    assert got and got[0] is src and got[1] == "wiki/gone.md"
    assert locate([src], tmp_path / "other" / "x.md") is None
    assert locate([src], real) is None  # the root itself is not a file in it
    assert locate([src], real / ".." / "real2" / "x.md") is None


# ---- file-level roots (a source whose root is one .md file, e.g. ~/CLAUDE.md) ----------------------------------
def test_file_root_yields_exactly_that_file(tmp_path: Path):
    (tmp_path / "CLAUDE.md").write_text("# rules\nbody\n")
    (tmp_path / "OTHER.md").write_text("# other\nbody\n")
    src = Source("home-md", tmp_path / "CLAUDE.md", "rules", include=("**/*.md",))
    assert src.file_root and src.abs_of("CLAUDE.md") == tmp_path / "CLAUDE.md"
    assert names(scan(src)) == {"CLAUDE.md"}


def test_file_root_through_a_symlink_is_indexed_under_the_link_name(tmp_path: Path):
    (tmp_path / "home").mkdir()
    (tmp_path / "home" / "CLAUDE.Darwin.md").write_text("# rules\nbody\n")
    (tmp_path / "CLAUDE.md").symlink_to(tmp_path / "home" / "CLAUDE.Darwin.md")
    assert names(scan(Source("t", tmp_path / "CLAUDE.md", "rules", include=("**/*.md",)))) == {"CLAUDE.md"}


def test_file_root_missing_is_empty_and_dir_root_is_not_a_file_root(tmp_path: Path):
    assert scan(Source("t", tmp_path / "CLAUDE.md", "rules", include=("**/*.md",))) == []
    assert not Source("t", tmp_path, "rules", include=("**/*.md",)).file_root


@pytest.mark.parametrize("d", ["raw", "coaching", "journal", "casework"])
def test_file_root_inside_denied_folder_is_refused(tree: Path, d: str):
    # the deny rules apply to a file root's absolute and resolved path, not only to relative paths
    (tree / d).mkdir(parents=True, exist_ok=True)
    f = tree / d / "CLAUDE.md"
    f.write_text("# private\nbody\n")
    refused: list = []
    assert scan(Source("t", f, "rules", include=("**/*.md",)), refused) == []
    assert refused


def test_file_root_symlink_into_denied_folder_is_refused(tree: Path):
    link = tree / "CLAUDE.md"
    link.symlink_to(tree / "coaching" / "client" / "journal.md")
    refused: list = []
    assert scan(Source("t", link, "rules", include=("**/*.md",)), refused) == []
    assert refused


def test_locate_file_root(tmp_path: Path):
    from kbcore.scan import locate

    (tmp_path / "home").mkdir()
    real = tmp_path / "home" / "CLAUDE.Darwin.md"
    real.write_text("# r\n")
    link = tmp_path / "CLAUDE.md"
    link.symlink_to(real)
    src = Source("home-md", link, "rules", include=("**/*.md",))
    other = Source("d", tmp_path / "docs", "rules", include=("**/*.md",))
    for p in (link, real):  # written through the link or through its target
        got = locate([other, src], p)
        assert got and got[0] is src and got[1] == "CLAUDE.md", p
    assert locate([src], tmp_path / "OTHER.md") is None
    assert locate([src], tmp_path / "home" / "CLAUDE.md") is None  # same name, different file
    link.unlink()  # a deleted file root must still map, so its index row can be dropped
    assert locate([src], link) is not None


def test_example_config_loads():
    from kbcore.config import load

    cfg = load(Path(__file__).resolve().parent.parent / "sources.example.toml")
    by = {s.name: s for s in cfg.sources}
    assert by["memory"].private and not by["wiki"].private
    assert by["rules"].file_root and not by["wiki"].file_root
    assert cfg.layer_weights == {"plans": 0.9}
    assert deny_reason("x/journal/a.md")  # deny_dirs from the example file


def test_config_deny_dirs_add_to_the_builtins_and_cannot_remove_them(tmp_path: Path):
    from kbcore import config as kbconfig

    toml = tmp_path / "c.toml"
    toml.write_text('deny_dirs = ["Diary"]\n[[source]]\nname = "n"\nroot = "."\n')
    kbconfig.load(toml)
    assert deny_reason("a/diary/x.md")  # case-folded, from the config
    assert deny_reason("a/raw/x.md")  # built-in
    toml.write_text('deny_dirs = []\n[[source]]\nname = "n"\nroot = "."\n')
    kbconfig.load(toml)
    assert deny_reason("a/diary/x.md") is None and deny_reason("a/raw/x.md")
