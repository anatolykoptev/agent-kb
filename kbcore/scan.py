"""Source scanning: what may be indexed, and the deny rules that no `include` glob can override.

The deny rules are applied to every candidate file regardless of the configured `include` globs or
roots. A source config cannot re-include a denied folder. Built-in: any `raw/` folder. `deny_dirs = [...]` in
the config file adds folder names (it cannot remove the built-in rule).
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

def _norm_names(names) -> frozenset[str]:
    return frozenset(unicodedata.normalize("NFC", n).casefold() for n in names)


# Directory names that are never indexed, wherever they sit in the path (NFC, case-folded).
BUILTIN_DENY_DIRS: frozenset[str] = frozenset()  # none: the raw/ rule below is the built-in one
DENY_DIRS = BUILTIN_DENY_DIRS


def set_deny_dirs(extra=()) -> None:
    """Built-ins plus the config's `deny_dirs`. Called by config.load; replaces any earlier extras."""
    global DENY_DIRS
    DENY_DIRS = BUILTIN_DENY_DIRS | _norm_names(extra)


# Raw layers (a `raw/` folder anywhere in the path) are not indexed. This is the one switch,
# and DENY_SEQUENCES below still refuses raw/interviews if it is ever turned off.
RAW_DENIED_V1 = True
DENY_SEQUENCES = (("raw", "interviews"),)  # other people's conversations: always


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s).casefold()


def deny_reason(path: str | os.PathLike) -> str | None:
    """Why `path` must never be indexed, or None. Works on relative and absolute paths."""
    parts = [_nfc(p) for p in PurePosixPath(os.fspath(path).replace(os.sep, "/")).parts]
    for p in parts:
        if p in DENY_DIRS:
            return f"denied folder {p!r}"
    if RAW_DENIED_V1 and "raw" in parts:
        return "denied layer 'raw' (v1)"
    for seq in DENY_SEQUENCES:
        n = len(seq)
        for i in range(len(parts) - n + 1):
            if tuple(parts[i : i + n]) == seq:
                return f"denied path {'/'.join(seq)!r}"
    return None


def denied(rel: str, abs_path: str | os.PathLike | None = None) -> str | None:
    """Deny check on the relative path, the absolute path and the symlink-resolved path."""
    for cand in (rel, abs_path, os.path.realpath(abs_path) if abs_path else None):
        if cand is not None:
            r = deny_reason(cand)
            if r:
                return r
    return None


@dataclass(frozen=True)
class Source:
    name: str
    root: Path
    layer: str
    include: tuple[str, ...]
    exclude: tuple[str, ...] = ()
    private: bool = False

    @property
    def file_root(self) -> bool:
        """A root that names one markdown file (e.g. a CLAUDE.md) instead of a directory.

        Decided by the name, not by existence, so a deleted file still maps to its index row.
        """
        return self.root.suffix.lower() == ".md"

    def abs_of(self, rel: str) -> Path:
        return self.root if self.file_root else self.root / rel


@dataclass(frozen=True)
class Found:
    rel: str  # NFC, posix, relative to the source root
    abs: str
    size: int
    mtime: float


def _match_any(rel: str, pats: tuple[str, ...]) -> bool:
    p = PurePosixPath(rel)
    for pat in pats:
        # "dir/**/*.md" must also match "dir/x.md": PurePath.full_match handles ** properly
        if p.full_match(pat) or fnmatch.fnmatchcase(rel, pat) or (("/" not in pat) and p.name == pat) or (
            ("/" not in pat) and fnmatch.fnmatchcase(p.name, pat)
        ):
            return True
    return False


NOT_IN_SCOPE = "not in scope"


def scope_verdict(src: Source, rel: str, abs_path: str) -> str | None:
    """The one scope decision, shared by the full scan and `kb sync --path`.

    None = index it. NOT_IN_SCOPE = the config does not select it. Any other string = a deny reason.
    """
    if not rel.endswith(".md") or not _match_any(rel, src.include):
        return NOT_IN_SCOPE
    if src.exclude and _match_any(rel, src.exclude):
        return NOT_IN_SCOPE
    return denied(rel, abs_path)


def scan(src: Source, refused: list[tuple[str, str]] | None = None) -> list[Found]:
    """Walk `src`, return the files to index. Denied candidates are dropped and (optionally) reported."""
    out: list[Found] = []
    root = str(src.root)
    if src.file_root:
        return _scan_file_root(src, refused)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        # prune denied directories early; each file is still re-checked below (defence in depth)
        keep = []
        for d in dirnames:
            r = denied(os.path.relpath(os.path.join(dirpath, d), root))
            if r:
                if refused is not None:
                    refused.append((os.path.join(dirpath, d), r))
            else:
                keep.append(d)
        dirnames[:] = sorted(keep)
        for fn in sorted(filenames):
            if not fn.endswith(".md"):
                continue
            ap = os.path.join(dirpath, fn)
            rel = unicodedata.normalize("NFC", os.path.relpath(ap, root)).replace(os.sep, "/")
            v = scope_verdict(src, rel, ap)
            if v == NOT_IN_SCOPE:
                continue
            if v:
                if refused is not None:
                    refused.append((ap, v))
                continue
            try:
                st = os.stat(ap)
            except OSError:
                continue
            out.append(Found(rel, ap, st.st_size, st.st_mtime))
    return out


def _scan_file_root(src: Source, refused: list[tuple[str, str]] | None) -> list[Found]:
    ap = str(src.root)
    rel = unicodedata.normalize("NFC", src.root.name)
    v = scope_verdict(src, rel, ap)
    if v and v != NOT_IN_SCOPE:
        if refused is not None:
            refused.append((ap, v))
        return []
    if v or not os.path.isfile(ap):
        return []
    st = os.stat(ap)
    return [Found(rel, ap, st.st_size, st.st_mtime)]


def locate(sources, path: str | os.PathLike) -> tuple[Source, str, str] | None:
    """Map a file path to (source, NFC posix rel path, absolute path) or None when no source root contains it.

    Containment is decided on the symlink-resolved parent directory, so `~/src/x` and `~/Developer/x` agree.
    The file itself is not resolved here: a deleted file must still map to its index row.
    """
    p = os.path.abspath(os.path.expanduser(os.fspath(path)))
    real = os.path.join(os.path.realpath(os.path.dirname(p)), os.path.basename(p))
    for src in sources:
        if src.file_root:
            # the root IS the file: match its parent-resolved path or its fully resolved one (CLAUDE.md is a symlink)
            rroot = os.path.join(os.path.realpath(os.path.dirname(src.root)), src.root.name)
            if real == rroot or os.path.realpath(p) == os.path.realpath(src.root):
                return src, unicodedata.normalize("NFC", src.root.name), str(src.root)
            continue
        root = os.path.realpath(src.root)
        rel = os.path.relpath(real, root)
        if rel == "." or rel == ".." or rel.startswith(".." + os.sep):
            continue
        return src, unicodedata.normalize("NFC", rel).replace(os.sep, "/"), os.path.join(root, rel)
    return None


_FM = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*(?:\n|\Z)", re.DOTALL)


def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def split_frontmatter(text: str) -> tuple[dict, str]:
    """(frontmatter dict, body with the frontmatter lines blanked so line numbers stay true)."""
    import yaml

    m = _FM.match(text)
    if not m:
        return {}, text
    try:
        fm = yaml.safe_load(m.group(1)) or {}
        if not isinstance(fm, dict):
            fm = {}
    except yaml.YAMLError:
        fm = {}
    blanked = "\n" * text[: m.end()].count("\n") + text[m.end() :]
    return fm, blanked


def _as_list(v) -> list[str] | None:
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    return [str(v)]


def doc_fields(fm: dict, body: str, stem: str) -> dict:
    meta = fm.get("metadata") if isinstance(fm.get("metadata"), dict) else {}
    title = fm.get("title") or fm.get("name")
    if not title:
        m = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
        title = m.group(1).strip() if m else stem
    d = fm.get("date") or fm.get("modified") or fm.get("updated")
    doc_date = None
    if d is not None:
        s = str(d)[:10]
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            doc_date = s
    return {
        "title": str(title),
        "type": str(fm.get("type") or fm.get("kind") or meta.get("type") or "") or None,
        "status": str(fm.get("status") or "") or None,
        "tags": _as_list(fm.get("tags")),
        "sources": _as_list(fm.get("sources")),
        "doc_date": doc_date,
        "description": str(fm.get("description") or ""),
    }
