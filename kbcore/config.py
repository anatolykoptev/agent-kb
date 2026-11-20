from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .scan import Source, set_deny_dirs

HERE = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = Path("~/.config/agent-kb/sources.toml")


@dataclass(frozen=True)
class Config:
    database: str
    host: str
    port: int
    user: str
    sources: tuple[Source, ...]
    path: Path
    # score multiplier per layer (sources.toml [layer_weights]); a layer not listed weighs 1.0
    layer_weights: dict[str, float] = field(default_factory=dict)

    def dsn(self, dbname: str | None = None, user: str | None = None) -> str:
        return f"host={self.host} port={self.port} dbname={dbname or self.database} user={user or self.user}"


def load(path: str | os.PathLike | None = None) -> Config:
    p = Path(path or os.environ.get("KB_CONFIG") or DEFAULT_CONFIG).expanduser()
    try:
        raw = tomllib.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"kb: no config at {p}: copy {HERE / 'sources.example.toml'} there, or pass --config / set KB_CONFIG")
    set_deny_dirs(raw.get("deny_dirs", []))
    srcs = tuple(
        Source(
            name=s["name"],
            root=Path(os.path.expanduser(s["root"])),
            layer=s.get("layer", "wiki"),
            include=tuple(s.get("include", ["**/*.md"])),
            exclude=tuple(s.get("exclude", ())),
            private=bool(s.get("private", False)),
        )
        for s in raw.get("source", [])
    )
    weights = {str(k): float(v) for k, v in raw.get("layer_weights", {}).items()}
    known = {s.layer for s in srcs}
    for layer, w in weights.items():
        # a typo'd layer name would silently weigh nothing: refuse it
        if layer not in known or not w > 0:
            raise ValueError(f"{p}: [layer_weights] {layer} = {w}: needs a layer used by a [[source]] ({', '.join(sorted(known))}) and a weight > 0")
    return Config(
        database=raw.get("database", "kb"),
        host=raw.get("host", "127.0.0.1"),
        port=int(raw.get("port", 5432)),
        user=raw.get("user", "kb_owner"),
        sources=srcs,
        path=p,
        layer_weights=weights,
    )
