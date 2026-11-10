"""Machine-specific paths, read from <repo>/.env (see .env.example).  Stdlib only: imported by the
brain venv, Isaac Sim's python.sh and the TRELLIS venv alike."""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load_env(path: Path = REPO / ".env") -> None:
    """KEY=VALUE lines into os.environ (existing environment variables win)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), os.path.expanduser(v.strip().strip('"').strip("'")))


load_env()


def get(name: str, default: str | None = None) -> str:
    v = os.environ.get(name, default)
    if v is None:
        raise SystemExit(f"{name} is not set; copy .env.example to .env and fill it in")
    return v


def resolve_asset(p: str, base: Path) -> str:
    """Config paths may be absolute, relative to the config file, use ${VARS}, or 'isaac://<path>',
    which is searched in every root of ISAAC_ASSET_ROOTS (colon-separated)."""
    if p.startswith("isaac://"):
        rel = p[len("isaac://"):]
        roots = [r for r in get("ISAAC_ASSET_ROOTS").split(":") if r]
        for r in roots:
            q = Path(os.path.expanduser(r)) / rel
            if q.exists():
                return str(q)
        raise FileNotFoundError(f"{rel} not found in ISAAC_ASSET_ROOTS={roots}")
    p = os.path.expandvars(p)
    if "$" in p:
        raise SystemExit(f"unset variable in config path: {p} (fill in .env)")
    q = Path(os.path.expanduser(p))
    return str(q if q.is_absolute() else (base / q).resolve())
