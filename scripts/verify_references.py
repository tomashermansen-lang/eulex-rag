#!/usr/bin/env python3

"""Thin wrapper (entrypoint).

Canonical implementation: tools/debug/verify_references.py
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "tools" / "debug" / "verify_references.py"
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
