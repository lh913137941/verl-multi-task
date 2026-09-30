"""Shared pytest bootstrap for source-layout and local Ray tests.

Pytest's `pythonpath = ["src"]` updates only the driver process's
`sys.path`.  Local Ray workers are fresh Python processes, so propagate the
same source root through `PYTHONPATH` before any test starts Ray.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


_SRC_ROOT = str((Path(__file__).resolve().parent.parent / "src").resolve())


def _prepend_path(value: str, entries: list[str]) -> str:
    normalized = [entry for entry in entries if entry]
    if value not in normalized:
        normalized.insert(0, value)
    return os.pathsep.join(normalized)


if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

os.environ["PYTHONPATH"] = _prepend_path(
    _SRC_ROOT,
    os.environ.get("PYTHONPATH", "").split(os.pathsep),
)
