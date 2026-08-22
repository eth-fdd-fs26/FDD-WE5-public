"""Test package bootstrap.

The eight functions the course notebook builds have **no implementation in this
repo** — their bodies raise ``NotImplementedError``. The suites exercise the real
modules, so before any of them import those modules we bind whatever the
participant has exported into ``solutions/``.

Anything not yet exported simply stays unimplemented, and the checks covering it
fail with a message naming the notebook exercise that builds it. That is the
intended report on a fresh checkout: the suite is the to-do list.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Suites are run both as ``python -m tests`` and as ``python -m tests.test_x``;
# either way the repo root must be importable before we can reach `solutions`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import solutions  # noqa: E402

try:
    solutions.apply(verbose=False)
except RuntimeError:
    # Not everything is exported yet — patch what exists, let the rest fail loudly
    # in the checks that need them.
    pass
