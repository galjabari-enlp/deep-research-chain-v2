from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure the repository root is on sys.path before tests import `app.*`.
ROOT = Path(__file__).resolve().parent.parent
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

# Defensive: avoid any accidental name clashes with other installed packages.
os.environ.setdefault("PYTHONPATH", root_str)
