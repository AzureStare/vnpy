from __future__ import annotations

import sys
from pathlib import Path


# Ensure local packages (e.g. `flagship/`) are importable when running pytest from an
# environment where the repository root is not automatically on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


