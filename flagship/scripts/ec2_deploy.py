"""Legacy wrapper for backwards compatibility.

Canonical implementation moved to `flagship.deploy.ec2_deploy`.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure `import flagship` works when executed as a script (python flagship/scripts/..py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flagship.deploy.ec2_deploy import *  # noqa: F401,F403


if __name__ == "__main__":
    main()
