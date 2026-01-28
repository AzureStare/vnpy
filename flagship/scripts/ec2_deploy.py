"""Legacy wrapper for backwards compatibility.

Canonical implementation moved to `flagship.deploy.ec2_deploy`.
"""

from __future__ import annotations


from flagship.deploy.ec2_deploy import *  # noqa: F401,F403


if __name__ == "__main__":
    main()
