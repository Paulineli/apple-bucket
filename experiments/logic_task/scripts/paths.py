"""Resolved paths for the logic-task experiment (`experiments/logic_task/`)."""
from pathlib import Path

_p = Path(__file__).resolve()
# paths.py sits in …/logic_task/scripts/
EXP_ROOT = _p.parents[1]   # experiments/logic_task
ARTIFACTS = EXP_ROOT / "artifacts"
REPO_ROOT = _p.parents[3]  # repo root (…/logic_task/scripts → repo)
