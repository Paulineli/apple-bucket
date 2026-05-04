"""Resolved paths for the entity-binding experiment (`experiments/entity_binding/`)."""
from pathlib import Path

_p = Path(__file__).resolve()
EXP_ROOT = _p.parents[1]      # experiments/entity_binding
ARTIFACTS = EXP_ROOT / "artifacts"
REPO_ROOT = _p.parents[3]   # repo root
