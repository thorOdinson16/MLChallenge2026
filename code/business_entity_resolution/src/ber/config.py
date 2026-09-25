"""Paths and global settings. Override locations with BER_DATA / BER_WORK / BER_OUT env vars."""
import os
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]          # code/business_entity_resolution
_PROJECT = _REPO.parents[1]                          # hackathon root

DATA = Path(os.environ.get("BER_DATA", _PROJECT / "student_resource" / "dataset"))
WORK = Path(os.environ.get("BER_WORK", _PROJECT / "work"))
OUT = Path(os.environ.get("BER_OUT", _PROJECT / "output"))

SPLITS = ("train", "test")
SEED = 2026

# Validation: S1 entities are assigned to folds by hash of their id.
N_FOLDS = 4
DEV_FRACTION = 0.10   # share of train S1 used for fast development runs


def path(*parts) -> Path:
    p = WORK.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
