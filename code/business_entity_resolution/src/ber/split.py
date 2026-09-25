"""Deterministic train-S1 folds (10 folds by hash of the S1 row index).

  fold 0       HOLDOUT  - never trained on; the full pipeline is scored here
  folds 1..5   ENC      - bi-encoder training pairs only
  folds 6..9   MATCH    - pruner / matcher training (encoder has never seen these entities, so
                          encoder similarities look exactly like they will on test)
"""
import polars as pl

N = 10
HOLDOUT = {0}
ENC = {1, 2, 3, 4, 5}
MATCH = {6, 7, 8, 9}


def fold_expr(col: str = "s1") -> pl.Expr:
    return (pl.col(col).cast(pl.UInt64) * 2654435761 % 4294967296 % N).cast(pl.UInt8)


def s1_in(folds) -> pl.Series:
    from . import io
    s1 = io.load("train", "s1", ["idx"]).rename({"idx": "s1"})
    return s1.filter(fold_expr().is_in(list(folds)))["s1"]


def holdout_s1() -> pl.Series:
    return s1_in(HOLDOUT)


def x_of_folds(folds) -> pl.DataFrame:
    """Train X records whose true S1 lies in `folds` (column x)."""
    from . import io
    gt = io.load("train", "gt")
    return gt.filter(fold_expr().is_in(list(folds))).select("x")


def clean_x() -> pl.DataFrame:
    """Train X usable for pruner/matcher training: unmatched X, or X whose true S1 is in MATCH folds."""
    from . import io
    x = io.load("train", "x", ["idx"]).rename({"idx": "x"})
    return x.join(x_of_folds(HOLDOUT | ENC), on="x", how="anti")
