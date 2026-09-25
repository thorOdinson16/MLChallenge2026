"""Deterministic train-S1 folds.

fold 0 of 10 = HOLDOUT: never used to train the encoder or the matcher; the full pipeline is scored on it.
Matcher out-of-fold training uses folds 1..9.
"""
import polars as pl

N = 10
HOLDOUT = 0


def fold_expr(col: str = "s1") -> pl.Expr:
    return (pl.col(col).cast(pl.UInt64) * 2654435761 % 4294967296 % N).cast(pl.UInt8)


def holdout_s1() -> pl.Series:
    from . import io
    s1 = io.load("train", "s1", ["idx"]).rename({"idx": "s1"})
    return s1.filter(fold_expr() == HOLDOUT)["s1"]
