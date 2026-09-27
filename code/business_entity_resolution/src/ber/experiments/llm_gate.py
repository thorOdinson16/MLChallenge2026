"""Evaluate the zero-shot LLM judge on its US/IN holdout gate: recover labels by matching the gate texts back to the
train records, then per slice compare LLM p_yes against our own signals (v2 p2, ce_v4) and find cut-offs.

    python -m ber.experiments.llm_gate <gate_pairs.parquet> <gate_scores.parquet>
"""
import sys

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from .. import config, io


def main(gp, gs):
    g = pl.read_parquet(gp).join(pl.read_parquet(gs).select("id", "p_yes"), on="id")
    X = io.load("train", "x", ["idx", "name", "addr"]).rename({"idx": "x", "name": "xn", "addr": "xa"})
    S = io.load("train", "s1", ["idx", "name", "addr"]).rename({"idx": "s1", "name": "sn", "addr": "sa"})
    fill = lambda d, cols: d.with_columns([pl.col(c).fill_null("") for c in cols])
    g, X, S = fill(g, ["xn", "xa", "sn", "sa"]), fill(X, ["xn", "xa"]), fill(S, ["sn", "sa"])
    m = g.join(X, on=["xn", "xa"], how="left").join(S, on=["sn", "sa"], how="left")
    tr = pl.read_parquet(config.path("preds", "train_m_v2.parquet"), columns=["x", "s1", "p2"])
    m = m.join(tr, on=["x", "s1"], how="inner")                        # keep only text matches that are real candidate pairs
    m = m.join(pl.read_parquet(config.path("preds", "ce_v4_holdout.parquet")).rename({"ce": "ce_v4"}), on=["x", "s1"], how="left")
    gt = io.load("train", "gt").with_columns(y=pl.lit(1, pl.UInt8))
    m = m.join(gt, on=["s1", "x"], how="left").with_columns(pl.col("y").fill_null(0))
    m = m.sort("p2", descending=True).unique("id", keep="first")      # one row per gate id
    print(f"gate ids {g.height}, recovered with labels {m.height}")
    for sl in ("add", "veto", "wordchg"):
        d = m.filter(pl.col("slice") == sl)
        y = d["y"].to_numpy()
        def auc(c):
            v = d[c].fill_null(-99).to_numpy()
            return roc_auc_score(y, v) if 0 < y.mean() < 1 else float("nan")
        print(f"\n[{sl}] n={d.height} base P(true)={y.mean():.3f}  AUC  llm {auc('p_yes'):.3f} | p2 {auc('p2'):.3f} | ce_v4 {auc('ce_v4'):.3f}")
        for t in (0.5, 0.8, 0.9, 0.95, 0.99):
            hi, lo = d.filter(pl.col("p_yes") >= t), d.filter(pl.col("p_yes") < 1 - t if t > 0.5 else pl.col("p_yes") < t)
            print(f"   llm p_yes >= {t:<4}: n={hi.height:5d} P(true)={hi['y'].mean() if hi.height else float('nan'):.3f}   |   p_yes < {1 - t if t > 0.5 else t:.2f}: n={lo.height:5d} P(true)={lo['y'].mean() if lo.height else float('nan'):.3f}")


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
