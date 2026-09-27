"""Proxy for adapting the v2 matcher to an unseen country with pseudo-labels (India plays France).

A: v2 trained on US only (ber.experiments.v2_us_only -> preds/train_m_v2us.parquet).
B: v2 trained on US true labels + India pseudo-labels from A (p2 >= HI -> 1, p2 <= LO -> 0, ambiguous India rows dropped).
Reports macro-F0.5 on the India (and US) holdout for A, B and the fully supervised shipped v2 (upper bound).

    python -m ber.experiments.v2_pseudo_proxy
"""
import sys

import polars as pl

from .. import config, decode, evaluate, io, matcher2, split

HI, LO = 0.9, 0.05


def score(pred, gt, meta, country, thr=0.8):
    s = meta.filter(pl.col("country") == country)["s1"]
    p = pred.join(pl.DataFrame({"s1": s}), on="s1", how="semi")
    ev_x = p.select("x").unique()
    q = pred.join(ev_x, on="x", how="semi")
    sel = decode.apply_caps(decode.assign_unique(q, "p2").filter(pl.col("p2") >= thr), "p2").select("s1", "x")
    return evaluate.macro_f05(sel, gt, s)


def main():
    P = config.path
    us = pl.read_parquet(P("preds", "train_m_v2us.parquet"), columns=["x", "s1", "src", "country", "p2"])
    ps = us.filter(pl.col("country") == "India").select("x", "s1", pl.col("p2").alias("p2_us"))
    orig = matcher2.train_keys

    def keys(name="train_all_v2"):
        k = orig(name).join(ps, on=["x", "s1"], how="left")
        ind = pl.col("country") == "India"
        conf = (pl.col("p2_us") >= HI) | (pl.col("p2_us") <= LO)
        k = k.with_columns(y=pl.when(ind).then((pl.col("p2_us") >= HI).cast(pl.UInt8)).otherwise(pl.col("y")),
                           trainable=pl.col("trainable") & (~ind | conf))
        print(f"trainable: US {k.filter(pl.col('trainable') & ~ind).height}, India pseudo {k.filter(pl.col('trainable') & ind).height}", flush=True)
        return k.drop("p2_us")

    matcher2.train_keys = keys
    matcher2.TAG = "m_v2usps"
    matcher2.train()
    gt = io.load("train", "gt")
    hold = pl.DataFrame({"s1": split.holdout_s1()})
    meta = io.load("train", "s1", ["idx", "country"]).rename({"idx": "s1"}).join(hold, on="s1", how="semi")
    for tag, lab in (("m_v2us", "A  US-only v2 (India unseen)"), ("m_v2usps", "B  US + India pseudo-labels"), ("m_v2", "UB shipped v2 (true India labels)")):
        pr = pl.read_parquet(P("preds", f"train_{tag}.parquet"), columns=["x", "s1", "src", "p2"])
        print(f"{lab:36s} India holdout {score(pr, gt, meta, 'India'):.5f}   US holdout {score(pr, gt, meta, 'US'):.5f}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
