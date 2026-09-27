"""Unseen-country proxy, stage-3 part: how much should stage 3 trust the adapted cross-encoder vs the unadapted matcher?

Matcher = US-only v2 (preds/train_m_v2us.parquet), CEs from ber.experiments.trust_proxy_ce.  Stage 3 is trained on
US holdout rows only and evaluated on the India holdout (India never seen by matcher or stage 3, like France):
  A  unadapted CE (prx_us)          B  adapted CE (prx_ps)
  C  adapted CE, no matcher-probability features (trust the CE fully)
  D  blends of B and C in logit space
Reference: US-only matcher p2 alone, and the shipped fully supervised v2.

    python -m ber.experiments.trust_proxy_s3
"""
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

from .. import config, decode, evaluate, io, matcher2, split
from .overnight import BASE_COLS, PARAMS, ROUNDS, build, ce_cols, raw_cols

P = config.path
P2COLS = ["l1", "l2", "rank_p2_x", "max_p2_x", "sum_p2_x", "sum_p2_s_other", "n_conf_s_src_other", "gap_p2_x"]


def main():
    gt = io.load("train", "gt").with_columns(y=pl.lit(1, pl.UInt8))
    hold = pl.DataFrame({"s1": split.holdout_s1()})
    meta = io.load("train", "s1", ["idx", "country"]).rename({"idx": "s1"}).join(hold, on="s1", how="semi")
    tr = pl.read_parquet(P("preds", "train_m_v2us.parquet"), columns=["x", "s1", "src", "country", "p1", "p2"])
    ev_x = tr.join(hold, on="s1", how="semi").select("x").unique()
    h = tr.join(ev_x, on="x", how="semi")
    rc = raw_cols()
    raw = pl.concat([pl.read_parquet(p, columns=["x", "s1"] + rc).join(h.select("x", "s1"), on=["x", "s1"], how="semi")
                     for p in matcher2.parts("train_all_v2")])
    ind_s1 = meta.filter(pl.col("country") == "India")["s1"]
    us_s1 = meta.filter(pl.col("country") == "US")

    def frame(tag):
        d = build(h, "train", [tag], "holdout").join(raw, on=["x", "s1"], how="left")
        return d.join(gt, on=["s1", "x"], how="left").with_columns(pl.col("y").fill_null(0))

    def sel_score(d, col, rule=0.75):
        base = decode.apply_caps(decode.assign_unique(d.select("x", "s1", "src", pl.col(col).alias("p")), "p").filter(pl.col("p") >= 0.02), "p")
        sel = decode.decode(base.select("x", "s1", "p"), "p", min_p=0.02) if rule == "expF" else base.filter(pl.col("p") >= rule).select("s1", "x")
        return evaluate.macro_f05(sel, gt, ind_s1)

    def fit(d, cols):
        t = d.join(us_s1, on="s1", how="semi")
        m = lgb.train(PARAMS, lgb.Dataset(t.select(cols).to_numpy().astype(np.float32), t["y"].to_numpy(), feature_name=cols), ROUNDS)
        return m.predict(d.select(cols).to_numpy().astype(np.float32))

    print(f"reference  US-only matcher p2 alone (thr .8): India {sel_score(h, 'p2', 0.8):.5f}", flush=True)
    dA, dB = frame("prx_us"), frame("prx_ps")
    res = {}
    pA = fit(dA, BASE_COLS + ce_cols(["prx_us"]) + rc)
    pB = fit(dB, BASE_COLS + ce_cols(["prx_ps"]) + rc)
    pC = fit(dB, [c for c in BASE_COLS if c not in P2COLS] + [f"ce_prx_ps", f"cemax_prx_ps", f"cegap_prx_ps"] + rc)
    d = dB.with_columns(pA=pl.Series(pA), pB=pl.Series(pB), pC=pl.Series(pC))
    lg = lambda c: (pl.col(c).clip(1e-6, 1 - 1e-6) / (1 - pl.col(c).clip(1e-6, 1 - 1e-6))).log()
    for w in (0.25, 0.5, 0.75, 0.9):
        d = d.with_columns((1 / (1 + (-((1 - w) * lg("pB") + w * lg("pC"))).exp())).alias(f"pD{w}"))
    for col, lab in [("pA", "A  unadapted CE"), ("pB", "B  adapted CE (like sub_K France)"), ("pC", "C  adapted CE, no matcher features"),
                     ("pD0.25", "D  blend B/C w=0.25"), ("pD0.5", "D  blend B/C w=0.5"), ("pD0.75", "D  blend B/C w=0.75"), ("pD0.9", "D  blend B/C w=0.9")]:
        res[col] = {r: sel_score(d, col, r) for r in (0.75, 0.85, 0.9, 0.93, 0.95, "expF")}
        print(f"{lab:36s} India holdout: " + "  ".join(f"{r}={v:.5f}" for r, v in res[col].items()), flush=True)
    print("upper bound (shipped v2 + ce_v2 stage 3, India holdout) ~0.9915", flush=True)


if __name__ == "__main__":
    sys.exit(main())
