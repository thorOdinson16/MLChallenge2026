"""Holdout check of ber.twin on top of the friend's stage 3 (overnight: train on full+rm22, expF, dup-propagation).
    python -m ber.experiments.twin_eval ce_v4,ce_xlmr_fr,ce_full
Saves the out-of-fold stage-3 holdout predictions to preds/s3oof_<eval>.parquet for reuse."""
import sys

import polars as pl

from .. import config, evaluate, io, split, twin
from . import overnight

if __name__ == "__main__":
    overnight.PARAMS["num_threads"] = 8
    tags = sys.argv[1].split(",")
    gt = io.load("train", "gt"); hold = pl.DataFrame({"s1": split.holdout_s1()})
    rm = io.load("train", "s1", ["idx"]).rename({"idx": "s1"}).sample(fraction=0.19, seed=11)
    kept = {"full": hold, "rm11": hold.join(rm, on="s1", how="anti")}
    G = overnight.holdout_frames(tags)
    cols = overnight.BASE_COLS + overnight.ce_cols(tags) + overnight.raw_cols()
    Kt = overnight.dup_keys("train")
    gty = gt.with_columns(y=pl.lit(1))
    for ev in ("full", "rm11"):
        pr = overnight.cv_predict(G, cols, ("full", "rm22"), ev, hold)
        pr.write_parquet(config.path("preds", f"s3oof_{ev}.parquet"))
        base = overnight.propagate(overnight.select(pr, "expF"), pr, Kt)
        halves = [kept[ev].filter((pl.col("s1") // 10) % 2 == h)["s1"] for h in (0, 1)]
        b = [evaluate.macro_f05(base, gt, s) for s in halves]
        overnight.log(f"twin_eval {ev}: base A {b[0]:.5f} B {b[1]:.5f}")
        for sim in (85, 90, 95):
            for pmin in (0.1, 0.2, 0.3, 0.5):
                s = twin.rescue(base, pr, "train", "p3", sim, pmin, verbose=False)
                add = s.join(base, on=["s1", "x"], how="anti").join(gty, on=["s1", "x"], how="left")
                f = [evaluate.macro_f05(s, gt, h) for h in halves]
                overnight.log(f"   {ev} sim>={sim} p>={pmin}: add {add.height:5d} prec {add['y'].fill_null(0).mean() if add.height else 0:.3f}"
                              f"  A {f[0] - b[0]:+.5f}  B {f[1] - b[1]:+.5f}")
