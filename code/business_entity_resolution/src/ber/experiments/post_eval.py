"""Holdout test of the post-rules on top of the real stage 3 (overnight: train on full+rm22, expF, dup-propagation).
    python -m ber.experiments.post_eval ce_v4,ce_xlmr_fr,ce_full[,ce_full2]
Rules (US/India, labelled holdout, halves A/B = (s1 // 10) % 2, full and thinned rm11 pools):
  twin      : ber.twin.rescue (near-duplicate X at one address)
  tie       : drop an accepted pair whose X's top p3 share p1st/(p1st+p2nd) < t
  cot_drop  : drop an accepted pair whose ce_<tag> logit <= d
  cot_add   : add the top pair of an unassigned X when p3 >= 0.3 and ce_<tag> >= a (caps respected)
Saves the out-of-fold stage-3 predictions to preds/s3oof_<pool>.parquet.
"""
import sys

import polars as pl

from .. import config, decode, evaluate, io, split, twin
from . import overnight


def top2(pr):
    d = pr.sort(["x", "p3"], descending=[False, True]).with_columns(r=pl.col("p3").cum_count().over("x"))
    a = d.filter(pl.col("r") == 1).select("x", "s1", "src", "p3")
    b = d.filter(pl.col("r") == 2).select("x", p2nd=pl.col("p3"))
    return a.join(b, on="x", how="left").with_columns(pl.col("p2nd").fill_null(0.0)).with_columns(
        share=pl.col("p3") / (pl.col("p3") + pl.col("p2nd")))


if __name__ == "__main__":
    overnight.PARAMS["num_threads"] = 8
    tags = sys.argv[1].split(",")
    gt = io.load("train", "gt"); hold = pl.DataFrame({"s1": split.holdout_s1()})
    rm = io.load("train", "s1", ["idx"]).rename({"idx": "s1"}).sample(fraction=0.19, seed=11)
    kept = {"full": hold, "rm11": hold.join(rm, on="s1", how="anti")}
    G = overnight.holdout_frames(tags)
    cols = overnight.BASE_COLS + overnight.ce_cols(tags) + overnight.raw_cols()
    Kt = overnight.dup_keys("train")
    # external views: a CE NOT inside the stage-3 tags mimics France (ce_full2 is not inside the W1 blend)
    ce = {t: pl.read_parquet(config.path("preds", f"{t}_holdout.parquet")) for t in ("ce_full", "ce_full2")}
    for ev in ("full", "rm11"):
        pr = overnight.cv_predict(G, cols, ("full", "rm22"), ev, hold)
        pr.write_parquet(config.path("preds", f"s3oof_{ev}.parquet"))
        base = overnight.propagate(overnight.select(pr, "expF"), pr, Kt)
        halves = [kept[ev].filter((pl.col("s1") // 10) % 2 == h)["s1"] for h in (0, 1)]
        b = [evaluate.macro_f05(base, gt, h) for h in halves]
        overnight.log(f"post_eval {ev}: base expF+dup  A {b[0]:.5f}  B {b[1]:.5f}")
        T = top2(pr)

        def report(name, sel):
            f = [evaluate.macro_f05(sel, gt, h) for h in halves]
            overnight.log(f"   {ev} {name:28s} A {f[0] - b[0]:+.5f}  B {f[1] - b[1]:+.5f}   pairs {sel.height - base.height:+d}")
        report("twin", twin.rescue(base, pr, "train", "p3", verbose=False))
        for t in (0.55, 0.6, 0.65):
            bad = T.filter(pl.col("share") < t).select("x", "s1")
            report(f"tie share<{t}", base.join(bad, on=["x", "s1"], how="anti"))
        for tag, c in ce.items():
            for d in (-3.0, -4.0, -5.0):
                bad = c.filter(pl.col("ce") <= d).select("x", "s1")
                report(f"cot_drop {tag}<={d}", base.join(bad, on=["x", "s1"], how="anti"))
            for a in (3.0, 4.0, 5.0):
                cand = (T.join(base.select("x"), on="x", how="anti").filter(pl.col("p3") >= 0.3)
                         .join(c.filter(pl.col("ce") >= a), on=["x", "s1"]))
                both = pl.concat([base.join(pr.select("x", "s1", "src", "p3"), on=["x", "s1"], how="left")
                                  .with_columns(k=pl.col("p3").fill_null(1.0) + 2.0).select("s1", "x", "src", "k"),
                                  cand.select("s1", "x", "src", k=pl.col("p3"))])
                report(f"cot_add {tag}>={a}", decode.apply_caps(both, "k").select("s1", "x"))
