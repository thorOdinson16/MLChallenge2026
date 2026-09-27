"""Thinned-pool simulation for stage 3.

The test S1 pool is thinner than train (5.76 X per S1 vs 4.67): about 19% of the entities behind the S2/S3 records are
absent, so their records are orphans that look like real variants of a present namesake.  To train and select stage 3
under those conditions, remove 19% of train S1 from the holdout candidate graph, RECOMPUTE v2's stage-2 graph context
without them (on the 2-hop graph, so per-S1 aggregates stay complete) and re-predict p2 with the shipped m_v2_s2 model.

Writes preds/scratch_sim_{full,rm11,rm22}.parquet (x, s1, src, country, p1, p2) for the holdout X; the full-graph
re-prediction reproduces the stored p2 exactly (sanity check printed).  Used by ber.experiments.overnight.

    python -m ber.experiments.thinned_sim
"""
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from .. import config, io, matcher2, split

P = config.path


def main():
    t0 = time.time()
    tr = pl.read_parquet(P("preds", "train_m_v2.parquet"))
    hold = pl.DataFrame({"s1": split.holdout_s1()})
    ev_x = tr.join(hold, on="s1", how="semi").select("x").unique()
    h = tr.join(ev_x, on="x", how="semi")
    h2 = tr.join(h.select("s1").unique(), on="s1", how="semi")          # 2-hop: every row of every S1 touched
    h2 = tr.join(h2.select("x").unique(), on="x", how="semi")
    m2 = lgb.Booster(model_file=str(P("models", "m_v2_s2.txt")))
    cols2 = m2.feature_name()
    rawc = [c for c in cols2 if not c.startswith("c_")]
    raw = pl.concat([pl.read_parquet(p, columns=["x", "s1"] + rawc).join(h2.select("x", "s1"), on=["x", "s1"], how="semi")
                     for p in matcher2.parts("train_all_v2")])
    all_s1 = io.load("train", "s1", ["idx"]).rename({"idx": "s1"})

    def repredict(hh, tag):
        ctx = pl.concat([matcher2.context(g.select("x", "s1", "src", "p1"), "train") for _, g in hh.group_by("country")])
        f = hh.select("x", "s1").join(raw, on=["x", "s1"], how="left").join(ctx, on=["x", "s1"], how="left")
        p2 = m2.predict(f.select(cols2).to_numpy().astype(np.float32), num_threads=12).astype(np.float32)
        out = (hh.drop("p2").join(f.select("x", "s1").with_columns(p2=pl.Series(p2)), on=["x", "s1"])
                 .join(h.select("x", "s1"), on=["x", "s1"], how="semi"))
        out.write_parquet(P("preds", f"scratch_sim_{tag}.parquet"))
        print(f"  {tag}: re-predicted {out.height} rows {time.time() - t0:.0f}s", flush=True)
        return out

    full = repredict(h2, "full")
    chk = full.join(h.select("x", "s1", pl.col("p2").alias("p2_old")), on=["x", "s1"])
    print("  full-graph re-prediction vs stored p2: max abs diff", float((chk["p2"] - chk["p2_old"]).abs().max()), flush=True)
    for seed in (11, 22):
        rm = all_s1.sample(fraction=0.19, seed=seed)
        repredict(h2.join(rm, on="s1", how="anti"), f"rm{seed}")


if __name__ == "__main__":
    sys.exit(main())
