"""Experiment: recall@k of the fine-tuned bi-encoder on HOLDOUT X records (never seen in training)."""
import sys
import time

import polars as pl

from ber import io, normalize, split
from ber.blocking import dense, encoder

TAG = sys.argv[1] if len(sys.argv) > 1 else "enc_v1"
N_Q = 100_000
KS = (1, 3, 5, 10, 20, 50)
COLS = ["idx", "country", "nm", "ad", "ad_state"]

gt = io.load("train", "gt").join(pl.DataFrame({"s1": split.holdout_s1()}), on="s1", how="semi")
s1 = normalize.load("train", "s1", COLS)
x = normalize.load("train", "x", COLS)
m = encoder.load(TAG)
for country in ("US", "India"):
    db = s1.filter(pl.col("country") == country)
    q = x.filter(pl.col("country") == country).join(gt.select(pl.col("x").alias("idx")), on="idx", how="semi")
    q = q.sample(min(N_Q, q.height), seed=0)
    g = gt.join(q.select(pl.col("idx").alias("x")), on="x", how="semi")
    t = time.time()
    Ed = encoder.encode(m, encoder.record_text(db))
    Eq = encoder.encode(m, encoder.record_text(q))
    t1 = time.time()
    I, S = dense.gpu_topk(Eq, Ed, max(KS))
    r = dense.topk_frame(q["idx"].to_numpy(), db["idx"].to_numpy(), I, S)
    j = g.join(r, on=["x", "s1"], how="left")
    print(f"{country:6s} enc encode {t1 - t:.0f}s ({(db.height + q.height) / (t1 - t):.0f}/s) search {time.time() - t1:.0f}s "
          + " ".join(f"R@{k}={(j['rank'].is_not_null() & (j['rank'] < k)).mean():.4f}" for k in KS), flush=True)
