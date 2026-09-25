"""Experiment: recall@k of the rare-token inverted index on the same train sample as recall_lexical."""
import time

import polars as pl

from ber import io, normalize
from ber.blocking import keys

N_Q = 150_000
KS = (1, 3, 5, 10, 20)
COLS = ["idx", "country", "nm_core", "nm_url", "ad", "ad_words", "ad_nums"]

gt = io.load("train", "gt")
s1 = normalize.load("train", "s1", COLS)
x = normalize.load("train", "x", COLS)
for country in ("US", "India"):
    db = s1.filter(pl.col("country") == country)
    q = x.filter(pl.col("country") == country).join(gt.select(pl.col("x").alias("idx")), on="idx", how="semi").sample(
        N_Q, seed=0)
    g = gt.join(q.select(pl.col("idx").alias("x")), on="x", how="semi")
    for df_max in (100, 300, 1000):
        t = time.time()
        r = keys.retrieve(q, db, k=max(KS), df_max=df_max)
        j = g.join(r, on=["x", "s1"], how="left")
        print(f"{country:6s} keys df_max={df_max:5d} {time.time() - t:5.1f}s " + " ".join(
            f"R@{k}={(j['rank'].is_not_null() & (j['rank'] < k)).mean():.4f}" for k in KS), flush=True)
