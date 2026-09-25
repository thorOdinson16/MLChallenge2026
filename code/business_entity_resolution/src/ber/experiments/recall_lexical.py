"""Experiment: recall@k of LSA dense views on a train sample (full S1 pool as targets)."""
import time

import polars as pl

from ber import io, normalize
from ber.blocking import dense

N_Q = 150_000
KS = (1, 3, 5, 10, 20, 50)
VIEWS = {
    "name": lambda d: d["nm_core"],
    "addr": lambda d: d["ad"],
    "full": lambda d: d["nm_core"] + " | " + d["ad"],
}

gt = io.load("train", "gt")
COLS = ["idx", "country", "nm_core", "ad"]
s1 = normalize.load("train", "s1", COLS)
x = normalize.load("train", "x", COLS)
for country in ("US", "India"):
    db = s1.filter(pl.col("country") == country)
    q = x.filter(pl.col("country") == country).join(gt.select(pl.col("x").alias("idx")), on="idx", how="semi").sample(
        N_Q, seed=0)
    g = gt.join(q.select(pl.col("idx").alias("x")), on="x", how="semi")
    hits = {}
    for view, col in VIEWS.items():
        t = time.time()
        dbt, qt = col(db).to_list(), col(q).to_list()
        lsa = dense.LSA().fit(dbt + qt[:100_000])
        E_db, E_q = lsa.transform(dbt), lsa.transform(qt)
        t1 = time.time()
        I, S = dense.gpu_topk(E_q, E_db, max(KS))
        r = dense.topk_frame(q["idx"].to_numpy(), db["idx"].to_numpy(), I, S)
        j = g.join(r, on=["x", "s1"], how="left")
        print(f"{country:6s} {view:5s} embed {t1 - t:5.1f}s search {time.time() - t1:5.1f}s " + " ".join(
            f"R@{k}={(j['rank'].is_not_null() & (j['rank'] < k)).mean():.4f}" for k in KS), flush=True)
        hits[view] = r
    for combo in (("name", "addr"), ("name", "addr", "full")):
        for k in (5, 10, 20, 50):
            u = pl.concat([hits[v].filter(pl.col("rank") < k).select("x", "s1") for v in combo]).unique()
            rec = g.join(u, on=["x", "s1"], how="semi").height / g.height
            print(f"{country:6s} union{combo} @{k}: recall={rec:.4f}  cands/x={u.height / q.height:.1f}", flush=True)
