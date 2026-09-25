"""Candidate generation: union of retrievers, X-centric (each S2/S3 record -> top-k S1 of same country).

Streams X in chunks so memory stays bounded. For each chunk: run every retriever, outer-join into one
frame with {r}_score / {r}_rank columns (null when r did not propose the pair), then optionally prune
with the stage-0 model (prune.py) and write a part file:
    WORK/cands/{out}/{country}_{chunk:03d}.parquet
"""
import shutil
import time

import numpy as np
import polars as pl

from . import config, normalize
from .blocking import dense, keys

# retriever name -> k kept from that retriever
K = {"lsa_full": 20, "lsa_addr": 10, "lsa_name": 10, "keys": 10, "enc": 20}
RETRIEVERS = ("lsa_full", "lsa_addr", "lsa_name", "keys", "enc")
COLS = ["idx", "country", "nm", "nm_core", "nm_url", "ad", "ad_words", "ad_nums", "ad_state"]
LSA_TEXT = {
    "lsa_full": lambda d: (d["nm_core"] + " | " + d["ad"]).to_list(),
    "lsa_addr": lambda d: d["ad"].to_list(),
    "lsa_name": lambda d: d["nm_core"].to_list(),
}


class Retrievers:
    """Per-country retriever state (fitted LSA models, DB embeddings, key index, encoder)."""

    def __init__(self, db: pl.DataFrame, q_sample: pl.DataFrame, retrievers, enc_tag):
        self.db, self.retrievers = db, retrievers
        self.db_idx = db["idx"].to_numpy()
        self.lsa, self.E = {}, {}
        for r in retrievers:
            if r.startswith("lsa"):
                f = LSA_TEXT[r]
                self.lsa[r] = dense.LSA().fit(f(db) + f(q_sample))
                self.E[r] = self.lsa[r].transform(f(db))
            elif r == "keys":
                self.keys = keys.KeyIndex(db, df_max=1000)
            elif r == "enc":
                from .blocking import encoder
                self.enc = encoder.load(enc_tag)
                self.E[r] = encoder.encode(self.enc, encoder.record_text(db))

    def run(self, q: pl.DataFrame) -> pl.DataFrame:
        parts = []
        qi = q["idx"].to_numpy()
        for r in self.retrievers:
            if r.startswith("lsa"):
                I, S = dense.gpu_topk(self.lsa[r].transform(LSA_TEXT[r](q)), self.E[r], K[r])
                f = dense.topk_frame(qi, self.db_idx, I, S)
            elif r == "keys":
                f = self.keys.query(q, K[r])
            else:
                from .blocking import encoder
                I, S = dense.gpu_topk(encoder.encode(self.enc, encoder.record_text(q)), self.E[r], K[r])
                f = dense.topk_frame(qi, self.db_idx, I, S)
            parts.append(f.filter(pl.col("score") > -1e9).select(
                "x", "s1", pl.col("score").alias(f"{r}_score"), pl.col("rank").alias(f"{r}_rank")))
        u = parts[0]
        for f in parts[1:]:
            u = u.join(f, on=["x", "s1"], how="full", coalesce=True)
        rank_cols = [f"{r}_rank" for r in self.retrievers]
        return u.with_columns(best_rank=pl.min_horizontal(rank_cols),
                              n_ret=pl.sum_horizontal([pl.col(c).is_not_null() for c in rank_cols]).cast(pl.UInt8))


def generate(split: str, out: str, retrievers=RETRIEVERS, enc_tag="enc_v1", x_subset: pl.Series | None = None,
             prune_k: int | None = None, chunk: int = 1_000_000, countries=None):
    from . import prune
    pruner = prune.load_model() if prune_k else None
    d = config.path("cands", out, "x")
    shutil.rmtree(d.parent, ignore_errors=True)
    d.parent.mkdir(parents=True, exist_ok=True)
    s1 = normalize.load(split, "s1", COLS)
    x = normalize.load(split, "x", COLS)
    if x_subset is not None:
        x = x.join(pl.DataFrame({"idx": x_subset}).cast(pl.UInt32), on="idx", how="semi")
    countries = countries or sorted(s1["country"].unique().to_list())
    for c in countries:
        t0 = time.time()
        db = s1.filter(pl.col("country") == c)
        q = x.filter(pl.col("country") == c)
        R = Retrievers(db, q.sample(min(q.height, 400_000), seed=config.SEED), retrievers, enc_tag)
        print(f"  {split} {c}: retrievers ready in {time.time() - t0:.0f}s", flush=True)
        n = 0
        for i in range(0, q.height, chunk):
            u = R.run(q.slice(i, chunk))
            n_raw = u.height
            if pruner is not None:
                u = prune.prune(u, pruner, prune_k)
            u.write_parquet(config.path("cands", out, f"{c}_{i // chunk:03d}.parquet"))
            n += u.height
            print(f"  {split} {c} chunk {i // chunk}: {n_raw} -> {u.height} pairs ({time.time() - t0:.0f}s)",
                  flush=True)
        print(f"{split} {c}: {n} pairs, {n / max(q.height, 1):.1f}/x, {time.time() - t0:.0f}s", flush=True)
        del R


def load(out: str, columns=None) -> pl.DataFrame:
    return pl.concat([pl.read_parquet(p, columns=columns) for p in sorted(config.path("cands", out).glob("*.parquet"))],
                     how="diagonal_relaxed")
