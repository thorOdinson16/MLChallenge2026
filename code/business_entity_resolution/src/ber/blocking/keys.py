"""Inverted-index blocking on rare tokens (name words, address words, address numbers).

score(x, s1) = sum of idf over shared tokens whose S1 document frequency <= df_max.
Cheap, exact, and complementary to dense retrieval (house numbers, rare surnames, glued URLs).
"""
import numpy as np
import polars as pl


def record_tokens(d: pl.DataFrame) -> pl.DataFrame:
    """(idx, tok) with tokens typed by field so a name word never matches an address word."""
    name = d.select("idx", tok=pl.col("nm_core").str.split(" ")).explode("tok").filter(
        pl.col("tok").str.len_chars() >= 3).with_columns(tok="n:" + pl.col("tok"))
    url = d.filter(pl.col("nm_url") != "").select("idx", tok="n:" + pl.col("nm_url"))
    glued = d.select("idx", tok="n:" + pl.col("nm_core").str.replace_all(" ", "")).filter(
        pl.col("tok").str.len_chars() >= 8)
    aw = d.select("idx", tok=pl.col("ad_words")).explode("tok").filter(
        pl.col("tok").str.len_chars() >= 3).with_columns(tok="a:" + pl.col("tok"))
    an = d.select("idx", tok=pl.col("ad_nums")).explode("tok").filter(
        pl.col("tok").str.len_chars() >= 2).with_columns(tok="#:" + pl.col("tok"))
    return pl.concat([name, url, glued, aw, an]).drop_nulls().unique()


class KeyIndex:
    def __init__(self, db: pl.DataFrame, df_max: int = 1000):
        dt = record_tokens(db)
        n = db.height
        df = dt.group_by("tok").len("df").filter(pl.col("df") <= df_max).with_columns(
            w=(np.log(n) - pl.col("df").log()).cast(pl.Float32))
        self.post = dt.join(df.select("tok", "w"), on="tok").rename({"idx": "s1"})

    def query(self, q: pl.DataFrame, k: int, chunk: int = 100_000) -> pl.DataFrame:
        out = []
        for i in range(0, q.height, chunk):
            qt = record_tokens(q.slice(i, chunk)).rename({"idx": "x"})
            m = (qt.join(self.post, on="tok")
                   .group_by("x", "s1").agg(score=pl.col("w").sum())
                   .with_columns(rank=pl.col("score").rank("ordinal", descending=True).over("x") - 1)
                   .filter(pl.col("rank") < k)
                   .with_columns(pl.col("rank").cast(pl.UInt8)))
            out.append(m)
        return pl.concat(out).select("x", "s1", "score", "rank")


def retrieve(q: pl.DataFrame, db: pl.DataFrame, k: int, df_max: int = 300, chunk: int = 100_000) -> pl.DataFrame:
    return KeyIndex(db, df_max).query(q, k, chunk)
