"""Address rescue for replaced-name X (rule C): an unassigned X whose name is a single invented word (the generator's
'fully replaced name', e.g. 'Gildlyra') is matched to the ONLY S1 of the pool at exactly its address.

Pool search: key (country, first house number, one shared address word), then the strict address check of
ber.acronym (every X street/city word fuzzy-matches an S1 address word), and the S1 must be unique for that address.
Measured on the labelled US/India holdout (out-of-fold stage-3 p3) before use on France.

    python -m ber.addr_rescue holdout
    python -m ber.addr_rescue pairs <france_probs.parquet> <pcol> [thr=0.85]   -> preds/fr_addr_pairs.parquet
"""
import sys

import polars as pl

from . import config, io
from .acronym import STREET, _addr_ok

P = config.path
ONEWORD = r"^[^\s@#\.]{4,}$"          # a single token, not an acronym-length string, not a URL / handle


def _x(split_name, xs, countries):
    xr = pl.scan_parquet(P("raw", f"{split_name}_x.parquet")).select(pl.col("idx").alias("x"), "name").join(
        xs.lazy(), on="x", how="semi").collect()
    xr = xr.filter(pl.col("name").str.strip_chars().str.contains(ONEWORD) &
                   ~pl.col("name").str.strip_chars().str.contains(r"^[A-Z0-9&\.]+$"))
    xn = pl.scan_parquet(P("norm", f"{split_name}_x.parquet")).select(
        pl.col("idx").alias("x"), "country", pl.col("ad_words").alias("xw"), pl.col("ad_nums").alias("xnum")).join(
        xr.lazy().select("x"), on="x", how="semi").collect()
    return xn.filter(pl.col("country").is_in(countries)).with_columns(n1=pl.col("xnum").list.first()).filter(
        pl.col("n1").is_not_null())


def rules(split_name, probs, pcol, thr, countries):
    top = probs.group_by("x").agg(best=pl.col(pcol).max())
    xs = _x(split_name, top.filter(pl.col("best") < thr).select("x"), countries)
    s = pl.scan_parquet(P("norm", f"{split_name}_s1.parquet")).select(
        pl.col("idx").alias("s1"), "country", pl.col("ad_words").alias("sw"), pl.col("ad_nums").alias("snum")).filter(
        pl.col("country").is_in(countries)).collect().with_columns(n1=pl.col("snum").list.first()).filter(
        pl.col("n1").is_not_null())
    street = list(STREET)
    xe = xs.select("x", "country", "n1", w=pl.col("xw")).explode("w").filter(~pl.col("w").is_in(street) & (pl.col("w").str.len_chars() >= 4))
    se = s.select("s1", "country", "n1", w=pl.col("sw")).explode("w").filter(~pl.col("w").is_in(street) & (pl.col("w").str.len_chars() >= 4))
    pr = xe.join(se, on=["country", "n1", "w"]).select("x", "s1").unique()
    d = pr.join(xs.select("x", "xw", "xnum"), on="x").join(s.select("s1", "sw", "snum"), on="s1")
    ok = [_addr_ok(a, b, c, e) for a, b, c, e in zip(d["xw"].to_list(), d["sw"].to_list(), d["xnum"].to_list(),
                                                     d["snum"].to_list())]
    d = d.with_columns(ok=pl.Series(ok, dtype=pl.Boolean)).filter(pl.col("ok"))
    d = d.with_columns(k=pl.len().over("x")).filter(pl.col("k") == 1)          # the only S1 at that address
    return d.select("x", "s1").with_columns(rule=pl.lit("C"))


def holdout():
    h = pl.read_parquet(P("preds", "s3oof_full.parquet"), columns=["x", "s1", "p3"])
    r = rules("train", h, "p3", 0.8, ["US", "India"])
    r = r.join(io.load("train", "gt").with_columns(y=pl.lit(1)), on=["s1", "x"], how="left").with_columns(
        pl.col("y").fill_null(0))
    new = r.join(h.select("x", "s1"), on=["x", "s1"], how="anti")
    print(f"holdout rule C: {r.height} pairs, precision {r['y'].mean() if r.height else 0:.3f}; of which new candidates "
          f"{new.height}, precision {new['y'].mean() if new.height else 0:.3f}")


def pairs(prob_file, pcol="pD", thr=0.85):
    cols = ["x", "s1", pcol] + (["ins_country_word"] if "ins_country_word" in pl.read_parquet_schema(prob_file) else [])
    t = pl.read_parquet(prob_file, columns=cols)
    if "ins_country_word" in t.columns:
        t = t.filter(pl.col("ins_country_word").fill_null(0) < 0.5)
    r = rules("test", t.select("x", "s1", pcol), pcol, thr, ["France"])
    s1 = io.load("test", "s1", ["idx", "eid"]).rename({"idx": "s1", "eid": "source1_entity_id"})
    xid = io.load("test", "x", ["idx", "eid"]).rename({"idx": "x", "eid": "xe"})
    out = r.join(s1, on="s1").join(xid, on="x").select("source1_entity_id", "xe", "rule")
    out.write_parquet(P("preds", "fr_addr_pairs.parquet"))
    print(f"France rule C: {out.height} pairs (new candidates: "
          f"{r.join(t.select('x', 's1'), on=['x', 's1'], how='anti').height})", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "holdout":
        holdout()
    else:
        pairs(sys.argv[2], sys.argv[3], *[float(v) for v in sys.argv[4:5]])
