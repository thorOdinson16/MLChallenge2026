"""No-house-number rescue (rule N): unassigned X whose address has street words but no house number, matched by a
whole-pool search (retrieval misses them: France X without a number are assigned 88.2% vs US 96.7%, and most unmatched
ones have no plausible candidate at all).

Pool search keys: the X's rare address words (S1 document frequency <= DF_MAX, not street types) and their 5-char
prefixes (typo tolerance: 'calembetr' ~ 'calembert'). Filters:
  * street: every X address word except street types fuzzy-matches an S1 address word (Jaro-Winkler >= 0.85);
  * name  : token_sort_ratio(nm_core) >= NAME_MIN. token_sort, NOT token_set: token_set scores 100 when one name
            contains the other, which lets 'Dupont Holding' decoys through (inserted business words are decoys:
            US/IN holdout precision 0.0-2.5%);
  * unique: exactly one S1 per X survives.
Validated on US/India no-number X of the labelled holdout (out-of-fold stage-3 p3) before use on France.

    python -m ber.nonum holdout
    python -m ber.nonum pairs <france_probs.parquet> <pcol> [thr=0.85]      -> preds/fr_nonum_pairs.parquet
    python -m ber.cn_rescue <in.tsv> <out.tsv> fr_nonum_pairs.parquet        (adds for unassigned X, caps, France only)
New candidates must also be added to candidate_pairs.tsv:  python -m ber.nonum cands <cand.tsv> <out.tsv>
(acronym + nonum together: run ber.acronym cands twice, the second time with fr_nonum_pairs.parquet)
"""
import sys

import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from . import config, io
from .acronym import ACR, STREET, _addr_ok

P = config.path
DF_MAX, NAME_MIN = 300, 90


def _keys(df, wcol, idcol, rare):
    e = df.select(idcol, "country", w=pl.col(wcol)).explode("w").drop_nulls("w")
    e = e.filter(~pl.col("w").is_in(list(STREET)) & (pl.col("w").str.len_chars() >= 4) & ~pl.col("w").str.contains(r"^\d"))
    e = e.join(rare, on=["country", "w"], how="semi")
    return pl.concat([e.select(idcol, "country", k=pl.col("w")),
                      e.filter(pl.col("w").str.len_chars() >= 6).select(idcol, "country", k=pl.col("w").str.slice(0, 5) + "~")]).unique()


def rules(split_name, probs, pcol, thr, countries):
    top = probs.group_by("x").agg(best=pl.col(pcol).max())
    un = top.filter(pl.col("best") < thr).select("x")
    xr = pl.scan_parquet(P("raw", f"{split_name}_x.parquet")).select(pl.col("idx").alias("x"), "name").join(
        un.lazy(), on="x", how="semi").collect().filter(~pl.col("name").str.strip_chars().str.contains(ACR))
    x = pl.scan_parquet(P("norm", f"{split_name}_x.parquet")).select(
        pl.col("idx").alias("x"), "country", pl.col("nm_core").alias("xc"), pl.col("ad_words").alias("xw"),
        pl.col("ad_nums").alias("xnum")).join(xr.lazy().select("x"), on="x", how="semi").collect()
    x = x.filter(pl.col("country").is_in(countries) & (pl.col("xnum").list.len() == 0) & (pl.col("xw").list.len() > 0))
    s = pl.scan_parquet(P("norm", f"{split_name}_s1.parquet")).select(
        pl.col("idx").alias("s1"), "country", pl.col("nm_core").alias("sc"), pl.col("ad_words").alias("sw"),
        pl.col("ad_nums").alias("snum")).filter(pl.col("country").is_in(countries)).collect()
    df = s.select("country", w=pl.col("sw").list.unique()).explode("w").group_by("country", "w").len("df")
    rare = df.filter(pl.col("df") <= DF_MAX).select("country", "w")
    # X side: keep every non-street word as a key if rare in S1 OR absent (typo); S1 side: rare words only
    xk = _keys(x, "xw", "x", pl.concat([rare, x.select("country", w=pl.col("xw")).explode("w").join(
        df.select("country", "w"), on=["country", "w"], how="anti").unique()]))
    sk = _keys(s, "sw", "s1", rare)
    pr = xk.join(sk, on=["country", "k"]).select("x", "s1").unique()
    d = pr.join(x.select("x", "xc", "xw", "xnum"), on="x").join(s.select("s1", "sc", "sw", "snum"), on="s1")
    d = d.with_columns(tsort=pl.Series(cpdist(d["xc"].to_list(), d["sc"].to_list(), scorer=fuzz.token_sort_ratio,
                                              workers=4), dtype=pl.Float32)).filter(pl.col("tsort") >= NAME_MIN)
    ok = [_addr_ok(a, b, c, e) for a, b, c, e in zip(d["xw"].to_list(), d["sw"].to_list(), d["xnum"].to_list(),
                                                     d["snum"].to_list())]
    d = d.with_columns(ok=pl.Series(ok, dtype=pl.Boolean)).filter(pl.col("ok"))
    d = d.with_columns(k=pl.len().over("x")).filter(pl.col("k") == 1)
    return d.select("x", "s1", "tsort").with_columns(rule=pl.lit("N"))


def holdout():
    h = pl.read_parquet(P("preds", "s3oof_full.parquet"), columns=["x", "s1", "p3"])
    r = rules("train", h, "p3", 0.8, ["US", "India"])
    r = r.join(io.load("train", "gt").with_columns(y=pl.lit(1)), on=["s1", "x"], how="left").with_columns(
        pl.col("y").fill_null(0))
    new = r.join(h.select("x", "s1"), on=["x", "s1"], how="anti")
    print(f"holdout rule N: {r.height} pairs, precision {r['y'].mean() if r.height else 0:.3f} | new candidates "
          f"{new.height}, precision {new['y'].mean() if new.height else 0:.3f}")
    for lo in (90, 95, 100):
        g = r.filter(pl.col("tsort") >= lo)
        print(f"   token_sort >= {lo}: n={g.height} precision {g['y'].mean() if g.height else 0:.3f}")


def pairs(prob_file, pcol="pD", thr=0.85):
    cols = ["x", "s1", pcol] + (["ins_country_word"] if "ins_country_word" in pl.read_parquet_schema(prob_file) else [])
    t = pl.read_parquet(prob_file, columns=cols)
    if "ins_country_word" in t.columns:
        t = t.filter(pl.col("ins_country_word").fill_null(0) < 0.5)
    r = rules("test", t.select("x", "s1", pcol), pcol, thr, ["France"])
    s1 = io.load("test", "s1", ["idx", "eid"]).rename({"idx": "s1", "eid": "source1_entity_id"})
    xid = io.load("test", "x", ["idx", "eid"]).rename({"idx": "x", "eid": "xe"})
    out = r.join(s1, on="s1").join(xid, on="x").select("source1_entity_id", "xe", "rule")
    out.write_parquet(P("preds", "fr_nonum_pairs.parquet"))
    print(f"France rule N: {out.height} pairs (new candidates: "
          f"{r.join(t.select('x', 's1'), on=['x', 's1'], how='anti').height})", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "holdout":
        holdout()
    elif sys.argv[1] == "pairs":
        pairs(sys.argv[2], sys.argv[3], *[float(v) for v in sys.argv[4:5]])
    else:
        from .acronym import cands
        cands(sys.argv[2], sys.argv[3], "fr_nonum_pairs.parquet")
