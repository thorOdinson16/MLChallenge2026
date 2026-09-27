"""Acronym rescue: X records whose name is an acronym of the S1 name at the same address.

France has 6x more acronym-named X than the US (23,976 vs 3,912) and assigns only 72% of them (US 93%, India 97%).
The US/India final is stage 3, which carries the GBDT's initials feature; the France final is a blend of text
cross-encoders, which cannot see that "RC" = "Rassemblement Club". Two rules, both label-free on France and measured
on the labelled US/India holdout (out-of-fold stage-3 p3, see holdout()):
  A  cell : an existing candidate whose core-name initials equal the acronym and whose address agrees (unique per X)
  B  pool : a NEW candidate from the whole S1 pool on the key (country, initials, first house number) with an agreeing
            address (unique per X); retrieval never finds these because an acronym has no name similarity.
Address agreement (strict): same first house number (or X has none), and every X address word except street types
fuzzy-matches an S1 address word (Jaro-Winkler >= 0.85: 'cahnzy'~'chanzy' passes, 'tropiques'/'pluviers' fails).

    python -m ber.acronym holdout
    python -m ber.acronym pairs <france_probs.parquet> <pcol> [thr=0.85]   -> preds/fr_acronym_pairs.parquet
    python -m ber.cn_rescue <in.tsv> <out.tsv> fr_acronym_pairs.parquet     (adds for unassigned X, caps, France only)
    python -m ber.acronym cands <candidate_pairs.tsv> <out.tsv> [pairs.parquet]  (adds rescue pairs to the candidate set)
"""
import sys

import polars as pl
from rapidfuzz.distance import JaroWinkler

from . import config, io

P = config.path
ACR = r"^[A-ZÀ-ÖØ-Þ0-9&.]{2,6}$"          # incl. accented capitals (ÉPSM)
STREET = {"rue", "avenue", "boulevard", "allee", "place", "impasse", "chemin", "route", "square", "passage", "quai",
          "cours", "faubourg", "residence", "bis", "ter", "b", "street", "road", "lane", "drive", "sector", "near"}


def _addr_ok(xw, sw, xn, sn) -> bool:
    if xn and (not sn or xn[0] != sn[0]):
        return False
    words = [w for w in (xw or []) if w not in STREET and not w.isdigit()]
    if len(words) < 1 or not sw:
        return False
    return all(max(JaroWinkler.normalized_similarity(w, s) for s in sw) >= 0.85 for w in words)


def _x_table(split_name, xs):
    xr = pl.scan_parquet(P("raw", f"{split_name}_x.parquet")).select(pl.col("idx").alias("x"), "name").join(
        xs.lazy(), on="x", how="semi").collect()
    xr = xr.filter(pl.col("name").str.strip_chars().str.contains(ACR))
    xn = pl.scan_parquet(P("norm", f"{split_name}_x.parquet")).select(
        pl.col("idx").alias("x"), "country", pl.col("nm_core").alias("xc"), pl.col("ad_words").alias("xw"),
        pl.col("ad_nums").alias("xnum")).join(xr.lazy().select("x"), on="x", how="semi").collect()
    return xn.with_columns(ini=pl.col("xc").str.replace_all(" ", ""))


def _s1_table(split_name, countries):
    s = pl.scan_parquet(P("norm", f"{split_name}_s1.parquet")).select(
        pl.col("idx").alias("s1"), "country", pl.col("nm_core").alias("sc"), pl.col("ad_words").alias("sw"),
        pl.col("ad_nums").alias("snum")).filter(pl.col("country").is_in(countries)).collect()
    return s.with_columns(ini=pl.col("sc").str.split(" ").list.eval(pl.element().str.slice(0, 1)).list.join(""))


def _strict(d):
    ok = [_addr_ok(a, b, c, e) for a, b, c, e in zip(d["xw"].to_list(), d["sw"].to_list(), d["xnum"].to_list(),
                                                     d["snum"].to_list())]
    d = d.with_columns(ok=pl.Series(ok, dtype=pl.Boolean)).filter(pl.col("ok"))
    return d.with_columns(k=pl.len().over("x")).filter(pl.col("k") == 1).drop("k", "ok")


def rules(split_name, probs, pcol, thr, countries):
    """probs: (x, s1, pcol) candidate pairs. Returns rescue pairs (x, s1, rule) for X whose top prob < thr."""
    top = probs.group_by("x").agg(best=pl.col(pcol).max())
    xa = _x_table(split_name, probs.select("x").unique())
    xa = xa.filter(pl.col("country").is_in(countries))
    un = xa.join(top.filter(pl.col("best") < thr).select("x"), on="x", how="semi")
    s = _s1_table(split_name, countries)
    # A: existing candidates
    a = probs.join(un.select("x"), on="x", how="semi").select("x", "s1").join(un, on="x").join(
        s.select("s1", "sc", "sw", "snum", s_ini="ini"), on="s1")
    a = _strict(a.filter(pl.col("ini") == pl.col("s_ini"))).select("x", "s1").with_columns(rule=pl.lit("A"))
    # B: whole-pool search on (country, initials, first house number)
    ub = un.join(a.select("x"), on="x", how="anti").with_columns(n1=pl.col("xnum").list.first()).filter(
        pl.col("n1").is_not_null())
    sb = s.with_columns(n1=pl.col("snum").list.first()).filter(pl.col("n1").is_not_null())
    b = ub.join(sb.select("s1", "country", "ini", "n1", "sw", "snum"), on=["country", "ini", "n1"])
    b = _strict(b).select("x", "s1").join(probs.select("x", "s1"), on=["x", "s1"], how="anti").with_columns(
        rule=pl.lit("B"))
    return pl.concat([a, b])


def holdout():
    h = pl.read_parquet(P("preds", "s3oof_full.parquet"), columns=["x", "s1", "p3"])
    r = rules("train", h, "p3", 0.8, ["US", "India"])
    r = r.join(io.load("train", "gt").with_columns(y=pl.lit(1)), on=["s1", "x"], how="left").with_columns(
        pl.col("y").fill_null(0))
    print(r.group_by("rule").agg(n=pl.len(), precision=pl.col("y").mean()).sort("rule"))


def pairs(prob_file, pcol="pD", thr=0.85):
    cols = ["x", "s1", pcol] + (["ins_country_word"] if "ins_country_word" in pl.read_parquet_schema(prob_file) else [])
    t = pl.read_parquet(prob_file, columns=cols)
    if "ins_country_word" in t.columns:
        t = t.filter(pl.col("ins_country_word").fill_null(0) < 0.5)
    r = rules("test", t.select("x", "s1", pcol), pcol, thr, ["France"])
    s1 = io.load("test", "s1", ["idx", "eid"]).rename({"idx": "s1", "eid": "source1_entity_id"})
    xid = io.load("test", "x", ["idx", "eid"]).rename({"idx": "x", "eid": "xe"})
    out = r.join(s1, on="s1").join(xid, on="x").select("source1_entity_id", "xe", "rule")
    out.write_parquet(P("preds", "fr_acronym_pairs.parquet"))
    print(out.group_by("rule").len().sort("rule"), flush=True)


def cands(inp, out, pairs_file="fr_acronym_pairs.parquet"):
    """Add rescue pairs to candidate_pairs.tsv so every final match stays inside the candidate set.
    Default: the acronym rule-B pairs; any other pair file (e.g. fr_blist_pairs.parquet) adds all its pairs that are not
    candidates yet."""
    c = pl.read_csv(inp, separator="\t", infer_schema=False).with_columns(pl.col("candidate_entity_ids").fill_null(""))
    add = pl.read_parquet(P("preds", pairs_file))
    if pairs_file == "fr_acronym_pairs.parquet":
        add = add.filter(pl.col("rule") == "B")
    ex = (c.filter(pl.col("candidate_entity_ids") != "").with_columns(pl.col("candidate_entity_ids").str.split(","))
            .explode("candidate_entity_ids").rename({"candidate_entity_ids": "xe"}))
    new = add.select("source1_entity_id", "xe").join(ex, on=["source1_entity_id", "xe"], how="anti")
    changed = new["source1_entity_id"].unique()
    lists = (pl.concat([ex.filter(pl.col("source1_entity_id").is_in(changed)), new]).sort("xe")
               .group_by("source1_entity_id").agg(pl.col("xe").str.join(",").alias("m")))
    res = c.join(lists, on="source1_entity_id", how="left").with_columns(
        m=pl.when(pl.col("source1_entity_id").is_in(changed)).then(pl.col("m")).otherwise(pl.col("candidate_entity_ids")))
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for a, b in res.select("source1_entity_id", "m").iter_rows():
            f.write(f"{a}\t{b}\n")
    print(f"cands: +{new.height} rule-B pairs on {changed.len()} S1", flush=True)


if __name__ == "__main__":
    {"holdout": lambda: holdout(), "pairs": lambda: pairs(sys.argv[2], sys.argv[3], *[float(v) for v in sys.argv[4:5]]),
     "cands": lambda: cands(sys.argv[2], sys.argv[3], *sys.argv[4:5])}[sys.argv[1]]()
