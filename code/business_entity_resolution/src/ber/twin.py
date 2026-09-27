"""Fuzzy-twin propagation (post-decode, after overnight.propagate).

An X left unassigned by the decoder is given the S1 of an ACCEPTED X ("its twin") when
  * both X have the same non-empty normalised address (norm.ad, exact string),
  * their core names are near-identical: rapidfuzz token_set_ratio(nm_core, nm_core_twin) >= SIM (90),
  * that S1 is one of the unassigned X's own candidates with probability >= PMIN (0.3),
  * the source caps (<=5 S2, <=6 S3 per S1) still hold afterwards (lowest-probability extras dropped).
Several twins -> highest similarity, then highest probability. overnight.propagate only covers EXACT duplicates
(country|nm|ad); this covers near-duplicates (typos, word order, a dropped legal word) at the same address.

Holdout, US/India labels (m_v2ce p2, thr 0.85 base): sim>=90, p>=0.3 adds pairs at 0.91 precision, +0.00005..0.00007 on
BOTH halves. FRANCE IS EXCLUDED by default: on France test 464 of 755 twins differ by a whole inserted/swapped/dropped
word (token_set_ratio = 100 when one name is a subset of the other), the sibling-decoy pattern that the France-only LB
A/B measured as negative (category swap + dropped word: -0.00088). Test rows use the scores AFTER the country-word filter (sc2 in overnight.build_test), so the rule can't
bring a country-word decoy back.

    from ber import twin
    sel = twin.rescue(sel, scores, "train" | "test", pcol="p3")        # sel: (s1, x); scores: x, s1, src, <pcol>
    python -m ber.twin <in.tsv> <out.tsv> <scores.parquet> [pcol=p3] [countries=US,India]   (test TSV applier)
"""
import subprocess
import sys

import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from . import config, decode, io

SIM, PMIN = 90, 0.3


def rescue(sel: pl.DataFrame, scores: pl.DataFrame, split_name: str, pcol: str = "p3", sim: float = SIM,
           pmin: float = PMIN, verbose: bool = True, exclude=("France",)) -> pl.DataFrame:
    xn = (pl.read_parquet(config.path("norm", f"{split_name}_x.parquet"), columns=["idx", "nm_core", "ad"])
          .rename({"idx": "x"}).filter(pl.col("ad").fill_null("") != "").with_columns(pl.col("nm_core").fill_null("")))
    sel = sel.select("s1", "x")
    acc = sel.join(xn, on="x").select(pl.col("x").alias("xa"), "s1", "ad", pl.col("nm_core").alias("nm_a"))
    un = xn.join(sel.select("x"), on="x", how="anti")
    j = un.join(acc, on="ad").join(scores.select("x", "s1", pl.col(pcol).alias("p")), on=["x", "s1"])
    j = j.filter(pl.col("p") >= pmin)
    if exclude:
        c = io.load(split_name, "s1", ["idx", "country"]).rename({"idx": "s1"})
        j = j.join(c.filter(~pl.col("country").is_in(list(exclude))).select("s1"), on="s1", how="semi")
    if j.height == 0:
        return sel
    j = j.with_columns(sim=pl.Series(cpdist(j["nm_core"].to_list(), j["nm_a"].to_list(), scorer=fuzz.token_set_ratio,
                                            workers=-1), dtype=pl.Float32)).filter(pl.col("sim") >= sim)
    add = j.sort(["x", "sim", "p"], descending=[False, True, True]).unique("x", keep="first").select("x", "s1", "p")
    src = scores.select("x", "src").unique("x")
    old = sel.join(scores.select("x", "s1", pl.col(pcol).alias("p")), on=["x", "s1"], how="left").with_columns(
        pl.col("p").fill_null(1.0), new=pl.lit(False))
    both = pl.concat([old, add.select("s1", "x", "p").with_columns(new=pl.lit(True))]).join(src, on="x", how="left")
    both = both.with_columns(pl.col("src").fill_null(0))
    # caps: existing matches keep priority over twin additions (p=1 floor), then probability
    both = both.with_columns(k=pl.when(pl.col("new")).then(pl.col("p")).otherwise(pl.col("p") + 2.0))
    out = decode.apply_caps(both, "k")
    if verbose:
        kept = out.filter(pl.col("new")).height
        print(f"twin: {add.height} candidate additions, {kept} kept after caps", flush=True)
    return out.select("s1", "x")


def apply_tsv(inp: str, out: str, scores_file: str, pcol: str = "p3", countries: str = ""):
    """Apply to a finished test submission TSV. Rows of S1 outside `countries` (if given) stay byte-identical."""
    s1 = io.load("test", "s1", ["idx", "eid", "country"]).rename({"idx": "s1", "eid": "source1_entity_id"})
    xid = io.load("test", "x", ["idx", "eid"]).rename({"idx": "x", "eid": "xe"})
    sub = pl.read_csv(inp, separator="\t", infer_schema=False).with_columns(pl.col("matched_entity_ids").fill_null(""))
    sel = (sub.filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
              .explode("matched_entity_ids").rename({"matched_entity_ids": "xe"})
              .join(s1, on="source1_entity_id").join(xid, on="xe").select("s1", "x"))
    sc = pl.read_parquet(scores_file)
    if "ins_country_word" in sc.columns:
        sc = sc.filter(pl.col("ins_country_word").fill_null(0) < 0.5)
    new = rescue(sel, sc, "test", pcol)
    if countries:
        keep = s1.filter(pl.col("country").is_in(countries.split(",")))["s1"]
        new = pl.concat([new.filter(pl.col("s1").is_in(keep)), sel.filter(~pl.col("s1").is_in(keep))])
    lists = (new.join(xid, on="x").sort("xe").group_by("s1").agg(pl.col("xe").str.join(",").alias("m"))
                .join(s1.select("s1", "source1_entity_id"), on="s1"))
    res = sub.join(lists.select("source1_entity_id", "m"), on="source1_entity_id", how="left")
    changed = res.filter(pl.col("m").fill_null("") != pl.col("matched_entity_ids"))["source1_entity_id"]
    res = res.with_columns(m=pl.when(pl.col("source1_entity_id").is_in(changed)).then(pl.col("m").fill_null(""))
                           .otherwise(pl.col("matched_entity_ids")))
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for a, b in res.select("source1_entity_id", "m").iter_rows():
            f.write(f"{a}\t{b}\n")
    print(f"twin: {new.height - sel.height:+d} pairs, {changed.len()} S1 rows changed", flush=True)
    tool = config.DATA.parent / "utils" / "validate_submission.py"
    r = subprocess.run([sys.executable, str(tool), "--matching", out, "--test-dir", str(config.DATA / "test")],
                       capture_output=True, text=True)
    print("validator:", "PASS" if r.returncode == 0 else r.stdout[-1500:])


if __name__ == "__main__":
    apply_tsv(*sys.argv[1:4], *sys.argv[4:6])
