"""Pre-submission guard for a final matching file (5 minutes).
    python -m ber.experiments.guard <matching_results.tsv> <candidate_pairs.tsv> [reference.tsv]
1. official validator with --candidate;
2. matches that are not in the candidate set (listed);
3. France pairs by edit class vs a reference file (default: the LB-tested L1): a jump in a decoy class (inserted business
   word, country word, the LB-negative nz_swap / del_only lists) is a red flag; acronym / CN+deletion growth is expected.
"""
import subprocess
import sys

import polars as pl
from rapidfuzz.distance import JaroWinkler

from .. import config, io

P = config.path
BUS = {"groupe", "holding", "participations", "distribution", "international", "developpement", "enterprises", "trading"}
NOISE = {"fils", "service", "associes"}


def _pairs(f, col):
    d = pl.read_csv(f, separator="\t", infer_schema=False).with_columns(pl.col(col).fill_null(""))
    return d.filter(pl.col(col) != "").with_columns(pl.col(col).str.split(",")).explode(col).rename({col: "xe"})


def main(match, cand, ref=None):
    ref = ref or str(config.OUT.parent / "submissions" / "L1_subK_cn" / "matching_results.tsv")
    tool = config.DATA.parent / "utils" / "validate_submission.py"
    r = subprocess.run([sys.executable, str(tool), "--matching", match, "--candidate", cand, "--test-dir",
                        str(config.DATA / "test")], capture_output=True, text=True, encoding="utf-8", errors="replace")
    print("validator:", "PASS" if r.returncode == 0 else "FAIL", "|", " ".join(l for l in r.stdout.splitlines() if "WARNING" in l)[:400])
    m, c = _pairs(match, "matched_entity_ids"), _pairs(cand, "candidate_entity_ids")
    miss = m.join(c, on=["source1_entity_id", "xe"], how="anti")
    print(f"matches not in candidates: {miss.height}", miss.head(10).rows() if miss.height else "")
    s1i = io.load("test", "s1", ["idx", "eid", "country"]).rename({"idx": "s1", "eid": "source1_entity_id"})
    xi = io.load("test", "x", ["idx", "eid"]).rename({"idx": "x", "eid": "xe"})
    xn = pl.read_parquet(P("norm", "test_x.parquet"), columns=["idx", "nm_core", "country"]).rename({"idx": "x"}).filter(
        pl.col("country") == "France").drop("country")
    sn = pl.read_parquet(P("norm", "test_s1.parquet"), columns=["idx", "nm_core", "country"]).rename({"idx": "s1"}).filter(
        pl.col("country") == "France").drop("country")
    df = sn.select(w=pl.col("nm_core").str.split(" ")).explode("w").group_by("w").len("df")
    real = set(df.filter(pl.col("df") >= 5)["w"].to_list())

    def cls(a, b):
        xt, st = set(a.split()), set(b.split())
        ins, dl = xt - st, st - xt
        if not ins and not dl:
            return "same core"
        if ins and ins <= BUS and not dl:
            return "INS business word (decoy)"
        if ins and ins <= NOISE:
            return "cn noise word" + (" + real-word drop" if [w for w in dl if w in real] else "")
        if "france" in ins:
            return "INS country word (decoy)"
        ri = [w for w in ins if w in real and max([JaroWinkler.normalized_similarity(w, v) for v in dl] or [0]) < 0.85]
        rd = [w for w in dl if w in real and max([JaroWinkler.normalized_similarity(w, v) for v in ins] or [0]) < 0.85]
        if ri and rd:
            return "real-word swap"
        if rd and not ins:
            return "real-word drop only"
        if ri and not dl:
            return "real-word insert"
        return "typo/other"

    def audit(f):
        p = (_pairs(f, "matched_entity_ids").join(s1i, on="source1_entity_id").filter(pl.col("country") == "France")
             .join(xi, on="xe").join(xn, on="x").join(sn, on="s1", suffix="_s"))
        return p.with_columns(c=pl.Series([cls(a, b) for a, b in zip(p["nm_core"].to_list(), p["nm_core_s"].to_list())])
                              ).select("source1_entity_id", "xe", "c")
    a, b = audit(match), audit(ref)
    t = (a.group_by("c").len("file").join(b.group_by("c").len("reference"), on="c", how="full", coalesce=True).fill_null(0)
          .with_columns(change=pl.col("file").cast(pl.Int64) - pl.col("reference").cast(pl.Int64)).sort("change"))
    with pl.Config(tbl_rows=20):
        print("France pairs by edit class (file vs reference):")
        print(t)
    for lst in ("nz_swap_pairs.parquet", "del_only_pairs.parquet"):
        q = pl.read_parquet(P("preds", lst))
        print(f"in LB-negative list {lst}: file {a.join(q, on=['source1_entity_id', 'xe'], how='semi').height}  "
              f"reference {b.join(q, on=['source1_entity_id', 'xe'], how='semi').height}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
