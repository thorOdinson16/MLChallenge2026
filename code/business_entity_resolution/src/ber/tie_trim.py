"""France namesake-tie trim: drop accepted France pairs whose X is a coin flip between near-equal S1 candidates.

An X matches at most one S1, so when the top candidate's probability share p1st / (p1st + p2nd) is below SHARE (0.6),
the pick is close to a coin flip. Typical case: an X with an empty address and two S1 entities with the same name in
different cities. US/India holdout (labelled, stage-2 p2, accepted p >= 0.85): precision 0.670 for share in [0.5, 0.6),
0.805 in [0.6, 0.7). The F0.5 break-even for keeping a pair is ~0.73-0.77 (for an S1 with 1-3 other matches), so pairs
below share 0.6 cost more than they give. France accepts twice as many of them as US/India (0.53% vs 0.27% of accepted
pairs): the France trust blend scores each pair independently, while the US/India stage 3 reads the competition.

    python -m ber.tie_trim pairs <probs.parquet> <pcol> [thr=0.85] [share=0.6]  -> preds/fr_tie_trim_pairs.parquet
    python -m ber.tie_trim apply <in.tsv> <out.tsv> [pairs.parquet]           (removes listed France pairs if present)
Non-France rows stay byte-identical; the official validator runs at the end.
"""
import subprocess
import sys

import polars as pl

from . import config, io

SHARE, THR = 0.6, 0.85


def pairs(prob_file: str, pcol: str = "pD", thr: float = THR, share: float = SHARE,
          out: str = "fr_tie_trim_pairs.parquet") -> pl.DataFrame:
    t = pl.read_parquet(prob_file, columns=["x", "s1", pcol] + (["ins_country_word"] if "ins_country_word" in
                        pl.read_parquet_schema(prob_file) else []))
    if "ins_country_word" in t.columns:
        t = t.filter(pl.col("ins_country_word").fill_null(0) < 0.5)
    t = t.sort(["x", pcol], descending=[False, True]).with_columns(r=pl.col(pcol).cum_count().over("x"))
    a = t.filter(pl.col("r") == 1).select("x", "s1", p1st=pl.col(pcol))
    b = t.filter(pl.col("r") == 2).select("x", p2nd=pl.col(pcol))
    d = a.join(b, on="x", how="left").with_columns(pl.col("p2nd").fill_null(0.0))
    d = d.filter((pl.col("p1st") >= thr) & (pl.col("p1st") / (pl.col("p1st") + pl.col("p2nd")) < share))
    s1 = io.load("test", "s1", ["idx", "eid", "country"]).rename({"idx": "s1", "eid": "source1_entity_id"})
    xid = io.load("test", "x", ["idx", "eid"]).rename({"idx": "x", "eid": "xe"})
    d = d.join(s1, on="s1").filter(pl.col("country") == "France").join(xid, on="x")
    res = d.select("source1_entity_id", "xe", "p1st", "p2nd")
    res.write_parquet(config.path("preds", out))
    print(f"tie_trim: {res.height} France pairs listed (share < {share}, top >= {thr}) -> preds/{out}")
    return res


def apply(inp: str, out: str, pairs_file: str = "fr_tie_trim_pairs.parquet"):
    sub = pl.read_csv(inp, separator="\t", infer_schema=False).with_columns(pl.col("matched_entity_ids").fill_null(""))
    s1c = io.load("test", "s1", ["eid", "country"]).rename({"eid": "source1_entity_id"})
    drop = pl.read_parquet(config.path("preds", pairs_file)).select("source1_entity_id", "xe")
    drop = drop.join(s1c, on="source1_entity_id").filter(pl.col("country") == "France").select("source1_entity_id", "xe")
    ex = (sub.filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
             .explode("matched_entity_ids").rename({"matched_entity_ids": "xe"}))
    hit = ex.join(drop, on=["source1_entity_id", "xe"], how="semi")
    keep = ex.join(drop, on=["source1_entity_id", "xe"], how="anti")
    changed = hit["source1_entity_id"].unique()
    lists = (keep.filter(pl.col("source1_entity_id").is_in(changed)).sort("xe").group_by("source1_entity_id")
                 .agg(pl.col("xe").str.join(",").alias("m")))
    res = sub.join(lists, on="source1_entity_id", how="left").with_columns(
        m=pl.when(pl.col("source1_entity_id").is_in(changed)).then(pl.col("m").fill_null(""))
            .otherwise(pl.col("matched_entity_ids")))
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for a, b in res.select("source1_entity_id", "m").iter_rows():
            f.write(f"{a}\t{b}\n")
    print(f"tie_trim: removed {hit.height} France pairs on {changed.len()} S1; other rows unchanged")
    tool = config.DATA.parent / "utils" / "validate_submission.py"
    r = subprocess.run([sys.executable, str(tool), "--matching", out, "--test-dir", str(config.DATA / "test")],
                       capture_output=True, text=True)
    print("validator:", "PASS" if r.returncode == 0 else r.stdout[-1500:])


if __name__ == "__main__":
    if sys.argv[1] == "pairs":
        pairs(sys.argv[2], sys.argv[3], *[float(v) for v in sys.argv[4:6]])
    else:
        apply(sys.argv[2], sys.argv[3], *sys.argv[4:5])
