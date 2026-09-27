"""French noise-word rescue (LB-validated: sub_K 0.986429 -> 0.986966, France-only change).

Pairs where a France X sits at the same address and first house number as its top S1 and the only name change is a
French noise-suffix word (fils / service / associes; role found label-free: inserted word with no upward house-number
shift and a strong-address profile, the same profile US/India noise words have at 0.985-0.999 holdout precision),
no rival candidate. The pair list is precomputed (work/preds/cn_rescue_pairs.parquet, entity ids).

Applies to ANY submission TSV: adds a listed pair only if its X is unassigned in that file, respects the source caps
(<=5 S2, <=6 S3 per S1), leaves every non-France row byte-identical, then runs the official validator.

    python -m ber.cn_rescue <in.tsv> <out.tsv>
"""
import subprocess
import sys

import polars as pl

from . import config

CAPS = {"S2": 5, "S3": 6}


def apply(inp: str, out: str, pairs_file: str = "cn_rescue_pairs.parquet"):
    sub = pl.read_csv(inp, separator="\t", infer_schema=False).with_columns(pl.col("matched_entity_ids").fill_null(""))
    pairs = (sub.filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
             .explode("matched_entity_ids").rename({"matched_entity_ids": "xe"}))
    s1c = pl.read_csv(config.DATA / "test" / "test_source1.tsv", separator="\t", quote_char=None, infer_schema=False,
                      columns=["entity_id", "country"]).rename({"entity_id": "source1_entity_id"})
    add = pl.read_parquet(config.path("preds", pairs_file))                  # source1_entity_id, xe
    add = add.join(pairs.select("xe"), on="xe", how="anti")                     # X unassigned in this file
    add = add.join(s1c, on="source1_entity_id").filter(pl.col("country") == "France")
    add = add.with_columns(srcp=pl.col("xe").str.slice(0, 2))
    cnt = pairs.with_columns(srcp=pl.col("xe").str.slice(0, 2)).group_by("source1_entity_id", "srcp").len("n")
    add = (add.join(cnt, on=["source1_entity_id", "srcp"], how="left").with_columns(pl.col("n").fill_null(0)).sort("xe")
              .with_columns(r=pl.col("xe").cum_count().over(["source1_entity_id", "srcp"]))
              .filter(pl.col("n") + pl.col("r") <= pl.col("srcp").replace_strict(CAPS, default=0)))
    new = pl.concat([pairs.select("source1_entity_id", "xe"), add.select("source1_entity_id", "xe")])
    lists = new.sort("xe").group_by("source1_entity_id").agg(pl.col("xe").str.join(",").alias("m"))
    res = sub.join(lists, on="source1_entity_id", how="left").join(s1c, on="source1_entity_id", how="left")
    res = res.with_columns(m=pl.when(pl.col("country") == "France").then(pl.col("m").fill_null(""))
                           .otherwise(pl.col("matched_entity_ids")))
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for a, b in res.select("source1_entity_id", "m").iter_rows():
            f.write(f"{a}\t{b}\n")
    print(f"cn_rescue: added {add.height} France pairs on {add['source1_entity_id'].n_unique()} S1; non-France rows unchanged")
    tool = config.DATA.parent / "utils" / "validate_submission.py"
    r = subprocess.run([sys.executable, str(tool), "--matching", out, "--test-dir", str(config.DATA / "test")],
                       capture_output=True, text=True)
    print("validator:", "PASS" if r.returncode == 0 else r.stdout[-1500:])


if __name__ == "__main__":
    apply(sys.argv[1], sys.argv[2], *(sys.argv[3:4]))
