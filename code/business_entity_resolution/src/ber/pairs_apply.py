"""Generic post-step on a submission TSV: add or drop a list of (source1_entity_id, xe) pairs for chosen countries.
    python -m ber.pairs_apply add  <in.tsv> <out.tsv> <pairs.parquet> [countries=US,India,France]
    python -m ber.pairs_apply drop <in.tsv> <out.tsv> <pairs.parquet> [countries=US,India,France]
add : a listed pair is added only if its X is unassigned in the file, respecting the source caps (<=5 S2, <=6 S3 per S1).
drop: a listed pair is removed if present. Rows of S1 outside the chosen countries stay byte-identical; the official
validator runs at the end. (ber.cn_rescue / ber.tie_trim are the France-only versions of the same two operations.)
"""
import subprocess
import sys

import polars as pl

from . import config, io

CAPS = {"S2": 5, "S3": 6}


def _load(inp):
    sub = pl.read_csv(inp, separator="\t", infer_schema=False).with_columns(pl.col("matched_entity_ids").fill_null(""))
    ex = (sub.filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
             .explode("matched_entity_ids").rename({"matched_entity_ids": "xe"}))
    return sub, ex


def _write(sub, new_pairs, changed, out):
    lists = (new_pairs.filter(pl.col("source1_entity_id").is_in(changed)).sort("xe").group_by("source1_entity_id")
                      .agg(pl.col("xe").str.join(",").alias("m")))
    res = sub.join(lists, on="source1_entity_id", how="left").with_columns(
        m=pl.when(pl.col("source1_entity_id").is_in(changed)).then(pl.col("m").fill_null(""))
            .otherwise(pl.col("matched_entity_ids")))
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for a, b in res.select("source1_entity_id", "m").iter_rows():
            f.write(f"{a}\t{b}\n")
    tool = config.DATA.parent / "utils" / "validate_submission.py"
    r = subprocess.run([sys.executable, str(tool), "--matching", out, "--test-dir", str(config.DATA / "test")],
                       capture_output=True, text=True)
    print("validator:", "PASS" if r.returncode == 0 else r.stdout[-1500:])


def run(op, inp, out, pairs_file, countries="US,India,France"):
    sub, ex = _load(inp)
    s1c = io.load("test", "s1", ["eid", "country"]).rename({"eid": "source1_entity_id"})
    lst = pl.read_parquet(config.path("preds", pairs_file)).select("source1_entity_id", "xe").unique()
    lst = lst.join(s1c, on="source1_entity_id").filter(pl.col("country").is_in(countries.split(","))).drop("country")
    if op == "drop":
        hit = ex.join(lst, on=["source1_entity_id", "xe"], how="semi")
        new = ex.join(lst, on=["source1_entity_id", "xe"], how="anti")
        changed = hit["source1_entity_id"].unique()
        print(f"pairs_apply drop: {hit.height} pairs removed on {changed.len()} S1")
    else:
        add = lst.join(ex.select("xe"), on="xe", how="anti").unique("xe", keep="first")   # X unassigned in this file
        add = add.with_columns(srcp=pl.col("xe").str.slice(0, 2))
        cnt = ex.with_columns(srcp=pl.col("xe").str.slice(0, 2)).group_by("source1_entity_id", "srcp").len("n")
        add = (add.join(cnt, on=["source1_entity_id", "srcp"], how="left").with_columns(pl.col("n").fill_null(0))
                  .sort("xe").with_columns(r=pl.col("xe").cum_count().over(["source1_entity_id", "srcp"]))
                  .filter(pl.col("n") + pl.col("r") <= pl.col("srcp").replace_strict(CAPS, default=0)))
        new = pl.concat([ex.select("source1_entity_id", "xe"), add.select("source1_entity_id", "xe")])
        changed = add["source1_entity_id"].unique()
        print(f"pairs_apply add: {add.height} pairs added on {changed.len()} S1")
    _write(sub, new, changed, out)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], *sys.argv[5:6])
