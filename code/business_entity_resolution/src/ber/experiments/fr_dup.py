"""France exact-duplicate consistency on a submission TSV (hard generator rule: in train, X records with identical
normalized name + address belong to the same S1 99.7% of the time and are never partially matched).

For every France duplicate group with at least one matched member, assign every unassigned member to the group's S1
(members already matched elsewhere are left alone), respecting source caps.  Non-France rows are untouched.

    python -m ber.experiments.fr_dup <in.tsv> <out.tsv>
"""
import sys

import polars as pl

from .. import config, io
from .overnight import dup_keys

CAPS = {"S2": 5, "S3": 6}


def apply(inp, out):
    s1 = io.load("test", "s1", ["idx", "eid", "country"]).rename({"idx": "s1", "eid": "source1_entity_id"})
    x = io.load("test", "x", ["idx", "eid"]).rename({"idx": "x", "eid": "xe"})
    sub = pl.read_csv(inp, separator="\t", infer_schema=False).with_columns(pl.col("matched_entity_ids").fill_null(""))
    pairs = (sub.filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
             .explode("matched_entity_ids").rename({"matched_entity_ids": "xe"}).join(x, on="xe"))
    K = dup_keys("test").join(pl.read_parquet(config.path("norm", "test_x.parquet"), columns=["idx", "country"]).rename({"idx": "x"}), on="x")
    K = K.filter(pl.col("country") == "France")
    m = K.join(pairs.select("x", "source1_entity_id"), on="x", how="left")
    grp = m.drop_nulls("source1_entity_id").group_by("key").agg(pl.col("source1_entity_id").unique().alias("t"))
    grp = grp.filter(pl.col("t").list.len() == 1).with_columns(tgt=pl.col("t").list.first())      # consistent groups only
    add = m.filter(pl.col("source1_entity_id").is_null()).join(grp.select("key", "tgt"), on="key").select(pl.col("tgt").alias("source1_entity_id"), "x")
    add = add.join(x, on="x").with_columns(srcp=pl.col("xe").str.slice(0, 2))
    cnt = pairs.with_columns(srcp=pl.col("xe").str.slice(0, 2)).group_by("source1_entity_id", "srcp").len("n")
    add = (add.join(cnt, on=["source1_entity_id", "srcp"], how="left").with_columns(pl.col("n").fill_null(0)).sort("xe")
              .with_columns(r=pl.col("xe").cum_count().over(["source1_entity_id", "srcp"]))
              .filter(pl.col("n") + pl.col("r") <= pl.col("srcp").replace_strict(CAPS, default=0)))
    new = pl.concat([pairs.select("source1_entity_id", "xe"), add.select("source1_entity_id", "xe")])
    lists = new.sort("xe").group_by("source1_entity_id").agg(pl.col("xe").str.join(",").alias("m"))
    res = sub.join(lists, on="source1_entity_id", how="left").join(s1.select("source1_entity_id", "country"), on="source1_entity_id", how="left")
    res = res.with_columns(m=pl.when(pl.col("country") == "France").then(pl.col("m").fill_null("")).otherwise(pl.col("matched_entity_ids")))
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for a, b in res.select("source1_entity_id", "m").iter_rows():
            f.write(f"{a}\t{b}\n")
    print(f"fr_dup: added {add.height} France pairs (duplicate members of groups already matched to one S1); non-France rows unchanged")


if __name__ == "__main__":
    apply(sys.argv[1], sys.argv[2])
