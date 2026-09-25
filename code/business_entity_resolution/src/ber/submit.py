"""Write output/matching_results.tsv and output/candidate_pairs.tsv (one row per test S1 entity)."""
import subprocess
import sys

import polars as pl

from . import config, io


def _write(pairs: pl.DataFrame, col: str, path):
    s1 = io.load("test", "s1", ["idx", "eid"]).rename({"idx": "s1", "eid": "source1_entity_id"})
    x = io.load("test", "x", ["idx", "eid"]).rename({"idx": "x", "eid": "xe"})
    lists = (pairs.select("s1", "x").unique().join(x, on="x")
             .sort("xe").group_by("s1").agg(pl.col("xe").str.join(",").alias(col)))
    out = s1.join(lists, on="s1", how="left").with_columns(pl.col(col).fill_null("")).select("source1_entity_id", col)
    path.parent.mkdir(parents=True, exist_ok=True)
    # write manually: no quoting, tab separated, empty lists as empty field
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for a, b in out.iter_rows():
            f.write(f"{a}\t{b}\n")
    n_nonempty = out.filter(pl.col(col) != "").height
    print(f"wrote {path}: {out.height} rows, {n_nonempty} non-empty, {pairs.height} pairs")


def write(matches: pl.DataFrame, cands: pl.DataFrame):
    _write(matches, "matched_entity_ids", config.OUT / "matching_results.tsv")
    _write(cands, "candidate_entity_ids", config.OUT / "candidate_pairs.tsv")


def validate():
    tool = config.DATA.parent / "utils" / "validate_submission.py"
    r = subprocess.run([sys.executable, str(tool), "--matching", str(config.OUT / "matching_results.tsv"),
                        "--candidate", str(config.OUT / "candidate_pairs.tsv"), "--test-dir", str(config.DATA / "test")],
                       capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-2000:])
    return r.returncode == 0
