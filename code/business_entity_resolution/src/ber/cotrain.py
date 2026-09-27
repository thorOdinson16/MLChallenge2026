"""France co-training post-rule: let the full-coverage cross-encoder (ce_full2) overrule the France decision when it is
strongly confident, in the two directions the labelled holdout supports.

US/India holdout, top pair per X, stage-2 p2 vs the out-of-sample ce_full2 logit:
  accepted (p2 >= 0.85) and ce <= -4        : P(true) 0.315  (n 1,106)  -> drop   (keep needs ~0.75 under F0.5)
  grey (0.3 <= p2 < 0.85) and ce >= 4       : P(true) ~0.96  (n ~2,550) -> add
  confident reject (p2 < 0.3) and ce >= 4   : P(true) 0.41               -> leave alone
    python -m ber.cotrain pairs <france_probs.parquet> <pcol> [ce_tag=ce_full2] [add_ce=4] [drop_ce=-4]
        -> preds/fr_cot_add_pairs.parquet, preds/fr_cot_drop_pairs.parquet  (source1_entity_id, xe)
Apply: python -m ber.cn_rescue <in> <out> fr_cot_add_pairs.parquet   (adds only for unassigned X, caps, France only)
       python -m ber.tie_trim apply <in> <out> fr_cot_drop_pairs.parquet
"""
import sys

import polars as pl

from . import config, io

THR, GREY = 0.85, 0.3


def pairs(prob_file, pcol="pD", ce_tag="ce_full2", add_ce=4.0, drop_ce=-4.0):
    cols = ["x", "s1", pcol] + (["ins_country_word"] if "ins_country_word" in pl.read_parquet_schema(prob_file) else [])
    t = pl.read_parquet(prob_file, columns=cols)
    if "ins_country_word" in t.columns:
        t = t.filter(pl.col("ins_country_word").fill_null(0) < 0.5)
    top = t.sort(["x", pcol], descending=[False, True]).unique("x", keep="first")
    d = top.join(pl.read_parquet(config.path("preds", f"{ce_tag}_test.parquet")), on=["x", "s1"])
    s1 = io.load("test", "s1", ["idx", "eid", "country"]).rename({"idx": "s1", "eid": "source1_entity_id"})
    xid = io.load("test", "x", ["idx", "eid"]).rename({"idx": "x", "eid": "xe"})
    d = d.join(s1, on="s1").filter(pl.col("country") == "France").join(xid, on="x")
    add = d.filter((pl.col(pcol) >= GREY) & (pl.col(pcol) < THR) & (pl.col("ce") >= add_ce))
    drop = d.filter((pl.col(pcol) >= THR) & (pl.col("ce") <= drop_ce))
    add.select("source1_entity_id", "xe").write_parquet(config.path("preds", "fr_cot_add_pairs.parquet"))
    drop.select("source1_entity_id", "xe").write_parquet(config.path("preds", "fr_cot_drop_pairs.parquet"))
    print(f"cotrain: add {add.height} (grey {pcol} and {ce_tag} >= {add_ce}), drop {drop.height} "
          f"(accepted and {ce_tag} <= {drop_ce})")
    return add, drop


if __name__ == "__main__":
    pairs(sys.argv[2], sys.argv[3], *sys.argv[4:5], *[float(v) for v in sys.argv[5:7]])
