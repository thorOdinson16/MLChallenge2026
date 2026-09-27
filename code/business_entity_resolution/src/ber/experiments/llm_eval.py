"""Evaluate the LLM judge on the labelled holdout gate, then turn France scores into add/drop pair lists.
    python -m ber.experiments.llm_eval gate <gate_scores.parquet>
    python -m ber.experiments.llm_eval france <france_scores.parquet> <add_min> <drop_max>
F0.5 per-pair break-even (S1 with ~1-3 other matches): keep/add a pair only if P(true) >= ~0.75.
  add slice  : add France pairs with p_yes >= add_min only if the gate shows precision >= 0.85 at that cut.
  veto slice : drop France pairs with p_yes <= drop_max only if the gate shows P(true | vetoed) <= 0.6 at that cut.
"""
import sys

import polars as pl

from .. import config, io

KIT = config.WORK / "llm_kit"


def gate(scores):
    g = pl.read_parquet(KIT / "gate_holdout.parquet").join(pl.read_parquet(scores), on="id")
    print(f"gate rows {g.height}, NaN p_yes {g['p_yes'].is_nan().sum()}")
    for sl in ("add", "veto", "wordchg"):
        d = g.filter(pl.col("slice") == sl)
        if d.height == 0:
            continue
        print(f"\n== {sl}: n={d.height} base precision {d['y'].mean():.3f}  mean p_yes {d['p_yes'].mean():.3f}")
        for t in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98):
            yes, no = d.filter(pl.col("p_yes") >= t), d.filter(pl.col("p_yes") < t)
            print(f"   p_yes>={t:<4}: yes n={yes.height:5d} P(true)={yes['y'].mean() if yes.height else 0:.3f} | "
                  f"no n={no.height:5d} P(true)={no['y'].mean() if no.height else 0:.3f}")


def france(scores, add_min, drop_max):
    f = pl.read_parquet(KIT / "france_slices.parquet").join(pl.read_parquet(scores), on="id")
    s1 = io.load("test", "s1", ["idx", "eid"]).rename({"idx": "s1", "eid": "source1_entity_id"})
    xid = io.load("test", "x", ["idx", "eid"]).rename({"idx": "x", "eid": "xe"})
    add = f.filter((pl.col("slice") == "add") & (pl.col("p_yes") >= add_min)).join(s1, on="s1").join(xid, on="x")
    drop = f.filter((pl.col("slice") == "veto") & (pl.col("p_yes") <= drop_max)).join(s1, on="s1").join(xid, on="x")
    add.select("source1_entity_id", "xe").write_parquet(config.path("preds", "fr_llm_add_pairs.parquet"))
    drop.select("source1_entity_id", "xe").write_parquet(config.path("preds", "fr_llm_drop_pairs.parquet"))
    print(f"France: add {add.height} (p_yes >= {add_min}), drop {drop.height} (p_yes <= {drop_max})")
    w = f.filter(pl.col("slice") == "wordchg")
    print(f"France wordchg: mean p_yes {w['p_yes'].mean():.3f}, share p_yes<0.5 {(w['p_yes'] < 0.5).mean():.3f}"
          "  (compare with the gate's wordchg slice: a much higher France 'No' share = label-free decoy signal)")
    for lab, d in (("ADD", add), ("DROP", drop)):
        for r in d.sample(min(5, d.height), seed=0).iter_rows(named=True):
            print(f"  {lab} p_yes {r['p_yes']:.2f}  X: {r['xn']} | {r['xa']}  ->  S1: {r['sn']} | {r['sa']}")


if __name__ == "__main__":
    if sys.argv[1] == "gate":
        gate(sys.argv[2])
    else:
        france(sys.argv[2], float(sys.argv[3]), float(sys.argv[4]))
