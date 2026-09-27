"""Does the LLM judge add information beyond the model's own probability? (labelled holdout gate)
    python -m ber.experiments.llm_gate2 gate <gate.parquet> <scores.parquet>
        per slice: AUC of p (stage-3 p3 or stage-2 p2), AUC of p_yes, AUC of a 5-fold logistic combination;
        decision tables for the combination; saves the combination (fit on the whole gate) to <scores>.combo.json
    python -m ber.experiments.llm_gate2 apply <slices.parquet> <scores.parquet> <combo.json> <add_min> <drop_max> <tag>
        -> preds/<tag>_add_pairs.parquet, preds/<tag>_drop_pairs.parquet  (source1_entity_id, xe)
F0.5 per-pair economics (S1 with ~1-3 other matches): add a pair only if P(true) >= ~0.8; drop an accepted pair only if
P(true) <= ~0.6.
"""
import json
import sys

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .. import config, io


def _X(d):
    lp = np.log(np.clip(d["p"].to_numpy(), 1e-4, 1 - 1e-4) / (1 - np.clip(d["p"].to_numpy(), 1e-4, 1 - 1e-4)))
    ly = np.log(np.clip(d["p_yes"].to_numpy(), 1e-4, 1 - 1e-4) / (1 - np.clip(d["p_yes"].to_numpy(), 1e-4, 1 - 1e-4)))
    return np.c_[lp, ly, lp * ly]


def gate(gate_file, scores):
    g = pl.read_parquet(gate_file).join(pl.read_parquet(scores), on="id").drop_nans("p_yes")
    combo = {}
    for sl in ("add", "veto", "wordchg"):
        d = g.filter(pl.col("slice") == sl)
        if d.height < 50 or d["y"].n_unique() < 2:
            continue
        y, X = d["y"].to_numpy(), _X(d)
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
            oof[te] = LogisticRegression(C=1.0, max_iter=1000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        m = LogisticRegression(C=1.0, max_iter=1000).fit(X, y)
        combo[sl] = {"coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0])}
        print(f"\n== {sl}: n={len(y)} base P(true)={y.mean():.3f} | AUC p={roc_auc_score(y, d['p'].to_numpy()):.3f} "
              f"p_yes={roc_auc_score(y, d['p_yes'].to_numpy()):.3f} combo(oof)={roc_auc_score(y, oof):.3f}")
        for t in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
            hi, lo = y[oof >= t], y[oof < t]
            print(f"   combo>={t:<4}: n={len(hi):5d} P(true)={hi.mean() if len(hi) else 0:.3f} | "
                  f"below: n={len(lo):5d} P(true)={lo.mean() if len(lo) else 0:.3f}")
    json.dump(combo, open(str(scores) + ".combo.json", "w"), indent=1)
    print("saved", str(scores) + ".combo.json")


def apply(slices, scores, combo_file, add_min, drop_max, tag):
    t = pl.read_parquet(slices).join(pl.read_parquet(scores), on="id").drop_nans("p_yes")
    combo = json.load(open(combo_file))
    s1 = io.load("test", "s1", ["idx", "eid"]).rename({"idx": "s1", "eid": "source1_entity_id"})
    xid = io.load("test", "x", ["idx", "eid"]).rename({"idx": "x", "eid": "xe"})
    out = {}
    for sl, cut, keep in (("add", add_min, "ge"), ("veto", drop_max, "le")):
        d = t.filter(pl.col("slice") == sl)
        if sl not in combo or d.height == 0:
            out[sl] = d.head(0)
            continue
        c = combo[sl]
        z = _X(d) @ np.array(c["coef"]) + c["intercept"]
        d = d.with_columns(q=pl.Series(1 / (1 + np.exp(-z))))
        out[sl] = d.filter(pl.col("q") >= cut) if keep == "ge" else d.filter(pl.col("q") <= cut)
    for sl, name in (("add", "add"), ("veto", "drop")):
        o = out[sl].join(s1, on="s1").join(xid, on="x").select("source1_entity_id", "xe")
        o.write_parquet(config.path("preds", f"{tag}_{name}_pairs.parquet"))
        print(f"{tag}: {name} {o.height} pairs")


if __name__ == "__main__":
    if sys.argv[1] == "gate":
        gate(sys.argv[2], sys.argv[3])
    else:
        apply(sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5]), float(sys.argv[6]), sys.argv[7])
