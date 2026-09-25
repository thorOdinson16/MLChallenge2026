"""Per-country decode-threshold tuning on the existing held-out validation split.

Reuses the cached training pair table + trained model (no retraining, no
blocking/feature recomputation) -- pure post-hoc analysis of the exact same
validation split train_model.py already used, split out by country to see
whether a single global threshold is leaving score on the table.

    python code\\business_entity_resolution\\src\\tune_threshold.py
"""
import argparse
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import feature_columns  # noqa: E402
from blocking import read_tsv  # noqa: E402
from eval import load_truth, entity_f05  # noqa: E402
from train_model import decode, score  # noqa: E402


def sweep(Pv, pva, truth, ids, coarse=np.arange(0.2, 0.951, 0.02)):
    best = (0.0, 0.5)
    for thr in coarse:
        f = score(decode(Pv, pva, thr, True), truth, ids)
        if f > best[0]:
            best = (f, thr)
    lo, hi = max(0.05, best[1] - 0.02), min(0.99, best[1] + 0.02)
    for thr in np.arange(lo, hi + 1e-9, 0.005):
        f = score(decode(Pv, pva, thr, True), truth, ids)
        if f > best[0]:
            best = (f, thr)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="logs/pairs_train.parquet")
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--model", default="logs/lgbm.txt")
    ap.add_argument("--global-thr", type=float, default=0.68,
                    help="the single threshold currently used in predict.py, for comparison")
    args = ap.parse_args()

    P = pd.read_parquet(args.pairs)
    S1 = pd.read_parquet(args.pairs.replace(".parquet", "_s1.parquet"))
    truth_all = load_truth(read_tsv(f"{args.data}/train/train_ground_truth.tsv"))
    truth = {s: truth_all[s] for s in S1.entity_id}

    # identical group split by S1 stratified by country, same seed as train_model.py
    rng = np.random.default_rng(args.seed)
    val_ids = set()
    for c, g in S1.groupby("country"):
        ids = g.entity_id.values.copy()
        rng.shuffle(ids)
        val_ids.update(ids[:int(len(ids) * args.val_frac)])
    is_val = P.s1_id.isin(val_ids).values
    cols = feature_columns(P)

    model = lgb.Booster(model_file=args.model)
    model.best_iteration = model.current_iteration()
    Xva = P.loc[is_val, cols]
    pva = model.predict(Xva, num_iteration=model.best_iteration)

    Pv = P.loc[is_val, ["s1_id", "o_id"]].reset_index(drop=True)
    val_s1 = sorted(val_ids)
    cty = S1.set_index("entity_id").country

    print(f"validation S1: {len(val_s1):,}\n")
    print(f"{'country':10s} {'n':>8s} {'global@{:.2f}'.format(args.global_thr):>14s} "
          f"{'best_thr':>10s} {'best_f05':>10s} {'gain':>8s}")

    per_country_thr = {}
    global_pred = decode(Pv, pva, args.global_thr, True)
    total_global, total_tuned, n_total = 0.0, 0.0, 0
    for c in sorted(cty.unique()):
        ids = [s for s in val_s1 if cty[s] == c]
        f_global = score(global_pred, truth, ids)
        f_best, thr_best = sweep(Pv, pva, truth, ids)
        per_country_thr[c] = thr_best
        print(f"{c:10s} {len(ids):8,d} {f_global:14.4f} {thr_best:10.2f} {f_best:10.4f} "
              f"{f_best - f_global:+8.4f}")
        total_global += f_global * len(ids)
        total_tuned += f_best * len(ids)
        n_total += len(ids)

    print(f"\n{'BLENDED':10s} {n_total:8,d} {total_global / n_total:14.4f} {'':>10s} "
          f"{total_tuned / n_total:10.4f} {(total_tuned - total_global) / n_total:+8.4f}")

    # sanity: apply each country's own best threshold simultaneously and re-score for real
    # (per-entity thresholds picked independently can overstate the combined gain slightly
    # if there's any cross-country leakage in decode's one-to-one step; there isn't here
    # since o_id never crosses countries, but we verify directly anyway)
    per_row_thr = Pv.s1_id.map(cty).map(per_country_thr).values
    D = pd.DataFrame({"s1_id": Pv.s1_id.values, "o_id": Pv.o_id.values, "p": pva, "thr": per_row_thr})
    D = D.sort_values("p", ascending=False).drop_duplicates("o_id", keep="first")
    D = D[D.p >= D.thr]
    combined_pred = {}
    for s, o in zip(D.s1_id.values, D.o_id.values):
        combined_pred.setdefault(s, set()).add(o)
    combined_f05 = score(combined_pred, truth, val_s1)
    print(f"\napplied-simultaneously macro F0.5 with per-country thresholds: {combined_f05:.4f} "
          f"(vs single global {args.global_thr}: {score(global_pred, truth, val_s1):.4f})")
    print(f"\nper-country thresholds to use in predict.py: {per_country_thr}")


if __name__ == "__main__":
    main()
