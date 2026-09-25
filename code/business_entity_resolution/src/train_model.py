"""Train the pair classifier, tune the decision threshold on macro F0.5, report errors.

    python code\\business_entity_resolution\\src\\train_model.py
    python code\\business_entity_resolution\\src\\train_model.py --skip-train   (reuse saved model)
"""
import argparse
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import feature_columns  # noqa: E402
from blocking import read_tsv  # noqa: E402
from eval import load_truth, entity_f05  # noqa: E402


def decode(P, proba, thr, one_to_one=True):
    """Return dict s1_id -> set(o_id). one_to_one: each o_id kept only for its best S1."""
    D = pd.DataFrame({"s1_id": P.s1_id.values, "o_id": P.o_id.values, "p": proba})
    if one_to_one:
        D = D.sort_values("p", ascending=False).drop_duplicates("o_id", keep="first")
    D = D[D.p >= thr]
    out = {}
    for s, o in zip(D.s1_id.values, D.o_id.values):
        out.setdefault(s, set()).add(o)
    return out


def score(pred, truth, s1_ids):
    return float(np.mean([entity_f05(pred.get(s, set()), truth[s]) for s in s1_ids]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="logs/pairs_train.parquet")
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--model-out", default="logs/lgbm.txt")
    ap.add_argument("--skip-train", action="store_true", help="load saved model instead of training")
    args = ap.parse_args()

    P = pd.read_parquet(args.pairs)
    S1 = pd.read_parquet(args.pairs.replace(".parquet", "_s1.parquet"))
    truth_all = load_truth(read_tsv(f"{args.data}/train/train_ground_truth.tsv"))
    truth = {s: truth_all[s] for s in S1.entity_id}
    print(f"pairs {len(P):,} | S1 {len(S1):,} | positives {int(P.label.sum()):,}")

    # ---- group split by S1, stratified by country ----
    rng = np.random.default_rng(args.seed)
    val_ids = set()
    for c, g in S1.groupby("country"):
        ids = g.entity_id.values.copy()
        rng.shuffle(ids)
        val_ids.update(ids[:int(len(ids) * args.val_frac)])
    is_val = P.s1_id.isin(val_ids).values
    cols = feature_columns(P)
    Xtr, ytr = P.loc[~is_val, cols], P.loc[~is_val, "label"].values
    Xva, yva = P.loc[is_val, cols], P.loc[is_val, "label"].values
    print(f"features {len(cols)} | train pairs {len(Xtr):,} | val pairs {len(Xva):,}")

    # ---- train (or load) ----
    if args.skip_train and os.path.exists(args.model_out):
        model = lgb.Booster(model_file=args.model_out)
        model.best_iteration = model.current_iteration()
        print(f"loaded model from {args.model_out}")
    else:
        params = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=50,
                      feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                      max_bin=255, verbose=-1, num_threads=max(1, (os.cpu_count() or 4) - 2),
                      seed=args.seed)
        dtr = lgb.Dataset(Xtr, ytr, free_raw_data=False)
        dva = lgb.Dataset(Xva, yva, reference=dtr)
        pbar = tqdm(total=args.rounds, desc="lightgbm", unit="rnd", mininterval=1.0)

        def _cb(env):
            pbar.update(1)
            if env.iteration % 50 == 0 and env.evaluation_result_list:
                pbar.set_postfix({f"{n}": f"{v:.4f}" for _, n, v, _ in env.evaluation_result_list})

        t0 = time.time()
        model = lgb.train(params, dtr, num_boost_round=args.rounds, valid_sets=[dva], valid_names=["val"],
                          callbacks=[lgb.early_stopping(60, verbose=False), _cb, lgb.log_evaluation(0)])
        pbar.close()
        print(f"trained in {time.time() - t0:.0f}s, best iteration {model.best_iteration}")
        os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
        model.save_model(args.model_out)

    pva = model.predict(Xva, num_iteration=model.best_iteration)
    print(f"pair-level  AUC {roc_auc_score(yva, pva):.5f} | AP {average_precision_score(yva, pva):.5f}")

    # ---- threshold tuning on macro F0.5 ----
    Pv = P.loc[is_val, ["s1_id", "o_id"]].reset_index(drop=True)
    val_s1 = sorted(val_ids)
    cty = S1.set_index("entity_id").country
    print("\nthreshold sweep (macro F0.5 on validation S1, singletons included):")
    print(f"{'thr':>6s} {'1to1':>8s} {'plain':>8s}")
    best = (0.0, 0.5)
    for thr in tqdm(np.arange(0.2, 0.951, 0.05), desc="sweep", leave=False):
        f1 = score(decode(Pv, pva, thr, True), truth, val_s1)
        f0 = score(decode(Pv, pva, thr, False), truth, val_s1)
        tqdm.write(f"{thr:6.2f} {f1:8.4f} {f0:8.4f}")
        if f1 > best[0]:
            best = (f1, thr)
    lo, hi = max(0.05, best[1] - 0.05), min(0.99, best[1] + 0.05)
    for thr in tqdm(np.arange(lo, hi + 1e-9, 0.01), desc="refine", leave=False):
        f1 = score(decode(Pv, pva, thr, True), truth, val_s1)
        if f1 > best[0]:
            best = (f1, thr)
    print(f"\nBEST macro F0.5 (1-to-1) = {best[0]:.4f} at threshold {best[1]:.2f}")

    # ---- per-country breakdown at best threshold ----
    pred = decode(Pv, pva, best[1], True)
    for c in sorted(cty.unique()):
        ids = [s for s in val_s1 if cty[s] == c]
        print(f"  {c}: {score(pred, truth, ids):.4f}  (n={len(ids):,})")
    n_single = sum(1 for s in val_s1 if not truth[s])
    sing_ok = sum(1 for s in val_s1 if not truth[s] and not pred.get(s))
    print(f"  singletons: {sing_ok}/{n_single} correctly empty")
    print(f"  mean predicted matches/S1 {np.mean([len(pred.get(s, ())) for s in val_s1]):.2f} "
          f"| mean true {np.mean([len(truth[s]) for s in val_s1]):.2f}")

    # ---- feature importance ----
    imp = pd.Series(model.feature_importance("gain"), index=cols).sort_values(ascending=False)
    print("\ntop 25 features by gain:")
    print((imp / imp.sum()).head(25).round(4).to_string())

    # ---- error analysis ----
    Pv["p"] = pva
    Pv["label"] = yva
    Pv["name_a"] = P.loc[is_val, "a_business_name"].values
    Pv["addr_a"] = P.loc[is_val, "a_business_address"].values
    Pv["name_b"] = P.loc[is_val, "b_business_name"].values
    Pv["addr_b"] = P.loc[is_val, "b_business_address"].values
    kept = Pv.sort_values("p", ascending=False).drop_duplicates("o_id")
    fp_mask = (kept.p >= best[1]) & (kept.label == 0)
    fn_mask = (Pv.label == 1) & (Pv.p < best[1])
    fp = kept[fp_mask].sample(min(12, int(fp_mask.sum())), random_state=1)
    fn = Pv[fn_mask].sample(min(12, int(fn_mask.sum())), random_state=1)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 55)
    print(f"\nFALSE POSITIVES (wrong merges): {int(fp_mask.sum()):,} total")
    print(fp[["p", "name_a", "addr_a", "name_b", "addr_b"]].round(3).to_string(index=False))
    print(f"\nFALSE NEGATIVES (missed matches): {int(fn_mask.sum()):,} total")
    print(fn[["p", "name_a", "addr_a", "name_b", "addr_b"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()