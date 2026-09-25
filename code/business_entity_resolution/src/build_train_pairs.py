"""Build the labeled pair table for a train subsample and cache it as parquet.

    python code\\business_entity_resolution\\src\\build_train_pairs.py --n 100000
"""
import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blocking import read_tsv, add_norm, build_candidates  # noqa: E402
from features import build_pairs, compute_features, add_context_features  # noqa: E402
from eval import load_truth  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--out", default="logs/pairs_train.parquet")
    ap.add_argument("--chunk", type=int, default=500)
    args = ap.parse_args()

    t_all = time.time()
    print("loading train files ...", flush=True)
    s1 = read_tsv(f"{args.data}/train/train_source1.tsv")
    s2 = read_tsv(f"{args.data}/train/train_source2.tsv")
    s3 = read_tsv(f"{args.data}/train/train_source3.tsv")
    truth = load_truth(read_tsv(f"{args.data}/train/train_ground_truth.tsv"))

    sub = s1.sample(min(args.n, len(s1)), random_state=args.seed).reset_index(drop=True)
    print(f"S1 subsample: {len(sub):,}", flush=True)
    sub, s2, s3 = add_norm(sub, "S1"), add_norm(s2, "S2"), add_norm(s3, "S3")

    parts = []
    for name, other in (("S2", s2), ("S3", s3)):
        print(f"\n=== {name}: blocking ===", flush=True)
        _, _, info = build_candidates(sub, other, chunk=args.chunk, tag=name, return_ranked=True)
        print(f"=== {name}: pairs + features ===", flush=True)
        P = build_pairs(sub, other, info)
        del info
        P = compute_features(P)
        parts.append(P)

    P = pd.concat(parts, ignore_index=True)
    print(f"\ntotal pairs: {len(P):,}", flush=True)
    P = add_context_features(P)
    P["label"] = [float(o in truth[s]) for s, o in zip(P.s1_id.values, P.o_id.values)]
    print(f"positive rate {P.label.mean():.4f}  ({int(P.label.sum()):,} positives)", flush=True)

    # store the full S1 id list + truth size so evaluation can include entities with no candidates
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    P.to_parquet(args.out, index=False)
    sub[["entity_id", "country"]].to_parquet(args.out.replace(".parquet", "_s1.parquet"), index=False)
    print(f"saved {args.out} in {time.time() - t_all:.0f}s total", flush=True)


if __name__ == "__main__":
    main()