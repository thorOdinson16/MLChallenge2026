"""Blocking diagnostic: recall@K per channel, union recall, and miss reasons.

    python code\\business_entity_resolution\\src\\block_diag.py --n 5000
"""
import argparse
import gc
import os
import re
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import addr_tokens, house_numbers  # noqa: E402
from blocking import read_tsv, add_norm  # noqa: E402
from eval import load_truth  # noqa: E402

KMAX = 50
KS = [5, 10, 15, 25, 40, 50]


def ranked_topk(sim, k):
    """Return list of arrays: top-k column indices per row, sorted by descending score."""
    out = []
    indptr, indices, data = sim.indptr, sim.indices, sim.data
    for i in range(sim.shape[0]):
        s, e = indptr[i], indptr[i + 1]
        if e == s:
            out.append(np.empty(0, dtype=np.int64))
            continue
        d, idx = data[s:e], indices[s:e]
        if len(d) > k:
            part = np.argpartition(-d, k - 1)[:k]
            d, idx = d[part], idx[part]
        order = np.argsort(-d)
        out.append(idx[order])
    return out


def channel(q_texts, t_texts, analyzer, ngram, max_df, chunk, label):
    vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, sublinear_tf=True,
                          dtype=np.float32, min_df=2, max_df=max_df, lowercase=False,
                          token_pattern=r"(?u)\b\w+\b" if analyzer == "word" else None)
    T = vec.fit_transform(t_texts)
    Q = vec.transform(q_texts)
    TT = T.T.tocsr()
    del T
    res, peak = [], 0
    for a in range(0, Q.shape[0], chunk):
        sim = (Q[a:a + chunk] @ TT).tocsr()
        peak = max(peak, sim.nnz)
        res.extend(ranked_topk(sim, KMAX))
        del sim
    print(f"      {label}: nnz/row(t)={TT.nnz / max(TT.shape[1], 1):.1f} peak_nnz={peak:,}", flush=True)
    del TT, Q
    gc.collect()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--data", default="dataset")
    args = ap.parse_args()

    print("loading ...", flush=True)
    s1 = read_tsv(f"{args.data}/train/train_source1.tsv")
    s2 = read_tsv(f"{args.data}/train/train_source2.tsv")
    s3 = read_tsv(f"{args.data}/train/train_source3.tsv")
    truth_all = load_truth(read_tsv(f"{args.data}/train/train_ground_truth.tsv"))
    sub = s1.sample(min(args.n, len(s1)), random_state=args.seed).reset_index(drop=True)
    sub = add_norm(sub, "S1")
    s2 = add_norm(s2, "S2")
    s3 = add_norm(s3, "S3")
    for df in (sub, s2, s3):
        df["comb"] = df.name_core + " " + df.addr_n

    # (label, source column, analyzer, ngram, max_df)
    CHANNELS = [
        ("addr_w.002", "addr_n", "word", (1, 1), 0.002),
        ("addr_w.02", "addr_n", "word", (1, 1), 0.02),
        ("name_w.005", "name_core", "word", (1, 1), 0.005),
        ("name_w.05", "name_core", "word", (1, 1), 0.05),
        ("name_c4.003", "name_core", "char_wb", (4, 4), 0.003),
        ("name_c4.02", "name_core", "char_wb", (4, 4), 0.02),
        ("comb_w.02", "comb", "word", (1, 1), 0.02),
        ("comb_w.1", "comb", "word", (1, 1), 0.1),
    ]

    # union configs to evaluate: (name, list of (channel label, k))
    UNIONS = {
        "base(addr.002,name.005,c4.003 @15/15/10)": [("addr_w.002", 15), ("name_w.005", 15), ("name_c4.003", 10)],
        "loose@15": [("addr_w.02", 15), ("name_w.05", 15), ("name_c4.02", 15)],
        "loose@25": [("addr_w.02", 25), ("name_w.05", 25), ("name_c4.02", 25)],
        "loose+comb@25": [("addr_w.02", 25), ("name_w.05", 25), ("name_c4.02", 25), ("comb_w.02", 25)],
        "comb.1+addr.02+c4@25": [("comb_w.1", 25), ("addr_w.02", 25), ("name_c4.02", 25)],
        "loose+comb@40": [("addr_w.02", 40), ("name_w.05", 40), ("name_c4.02", 40), ("comb_w.02", 40), ("comb_w.1", 40)],
    }

    miss_records = []

    for sname, other in (("S2", s2), ("S3", s3)):
        print(f"\n################ {sname} ################", flush=True)
        truth = {r.entity_id: {i for i in truth_all[r.entity_id] if i.startswith(sname + "-")}
                 for r in sub.itertuples()}
        # per (channel) -> dict s1_id -> ranked list of target ids, all countries pooled
        ranked = {lab: {} for lab, *_ in CHANNELS}
        for c in sorted(sub.country.unique()):
            q = sub[sub.country == c].reset_index(drop=True)
            t = other[other.country == c].reset_index(drop=True)
            print(f"  [{sname}|{c}] q={len(q):,} t={len(t):,}", flush=True)
            for lab, col, an, ng, mdf in CHANNELS:
                t0 = time.time()
                if col == "addr_n":
                    qi = np.where(q.addr_n.str.len().values > 0)[0]
                    ti = np.where(t.addr_n.str.len().values > 0)[0]
                else:
                    qi = np.arange(len(q))
                    ti = np.arange(len(t))
                res = channel(q[col].values[qi], t[col].values[ti], an, ng, mdf, args.chunk, f"{lab}")
                q_ids, t_ids = q.entity_id.values, t.entity_id.values[ti]
                for a, arr in zip(qi, res):
                    ranked[lab][q_ids[a]] = list(t_ids[arr])
                print(f"    {lab} done {time.time() - t0:.0f}s", flush=True)

        s1_ids = sub.entity_id.tolist()
        cty = sub.set_index("entity_id").country

        def recall_of(cands, ids):
            hit = tot = 0
            for s in ids:
                t = truth[s]
                hit += len(t & cands.get(s, set()))
                tot += len(t)
            return hit / max(tot, 1)

        print(f"\n--- {sname}: single-channel recall@K (all / India / US) ---")
        idx = {c: [s for s in s1_ids if cty[s] == c] for c in cty.unique()}
        print(f"{'channel':14s} " + " ".join(f"K={k:<3d}          " for k in KS))
        for lab, *_ in CHANNELS:
            row = []
            for k in KS:
                cands = {s: set(v[:k]) for s, v in ranked[lab].items()}
                row.append(f"{recall_of(cands, s1_ids):.3f}/{recall_of(cands, idx.get('India', [])):.3f}/{recall_of(cands, idx.get('US', [])):.3f}")
            print(f"{lab:14s} " + " ".join(row))

        print(f"\n--- {sname}: union configs (recall all / India / US | avg cands/S1) ---")
        best_cfg, best_cands = None, None
        for name, cfg in UNIONS.items():
            cands = defaultdict(set)
            for lab, k in cfg:
                for s, v in ranked[lab].items():
                    cands[s].update(v[:k])
            avg = np.mean([len(cands.get(s, ())) for s in s1_ids])
            print(f"{name:44s} {recall_of(cands, s1_ids):.4f} / {recall_of(cands, idx.get('India', [])):.4f} / "
                  f"{recall_of(cands, idx.get('US', [])):.4f} | {avg:.1f}")
            best_cfg, best_cands = name, cands  # last config is the widest; used for miss analysis

        # ---------- miss analysis on the widest union ----------
        lut = {r.entity_id: r for r in other.itertuples(index=False)}
        s1lut = {r.entity_id: r for r in sub.itertuples(index=False)}
        reasons = defaultdict(int)
        n_miss = 0
        examples = defaultdict(list)
        for s in s1_ids:
            miss = truth[s] - best_cands.get(s, set())
            for m in miss:
                n_miss += 1
                a, b = s1lut[s], lut[m]
                ta, tb = set(addr_tokens(a.business_address)), set(addr_tokens(b.business_address))
                na, nb = house_numbers(a.business_address), house_numbers(b.business_address)
                nm_a, nm_b = set(a.name_core.split()), set(b.name_core.split())
                if b.business_address.strip() == "":
                    r = "empty_addr"
                elif not b.business_name.isascii() and not (nm_a & nm_b):
                    r = "foreign_script_name"
                elif ta & tb:
                    r = "shares_addr_token(rank/crowded)"
                elif na & nb:
                    r = "shares_number_only"
                elif nm_a & nm_b:
                    r = "shares_name_token_only"
                else:
                    r = "nothing_shared"
                key = (a.country, r)
                reasons[key] += 1
                if len(examples[key]) < 3:
                    examples[key].append((a.business_name, a.business_address, b.business_name, b.business_address))
        print(f"\n--- {sname}: miss reasons for widest union ({n_miss:,} missed pairs) ---")
        for (cty_, r), v in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"{cty_:6s} {r:34s} {v:6d}  ({100 * v / max(n_miss, 1):.1f}%)")
        print("\nexamples (S1 name | S1 addr || missed name | missed addr):")
        for key, exs in sorted(examples.items(), key=lambda x: -reasons[x[0]])[:6]:
            print(f"  [{key[0]} / {key[1]}]")
            for e in exs:
                print(f"     {e[0]} | {e[1]}  ||  {e[2]} | {e[3]}")

    print("\nDone.")


if __name__ == "__main__":
    main()