"""Candidate generation (blocking), v4.

Per country, per source (S2 / S3), retrieve candidates for each S1 record as the
union of four channels:
  comb     : word TF-IDF over (name_core + addr_n)           K=20, max_df=0.03
  addr     : word TF-IDF over addr_n                         K=25, max_df=0.02
  glue     : char-4gram TF-IDF over the compact name         K=25, max_df=0.02
  hn_city  : exact key (house number + a locality token)     K=10
Vectorizers are fit once per (source, country); S1 rows stream through in chunks.
Only provided data is used.

CLI:
    python code\\business_entity_resolution\\src\\blocking.py --n 5000
"""
import argparse
import gc
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import norm_name, norm_addr, compact, house_numbers, addr_tokens  # noqa: E402
from eval import load_truth, blocking_recall  # noqa: E402

DEFAULT_K = dict(comb=20, addr=25, glue=25, hn=10)
COMB_MAX_DF = 0.03


def read_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def add_norm(df, desc="normalize"):
    """Adds columns: name_core, legal, addr_n, comb, glue."""
    names = df.business_name.values
    addrs = df.business_address.values
    nn = [norm_name(x) for x in tqdm(names, desc=f"{desc} names", unit="row", mininterval=1.0)]
    aa = [norm_addr(x) for x in tqdm(addrs, desc=f"{desc} addrs", unit="row", mininterval=1.0)]
    df = df.copy()
    df["name_core"] = [a for a, _ in nn]
    df["legal"] = [b for _, b in nn]
    df["addr_n"] = aa
    df["comb"] = df.name_core + " " + df.addr_n
    df["glue"] = [c.replace(" ", "") for c in df.name_core.values]
    return df


def _ranked_topk(sim, k):
    """sim CSR (rows x N) -> list of (idx array, score array) sorted by descending score."""
    out = []
    indptr, indices, data = sim.indptr, sim.indices, sim.data
    for i in range(sim.shape[0]):
        s, e = indptr[i], indptr[i + 1]
        if e == s:
            out.append((np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)))
            continue
        d, idx = data[s:e], indices[s:e]
        if len(d) > k:
            part = np.argpartition(-d, k - 1)[:k]
            d, idx = d[part], idx[part]
        order = np.argsort(-d)
        out.append((idx[order], d[order]))
    return out


class TfidfChannel:
    """Fit once on target texts, then query many times in chunks."""

    def __init__(self, analyzer, ngram, max_df, label):
        self.label = label
        self.vec = TfidfVectorizer(
            analyzer=analyzer, ngram_range=ngram, sublinear_tf=True, dtype=np.float32,
            min_df=2, max_df=max_df, lowercase=False,
            token_pattern=r"(?u)\b\w+\b" if analyzer == "word" else None)

    def fit(self, t_texts):
        t0 = time.time()
        print(f"      {self.label}: fitting on {len(t_texts):,} rows ...", flush=True)
        T = self.vec.fit_transform(t_texts)
        self.TT = T.T.tocsr()
        print(f"      {self.label}: fit {time.time() - t0:.0f}s vocab={T.shape[1]:,} "
              f"nnz/row={T.nnz / max(T.shape[0], 1):.1f}", flush=True)
        del T
        gc.collect()
        return self

    def query(self, q_texts, k, chunk):
        Q = self.vec.transform(q_texts)
        res = []
        peak = 0
        t0 = time.time()
        with tqdm(total=Q.shape[0], desc=f"      {self.label} query", unit="q",
                  mininterval=1.0, leave=False) as bar:
            for a in range(0, Q.shape[0], chunk):
                sim = (Q[a:a + chunk] @ self.TT).tocsr()
                peak = max(peak, sim.nnz)
                res.extend(_ranked_topk(sim, k))
                bar.update(sim.shape[0])
                del sim
        dt = time.time() - t0
        print(f"      {self.label}: peak nnz/chunk {peak:,} | {1000 * dt / max(Q.shape[0], 1):.1f} ms/query",
              flush=True)
        return res

    def free(self):
        del self.TT
        gc.collect()


def _hn_city_index(t, label=""):
    """Inverted index: (house_number, locality_token) -> list of target row indices."""
    idx = defaultdict(list)
    it = zip(t.business_address.values, t.addr_n.values)
    for i, (a_raw, a_n) in enumerate(tqdm(it, total=len(t), desc=f"      {label} hn index",
                                          unit="row", mininterval=1.0, leave=False)):
        if not a_n:
            continue
        nums = house_numbers(a_raw)
        toks = set(addr_tokens(a_raw))
        for n in nums:
            if len(n) < 1:
                continue
            for tk in toks:
                idx[(n, tk)].append(i)
    return idx


def _hn_city_query(q, tindex, k, label=""):
    """For each query row, collect target rows sharing (house_number, token); rank by #shared keys."""
    res = []
    it = zip(q.business_address.values, q.addr_n.values)
    for a_raw, a_n in tqdm(it, total=len(q), desc=f"      {label} hn query",
                           unit="q", mininterval=1.0, leave=False):
        if not a_n:
            res.append(np.empty(0, dtype=np.int64))
            continue
        cnt = defaultdict(int)
        nums = house_numbers(a_raw)
        toks = set(addr_tokens(a_raw))
        for n in nums:
            for tk in toks:
                lst = tindex.get((n, tk))
                if lst is not None and len(lst) <= 200:  # skip very generic keys
                    for j in lst:
                        cnt[j] += 1
        if cnt:
            best = sorted(cnt.items(), key=lambda x: -x[1])[:k]
            res.append(np.array([j for j, _ in best], dtype=np.int64))
        else:
            res.append(np.empty(0, dtype=np.int64))
    return res


def build_candidates(s1, other, ks=None, chunk=500, tag="", return_ranked=False):
    """Returns dict s1_entity_id -> set(other entity_ids), blocking within country.

    If return_ranked, also returns dict s1_id -> {other_id: {channel: (rank, score)}}
    for use as classifier features.
    """
    ks = {**DEFAULT_K, **(ks or {})}
    cands = defaultdict(set)
    info = defaultdict(lambda: defaultdict(dict)) if return_ranked else None
    per_channel = defaultdict(lambda: defaultdict(set))  # channel -> s1 -> set

    def add(chan, q_ids, t_ids, res, qmap=None):
        for pos, r in enumerate(res):
            arr, sc = (r if isinstance(r, tuple) else (r, None))
            if len(arr) == 0:
                continue
            s = q_ids[qmap[pos]] if qmap is not None else q_ids[pos]
            tid = t_ids[arr]
            cands[s].update(tid)
            per_channel[chan][s].update(tid)
            if info is not None:
                for rk, t in enumerate(tid):
                    info[s][t][chan] = (rk, float(sc[rk]) if sc is not None else 1.0)

    countries = sorted(s1.country.unique())
    overall = tqdm(total=len(countries) * 4, desc=f"  [{tag}] channels", unit="chan", position=0)
    for c in countries:
        q = s1[s1.country == c].reset_index(drop=True)
        t = other[other.country == c].reset_index(drop=True)
        if len(t) == 0:
            overall.update(4)
            continue
        print(f"\n  [{tag} | {c}] queries={len(q):,} targets={len(t):,}", flush=True)
        q_ids, t_ids = q.entity_id.values, t.entity_id.values
        t0 = time.time()

        # comb
        ch = TfidfChannel("word", (1, 1), COMB_MAX_DF, f"{tag}/{c}/comb").fit(t.comb.values)
        add("comb", q_ids, t_ids, ch.query(q.comb.values, ks["comb"], chunk))
        ch.free()
        overall.update(1)

        # addr (non-empty addresses only)
        qa = np.where(q.addr_n.str.len().values > 0)[0]
        ta = np.where(t.addr_n.str.len().values > 0)[0]
        ch = TfidfChannel("word", (1, 1), 0.02, f"{tag}/{c}/addr").fit(t.addr_n.values[ta])
        add("addr", q_ids, t_ids[ta], ch.query(q.addr_n.values[qa], ks["addr"], chunk), qmap=qa)
        ch.free()
        overall.update(1)

        # glue (char 4-grams on compact name)
        ch = TfidfChannel("char", (4, 4), 0.02, f"{tag}/{c}/glue").fit(t.glue.values)
        add("glue", q_ids, t_ids, ch.query(q.glue.values, ks["glue"], chunk))
        ch.free()
        overall.update(1)

        # house number + locality token exact-key channel
        tindex = _hn_city_index(t, label=f"{tag}/{c}")
        add("hn", q_ids, t_ids, _hn_city_query(q, tindex, ks["hn"], label=f"{tag}/{c}"))
        del tindex
        gc.collect()
        overall.update(1)
        print(f"  [{tag} | {c}] done {time.time() - t0:.0f}s", flush=True)
    overall.close()

    if return_ranked:
        return cands, per_channel, info
    return cands, per_channel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100000, help="S1 train subsample size")
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--min-recall", type=float, default=0.0,
                    help="exit with code 1 if pair recall (S2 or S3) is below this")
    args = ap.parse_args()

    t0 = time.time()
    print("loading train files ...", flush=True)
    s1 = read_tsv(f"{args.data}/train/train_source1.tsv")
    s2 = read_tsv(f"{args.data}/train/train_source2.tsv")
    s3 = read_tsv(f"{args.data}/train/train_source3.tsv")
    truth_all = load_truth(read_tsv(f"{args.data}/train/train_ground_truth.tsv"))
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)

    sub = s1.sample(min(args.n, len(s1)), random_state=args.seed).reset_index(drop=True)
    sub_ids = sub.entity_id.tolist()
    print(f"S1 subsample: {len(sub):,}", flush=True)
    sub, s2, s3 = add_norm(sub, "S1"), add_norm(s2, "S2"), add_norm(s3, "S3")
    cty = sub.set_index("entity_id").country

    worst = 1.0
    for name, other in (("S2", s2), ("S3", s3)):
        truth = {s: {i for i in truth_all[s] if i.startswith(name + "-")} for s in sub_ids}
        print(f"\n=== blocking vs {name} ===", flush=True)
        t0 = time.time()
        cands, per_ch = build_candidates(sub, other, chunk=args.chunk, tag=name)
        pr, er, avg = blocking_recall(cands, truth, sub_ids)
        worst = min(worst, pr)
        print(f"\n{name}: pair recall {pr:.4f} | S1 fully recalled {er:.4f} | "
              f"avg candidates/S1 {avg:.1f} | {time.time() - t0:.0f}s")
        for c in sorted(cty.unique()):
            ids = [s for s in sub_ids if cty[s] == c]
            pr_c, er_c, avg_c = blocking_recall(cands, truth, ids)
            print(f"   {c}: pair recall {pr_c:.4f} | full {er_c:.4f} | avg cands {avg_c:.1f}")
        print("   per-channel recall (alone) / avg cands:")
        for chan, d in per_ch.items():
            pr_ch, _, avg_ch = blocking_recall(d, truth, sub_ids)
            print(f"      {chan:6s} {pr_ch:.4f} / {avg_ch:.1f}")

    if worst < args.min_recall:
        print(f"\nGATE FAILED: worst pair recall {worst:.4f} < {args.min_recall}")
        sys.exit(1)
    print(f"\nGATE OK: worst pair recall {worst:.4f}")


if __name__ == "__main__":
    main()