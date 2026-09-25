"""Pair feature construction.

build_pairs(s1_df, other_df, info)  -> DataFrame of candidate pairs (one row per pair)
compute_features(pairs)             -> adds all feature columns (multiprocessing)
add_context_features(pairs)         -> rank / gap / reverse-rank features

Input frames must have passed through blocking.add_norm (name_core, legal, addr_n, ...).
Only provided data is used.
"""
import os
import re
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import compact, acronym, house_numbers, addr_tokens  # noqa: E402

CHANNELS = ["comb", "addr", "glue", "hn"]
_DOMAIN_RE = re.compile(r"\.(com|net|org|in|co|io|fr|us)\b")

NAME_KEYS = ("n_tset", "n_tsort", "n_part", "n_ratio", "n_jw", "n_jac", "n_comp_ratio",
             "n_comp_part", "n_comp_jw", "n_lev", "n_prefix", "n_first", "n_len_ratio", "n_acr")
ADDR_KEYS = ("a_tset", "a_tsort", "a_part", "a_ratio", "a_jw", "a_jac", "a_shared", "a_cov_a", "a_cov_b",
             "hn_shared", "hn_jac", "hn_conflict", "lead_num_eq", "hn_digit_dist", "addr_exact")


def build_pairs(s1, other, info):
    """Expand blocking info (from build_candidates(return_ranked=True)) into a pair table.

    info: dict s1_id -> {other_id: {channel: (rank, score)}}
    """
    s1i = s1.set_index("entity_id")
    oi = other.set_index("entity_id")
    rows = []
    for s, d in tqdm(info.items(), total=len(info), desc="expand pairs", unit="s1", mininterval=1.0):
        for t, ch in d.items():
            r = {"s1_id": s, "o_id": t}
            for c in CHANNELS:
                if c in ch:
                    r[f"rk_{c}"], r[f"sc_{c}"] = ch[c]
                else:
                    r[f"rk_{c}"], r[f"sc_{c}"] = np.nan, np.nan
            rows.append(r)
    print(f"  building pair table ({len(rows):,} pairs) ...", flush=True)
    P = pd.DataFrame(rows)
    keep = ["business_name", "business_address", "name_core", "legal", "addr_n", "country"]
    A = s1i[keep].add_prefix("a_").reindex(P.s1_id.values).reset_index(drop=True)
    B = oi[keep].add_prefix("b_").reindex(P.o_id.values).reset_index(drop=True)
    P = pd.concat([P.reset_index(drop=True), A, B], axis=1)
    P["src"] = P.o_id.str[:2]
    return P


def _safe_ratio(f, a, b):
    return f(a, b) if a and b else np.nan

def _min_digit_dist(na, nb):
    """Min Levenshtein distance between any house number of a and any of b (digits only)."""
    best = np.nan
    for x in na:
        for y in nb:
            d = Levenshtein.distance(x, y)
            if best != best or d < best:
                best = d
    return best

def _pair_feats(args):
    (a_name_raw, a_addr_raw, a_core, a_legal, a_addr,
     b_name_raw, b_addr_raw, b_core, b_legal, b_addr) = args
    f = {}
    # non-latin script (indic etc.); accents alone are folded by normalization
    b_nonlatin = any(ord(ch) > 0x2FF and ch.isalpha() for ch in b_name_raw)
    f["name_nonlatin"] = float(b_nonlatin)
    ca, cb = compact(a_core), compact(b_core)

    if b_nonlatin and not cb:
        for k in NAME_KEYS:
            f[k] = np.nan
    else:
        f["n_tset"] = _safe_ratio(fuzz.token_set_ratio, a_core, b_core)
        f["n_tsort"] = _safe_ratio(fuzz.token_sort_ratio, a_core, b_core)
        f["n_part"] = _safe_ratio(fuzz.partial_ratio, a_core, b_core)
        f["n_ratio"] = _safe_ratio(fuzz.ratio, a_core, b_core)
        f["n_jw"] = JaroWinkler.similarity(a_core, b_core) if a_core and b_core else np.nan
        ta, tb = set(a_core.split()), set(b_core.split())
        f["n_jac"] = len(ta & tb) / max(len(ta | tb), 1) if ta and tb else np.nan
        f["n_comp_ratio"] = _safe_ratio(fuzz.ratio, ca, cb)
        f["n_comp_part"] = _safe_ratio(fuzz.partial_ratio, ca, cb)
        f["n_comp_jw"] = JaroWinkler.similarity(ca, cb) if ca and cb else np.nan
        f["n_lev"] = Levenshtein.distance(ca, cb) if ca and cb else np.nan
        f["n_prefix"] = float(bool(ca) and bool(cb) and ca[:4] == cb[:4])
        fa, fb = a_core.split()[:1], b_core.split()[:1]
        f["n_first"] = float(fa == fb and bool(fa))
        f["n_len_ratio"] = min(len(ca), len(cb)) / max(len(ca), len(cb), 1)
        acr_a, acr_b = acronym(a_core), acronym(b_core)
        f["n_acr"] = float(
            (len(a_core.split()) > 1 and len(cb) > 1 and acr_a == cb)
            or (len(b_core.split()) > 1 and len(ca) > 1 and acr_b == ca))
    f["name_domain"] = float(bool(_DOMAIN_RE.search(b_name_raw.lower())))
    f["legal_eq"] = float(a_legal == b_legal) if (a_legal and b_legal) else np.nan
    la_set, lb_set = set(a_legal.split()), set(b_legal.split())
    f["legal_jac"] = len(la_set & lb_set) / max(len(la_set | lb_set), 1) if (la_set or lb_set) else np.nan
    f["n_len_a"] = float(len(a_core))
    f["n_len_b"] = float(len(b_core))

    # ---- address ----
    f["addr_empty_b"] = float(not b_addr)
    if a_addr and b_addr:
        f["a_tset"] = fuzz.token_set_ratio(a_addr, b_addr)
        f["a_tsort"] = fuzz.token_sort_ratio(a_addr, b_addr)
        f["a_part"] = fuzz.partial_ratio(a_addr, b_addr)
        f["a_ratio"] = fuzz.ratio(a_addr, b_addr)
        f["a_jw"] = JaroWinkler.similarity(a_addr, b_addr)
        ata, atb = set(addr_tokens(a_addr_raw)), set(addr_tokens(b_addr_raw))
        f["a_jac"] = len(ata & atb) / max(len(ata | atb), 1) if (ata or atb) else np.nan
        f["a_shared"] = float(len(ata & atb))
        f["a_cov_a"] = len(ata & atb) / max(len(ata), 1)
        f["a_cov_b"] = len(ata & atb) / max(len(atb), 1)
        na, nb = house_numbers(a_addr_raw), house_numbers(b_addr_raw)
        f["hn_shared"] = float(len(na & nb))
        f["hn_jac"] = len(na & nb) / max(len(na | nb), 1) if (na or nb) else np.nan
        f["hn_conflict"] = float(bool(na) and bool(nb) and not (na & nb))
        f["hn_digit_dist"] = _min_digit_dist(na, nb) if (na and nb) else np.nan
        f["addr_exact"] = float(a_addr == b_addr)
        la = re.match(r"\s*(\d+)", a_addr_raw)
        lb = re.match(r"\s*(\d+)", b_addr_raw)
        f["lead_num_eq"] = (float(la.group(1).lstrip("0") == lb.group(1).lstrip("0"))
                            if (la and lb) else np.nan)
    else:
        for k in ADDR_KEYS:
            f[k] = np.nan
    return f


def compute_features(P, n_proc=None, chunksize=2000):
    """Adds per-pair feature columns to P (returns new DataFrame)."""
    n_proc = n_proc or max(1, (os.cpu_count() or 4) - 2)
    cols = ["a_business_name", "a_business_address", "a_name_core", "a_legal", "a_addr_n",
            "b_business_name", "b_business_address", "b_name_core", "b_legal", "b_addr_n"]
    it = zip(*[P[c].values for c in cols])
    t0 = time.time()
    if n_proc == 1:
        feats = [_pair_feats(x) for x in tqdm(it, total=len(P), desc="features", unit="pair", mininterval=1.0)]
    else:
        with Pool(n_proc) as pool:
            feats = list(tqdm(pool.imap(_pair_feats, it, chunksize=chunksize), total=len(P),
                              desc=f"features x{n_proc}", unit="pair", mininterval=1.0))
    print("  assembling feature frame ...", flush=True)
    F = pd.DataFrame(feats).astype(np.float32)
    dt = max(time.time() - t0, 1e-9)
    print(f"  features: {len(P):,} pairs in {dt:.0f}s ({len(P) / dt:,.0f} pairs/s)", flush=True)
    return pd.concat([P.reset_index(drop=True), F], axis=1)


def add_context_features(P):
    """Rank / gap / reverse-rank features. P must contain ALL candidate pairs of the batch.
    Memory-safe: uses groupby.rank/transform instead of full-frame sort_values, which avoids
    the large reorder-copy that a global sort forces on very large tables."""
    for c in CHANNELS:
        P[f"in_{c}"] = P[f"rk_{c}"].notna().astype(np.float32)
    P["n_channels"] = P[[f"in_{c}" for c in CHANNELS]].sum(axis=1).astype(np.float32)
    P["best_rank"] = P[[f"rk_{c}" for c in CHANNELS]].min(axis=1).astype(np.float32)
    P["comb_score"] = P["sc_comb"].fillna(0).astype(np.float32)

    n = P["n_tset"].fillna(P["n_comp_ratio"]).fillna(0).astype(np.float32) / np.float32(100.0)
    a = P["a_tset"].fillna(0).astype(np.float32) / np.float32(100.0)
    P["base"] = (np.float32(0.5) * n + np.float32(0.35) * a + np.float32(0.15) * P["comb_score"]).astype(np.float32)

    # per-S1 ranking (groupby, no full-frame sort/copy)
    g = P.groupby("s1_id", sort=False)["base"]
    P["rank_s1"] = g.rank(ascending=False, method="first").astype(np.float32)
    top = g.transform("max").astype(np.float32)
    P["gap_top_s1"] = (P["base"] - top).astype(np.float32)
    # second-best per group without a global sort: mask rank==2, take group max of that mask
    r2 = P["base"].where(P["rank_s1"] == 2)
    second = r2.groupby(P["s1_id"], sort=False).transform("max").fillna(0.0).astype(np.float32)
    P["gap_second_s1"] = (P["base"] - second).astype(np.float32)
    P["n_cands_s1"] = g.transform("size").astype(np.float32)
    del g, top, r2, second

    # per-other-record ranking (reverse rank)
    go = P.groupby("o_id", sort=False)["base"]
    P["rank_o"] = go.rank(ascending=False, method="first").astype(np.float32)
    top_o = go.transform("max").astype(np.float32)
    P["gap_top_o"] = (P["base"] - top_o).astype(np.float32)
    r2o = P["base"].where(P["rank_o"] == 2)
    second_o = r2o.groupby(P["o_id"], sort=False).transform("max").fillna(0.0).astype(np.float32)
    P["gap_second_o"] = (P["base"] - second_o).astype(np.float32)
    P["n_s1_for_o"] = go.transform("size").astype(np.float32)
    P["is_best_s1_for_o"] = (P["rank_o"] == 1).astype(np.float32)
    del go, top_o, r2o, second_o

    P["src_is_s2"] = (P["src"] == "S2").astype(np.float32)
    return P


def feature_columns(P):
    drop_prefix = ("a_business", "b_business", "a_name_core", "b_name_core", "a_legal", "b_legal",
                   "a_addr_n", "b_addr_n", "a_country", "b_country")
    drop_exact = {"s1_id", "o_id", "src", "label"}
    return [c for c in P.columns
            if c not in drop_exact and not c.startswith(drop_prefix)
            and pd.api.types.is_numeric_dtype(P[c])]


if __name__ == "__main__":
    # timing test on a small train subsample (blocking + features), to extrapolate cost
    import argparse
    from blocking import read_tsv, add_norm, build_candidates
    from eval import load_truth

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--proc", type=int, default=None)
    args = ap.parse_args()

    print("loading ...", flush=True)
    s1 = read_tsv(f"{args.data}/train/train_source1.tsv")
    s2 = read_tsv(f"{args.data}/train/train_source2.tsv")
    truth = load_truth(read_tsv(f"{args.data}/train/train_ground_truth.tsv"))
    sub = s1.sample(args.n, random_state=7).reset_index(drop=True)
    sub, s2 = add_norm(sub, "S1"), add_norm(s2, "S2")
    cands, per_ch, info = build_candidates(sub, s2, tag="S2", return_ranked=True)
    P = build_pairs(sub, s2, info)
    print(f"pairs: {len(P):,}", flush=True)
    P = compute_features(P, n_proc=args.proc)
    P = add_context_features(P)
    P["label"] = [float(o in truth[s]) for s, o in zip(P.s1_id, P.o_id)]
    cols = feature_columns(P)
    print(f"{len(cols)} feature columns; positive rate {P.label.mean():.4f}")
    print(P[cols].isna().mean().round(3).sort_values(ascending=False).head(12).to_string())