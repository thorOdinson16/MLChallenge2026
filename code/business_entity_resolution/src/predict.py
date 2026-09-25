"""Test-time pipeline: blocking -> features -> model -> decode -> submission files.

Resumable: every (source, country) unit is checkpointed to logs/test_scored/.
Within a unit, S1 rows are processed in slices; each slice's feature table is
cached to logs/test_tmp/ until the whole unit is done, then concatenated once
for context features (rank/gap/reverse-rank) and prediction.

    python code\\business_entity_resolution\\src\\predict.py --thr 0.68
    python code\\business_entity_resolution\\src\\predict.py --decode-only --thr 0.68
"""
import argparse
import gc
import glob
import os
import subprocess
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blocking import read_tsv, add_norm, TfidfChannel, _hn_city_index, _hn_city_query, \
    DEFAULT_K, COMB_MAX_DF  # noqa: E402
from features import build_pairs, compute_features, add_context_features  # noqa: E402

SCORED_DIR = "logs/test_scored"
SLICE_TMP = "logs/test_tmp"


def unit_candidates(q, t, ks, chunk, tag):
    """Blocking for one (source, country) unit / slice. q, t already filtered to the country.
    Returns info: dict s1_id -> {o_id: {channel: (rank, score)}}."""
    from collections import defaultdict
    info = defaultdict(lambda: defaultdict(dict))
    q_ids, t_ids = q.entity_id.values, t.entity_id.values

    def add(chan, tids, res, qmap=None):
        for pos, r in enumerate(res):
            arr, sc = (r if isinstance(r, tuple) else (r, None))
            if len(arr) == 0:
                continue
            s = q_ids[qmap[pos]] if qmap is not None else q_ids[pos]
            tid = tids[arr]
            for rk, tt in enumerate(tid):
                info[s][tt][chan] = (rk, float(sc[rk]) if sc is not None else 1.0)

    ch = TfidfChannel("word", (1, 1), COMB_MAX_DF, f"{tag}/comb").fit(t.comb.values)
    add("comb", t_ids, ch.query(q.comb.values, ks["comb"], chunk)); ch.free()

    qa = np.where(q.addr_n.str.len().values > 0)[0]
    ta = np.where(t.addr_n.str.len().values > 0)[0]
    ch = TfidfChannel("word", (1, 1), 0.02, f"{tag}/addr").fit(t.addr_n.values[ta])
    add("addr", t_ids[ta], ch.query(q.addr_n.values[qa], ks["addr"], chunk), qmap=qa); ch.free()

    ch = TfidfChannel("char", (4, 4), 0.02, f"{tag}/glue").fit(t.glue.values)
    add("glue", t_ids, ch.query(q.glue.values, ks["glue"], chunk)); ch.free()

    tindex = _hn_city_index(t, label=tag)
    add("hn", t_ids, _hn_city_query(q, tindex, ks["hn"], label=tag))
    del tindex
    gc.collect()
    return info


def process_unit(s1c, other_c, src, country, model, cols, slice_size, chunk, ks):
    """Runs one unit and saves scored pairs (s1_id, o_id, p) to SCORED_DIR."""
    out_path = f"{SCORED_DIR}/{src}_{country}.parquet"
    if os.path.exists(out_path):
        print(f"[{src}|{country}] already done, skipping", flush=True)
        return
    os.makedirs(SLICE_TMP, exist_ok=True)
    n = len(s1c)
    slices = list(range(0, n, slice_size))
    tmp_files = []
    for si, a in enumerate(slices):
        f = f"{SLICE_TMP}/{src}_{country}_{si}.parquet"
        tmp_files.append(f)
        if os.path.exists(f):
            print(f"[{src}|{country}] slice {si + 1}/{len(slices)} cached", flush=True)
            continue
        print(f"\n[{src}|{country}] slice {si + 1}/{len(slices)}  ({a:,}-{min(a + slice_size, n):,} of {n:,})",
              flush=True)
        q = s1c.iloc[a:a + slice_size].reset_index(drop=True)
        info = unit_candidates(q, other_c, ks, chunk, f"{src}/{country}/s{si}")
        P = build_pairs(q, other_c, info)
        del info
        P = compute_features(P)
        P = P.drop(columns=[c for c in P.columns
                            if c.startswith(("a_business", "b_business", "a_name_core", "b_name_core",
                                             "a_legal", "b_legal", "a_addr_n", "b_addr_n",
                                             "a_country", "b_country"))])
        P.to_parquet(f, index=False)
        del P
        gc.collect()

    print(f"\n[{src}|{country}] concatenating {len(tmp_files)} slices + context features ...", flush=True)
    P = pd.concat([pd.read_parquet(f) for f in tmp_files], ignore_index=True)
    P = add_context_features(P)
    print(f"[{src}|{country}] predicting {len(P):,} pairs ...", flush=True)
    missing = [c for c in cols if c not in P.columns]
    if missing:
        raise RuntimeError(f"model expects columns not present in pair table: {missing}")
    X = P[cols]
    preds = np.empty(len(P), dtype=np.float32)
    step = 2_000_000
    for a in tqdm(range(0, len(P), step), desc="predict", unit="blk"):
        preds[a:a + step] = model.predict(X.iloc[a:a + step], num_iteration=model.best_iteration)
    res = pd.DataFrame({"s1_id": P.s1_id.values, "o_id": P.o_id.values, "p": preds})
    os.makedirs(SCORED_DIR, exist_ok=True)
    res.to_parquet(out_path, index=False)
    for f in tmp_files:
        os.remove(f)
    print(f"[{src}|{country}] saved {len(res):,} scored pairs", flush=True)
    del P, X, res
    gc.collect()


def decode_and_write(s1_ids, thr, out_dir="output"):
    files = sorted(glob.glob(f"{SCORED_DIR}/*.parquet"))
    print(f"decoding {len(files)} scored files at threshold {thr}", flush=True)
    R = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"total scored pairs {len(R):,}", flush=True)

    # candidate_pairs.tsv: everything the model scored (before the final threshold + 1-to-1 cut)
    cand = R.groupby("s1_id", sort=False)["o_id"].agg(",".join)

    # matching_results.tsv: one-to-one (each S2/S3 record kept for its single best S1), then threshold
    R = R.sort_values("p", ascending=False).drop_duplicates("o_id", keep="first")
    R = R[R.p >= thr]
    match = R.groupby("s1_id", sort=False)["o_id"].agg(",".join)

    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame({"source1_entity_id": s1_ids})
    df["matched_entity_ids"] = df.source1_entity_id.map(match).fillna("")
    df.to_csv(f"{out_dir}/matching_results.tsv", sep="\t", index=False)

    df2 = pd.DataFrame({"source1_entity_id": s1_ids})
    df2["candidate_entity_ids"] = df2.source1_entity_id.map(cand).fillna("")
    df2.to_csv(f"{out_dir}/candidate_pairs.tsv", sep="\t", index=False)

    n_match = (df.matched_entity_ids != "").mean()
    avg = df.matched_entity_ids.map(lambda x: 0 if not x else x.count(",") + 1).mean()
    print(f"S1 with >=1 match: {100 * n_match:.1f}% | mean matches/S1 {avg:.2f} "
          f"(train truth: 94.4% and 3.46)", flush=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--model", default="logs/lgbm.txt")
    ap.add_argument("--thr", type=float, default=0.68)
    ap.add_argument("--slice", type=int, default=100000, help="S1 rows per slice within a unit")
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--decode-only", action="store_true")
    args = ap.parse_args()

    t_all = time.time()
    s1 = read_tsv(f"{args.data}/test/test_source1.tsv")
    s1_ids = s1.entity_id.tolist()

    if not args.decode_only:
        model = lgb.Booster(model_file=args.model)
        model.best_iteration = model.current_iteration()
        cols = model.feature_name()
        s1 = add_norm(s1, "S1")
        others = {}
        for src in ("S2", "S3"):
            df = read_tsv(f"{args.data}/test/test_source{src[1]}.tsv")
            others[src] = add_norm(df, src)
        countries = sorted(s1.country.unique())
        print(f"countries: {countries}", flush=True)
        ks = dict(DEFAULT_K)
        # smallest units first so problems surface early (France first)
        order = sorted([(src, c) for src in ("S2", "S3") for c in countries],
                       key=lambda u: (s1.country == u[1]).sum())
        for src, c in order:
            s1c = s1[s1.country == c].reset_index(drop=True)
            oc = others[src][others[src].country == c].reset_index(drop=True)
            print(f"\n######## unit {src} | {c}: S1={len(s1c):,} targets={len(oc):,} ########", flush=True)
            process_unit(s1c, oc, src, c, model, cols, args.slice, args.chunk, ks)
            print(f"elapsed {(time.time() - t_all) / 3600:.2f} h", flush=True)

    decode_and_write(s1_ids, args.thr)
    print("\nrunning validator ...", flush=True)
    subprocess.run([sys.executable, "utils/validate_submission.py",
                    "--matching", "output/matching_results.tsv",
                    "--candidate", "output/candidate_pairs.tsv",
                    "--test-dir", f"{args.data}/test", "--check-ids"])
    print(f"\nALL DONE in {(time.time() - t_all) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()