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


def build_unit_channels(t, ks, chunk, tag):
    """Fit all blocking channels once per (source, country) unit's target set `t`.
    Reused across every S1 slice of that unit — avoids refitting TF-IDF / the
    house-number index (each a full pass over millions of target rows) per slice."""
    comb = TfidfChannel("word", (1, 1), COMB_MAX_DF, f"{tag}/comb").fit(t.comb.values)
    ta = np.where(t.addr_n.str.len().values > 0)[0]
    addr = TfidfChannel("word", (1, 1), 0.02, f"{tag}/addr").fit(t.addr_n.values[ta])
    glue = TfidfChannel("char", (4, 4), 0.02, f"{tag}/glue").fit(t.glue.values)
    hn = _hn_city_index(t, label=tag)
    return {"comb": comb, "addr": (addr, ta), "glue": glue, "hn": hn}


def free_unit_channels(channels):
    channels["comb"].free()
    channels["addr"][0].free()
    channels["glue"].free()
    del channels["hn"]
    gc.collect()


def unit_candidates(q, t, channels, ks, chunk, tag):
    """Blocking for one S1 slice against a unit's prebuilt channels.
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

    add("comb", t_ids, channels["comb"].query(q.comb.values, ks["comb"], chunk))

    addr_ch, ta = channels["addr"]
    qa = np.where(q.addr_n.str.len().values > 0)[0]
    add("addr", t_ids[ta], addr_ch.query(q.addr_n.values[qa], ks["addr"], chunk), qmap=qa)

    add("glue", t_ids, channels["glue"].query(q.glue.values, ks["glue"], chunk))

    add("hn", t_ids, _hn_city_query(q, channels["hn"], ks["hn"], label=tag))
    return info


def process_unit(s1c, other_c, src, country, model, cols, slice_size, chunk, ks, n_proc=None):
    """Runs one unit and saves scored pairs (s1_id, o_id, p) to SCORED_DIR."""
    out_path = f"{SCORED_DIR}/{src}_{country}.parquet"
    if os.path.exists(out_path):
        print(f"[{src}|{country}] already done, skipping", flush=True)
        return
    os.makedirs(SLICE_TMP, exist_ok=True)
    n = len(s1c)
    slices = list(range(0, n, slice_size))
    tmp_files = [f"{SLICE_TMP}/{src}_{country}_{si}.parquet" for si in range(len(slices))]
    pending = [si for si, f in enumerate(tmp_files) if not os.path.exists(f)]

    channels = None
    if pending:
        print(f"\n[{src}|{country}] fitting blocking channels once on {len(other_c):,} targets "
              f"(reused across {len(slices)} slice(s)) ...", flush=True)
        t0 = time.time()
        channels = build_unit_channels(other_c, ks, chunk, f"{src}/{country}")
        print(f"[{src}|{country}] channels fit in {time.time() - t0:.0f}s", flush=True)

    for si, a in enumerate(slices):
        f = tmp_files[si]
        if si not in pending:
            print(f"[{src}|{country}] slice {si + 1}/{len(slices)} cached", flush=True)
            continue
        print(f"\n[{src}|{country}] slice {si + 1}/{len(slices)}  ({a:,}-{min(a + slice_size, n):,} of {n:,})",
              flush=True)
        q = s1c.iloc[a:a + slice_size].reset_index(drop=True)
        info = unit_candidates(q, other_c, channels, ks, chunk, f"{src}/{country}/s{si}")
        P = build_pairs(q, other_c, info)
        del info
        P = compute_features(P, n_proc=n_proc)
        P = P.drop(columns=[c for c in P.columns
                            if c.startswith(("a_business", "b_business", "a_name_core", "b_name_core",
                                             "a_legal", "b_legal", "a_addr_n", "b_addr_n",
                                             "a_country", "b_country"))])
        P.to_parquet(f, index=False)
        del P
        gc.collect()

    if channels is not None:
        free_unit_channels(channels)

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
    ap.add_argument("--src", default="S2,S3",
                    help="comma list of sources to process (e.g. 'S2' or 'S3') -- run two "
                         "terminals with --src S2 and --src S3 in parallel to use idle cores "
                         "(blocking's TF-IDF query is single-threaded, so S2 and S3 units don't "
                         "compete much); auto-decode only fires when both are requested")
    ap.add_argument("--nproc", type=int, default=None,
                    help="worker processes for feature engineering (default cpu_count()-2); "
                         "lower this when running --src S2 and --src S3 concurrently so the "
                         "two instances don't oversubscribe the CPU")
    args = ap.parse_args()
    srcs = tuple(s.strip() for s in args.src.split(",") if s.strip())

    t_all = time.time()
    s1 = read_tsv(f"{args.data}/test/test_source1.tsv")
    s1_ids = s1.entity_id.tolist()

    if not args.decode_only:
        model = lgb.Booster(model_file=args.model)
        model.best_iteration = model.current_iteration()
        cols = model.feature_name()
        s1 = add_norm(s1, "S1")
        others = {}
        for src in srcs:
            df = read_tsv(f"{args.data}/test/test_source{src[1]}.tsv")
            others[src] = add_norm(df, src)
        countries = sorted(s1.country.unique())
        print(f"countries: {countries} | sources: {srcs}", flush=True)
        ks = dict(DEFAULT_K)
        # smallest units first so problems surface early (France first)
        order = sorted([(src, c) for src in srcs for c in countries],
                       key=lambda u: (s1.country == u[1]).sum())
        for src, c in order:
            s1c = s1[s1.country == c].reset_index(drop=True)
            oc = others[src][others[src].country == c].reset_index(drop=True)
            print(f"\n######## unit {src} | {c}: S1={len(s1c):,} targets={len(oc):,} ########", flush=True)
            process_unit(s1c, oc, src, c, model, cols, args.slice, args.chunk, ks, n_proc=args.nproc)
            print(f"elapsed {(time.time() - t_all) / 3600:.2f} h", flush=True)

    if set(srcs) != {"S2", "S3"}:
        print(f"\n--src {args.src} requested (not both sources) -- skipping auto-decode. "
              f"Run `predict.py --decode-only --thr {args.thr}` once every --src run has "
              f"finished.", flush=True)
        return

    decode_and_write(s1_ids, args.thr)
    print("\nrunning validator ...", flush=True)
    subprocess.run([sys.executable, "utils/validate_submission.py",
                    "--matching", "output/matching_results.tsv",
                    "--candidate", "output/candidate_pairs.tsv",
                    "--test-dir", f"{args.data}/test", "--check-ids"])
    print(f"\nALL DONE in {(time.time() - t_all) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()