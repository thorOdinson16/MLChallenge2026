"""Overnight candidate builder (runs fully offline, no supervision).

Stage 3 with a configurable set of cross-encoders, optionally trained on thinned-pool simulations
(work/preds/scratch_sim_{full,rm11,rm22}.parquet from the orphan simulation), decoded with threshold or the
expected-F0.5 decoder, plus exact-duplicate consistency.  Every choice is made on the labeled holdout
(5-fold grouped CV, evaluated on the full graph AND on a 19%-thinned graph that mimics the smaller test pool).

    python -m ber.experiments.overnight H          -> output/sub_H.tsv   (ce_v3)
    python -m ber.experiments.overnight I          -> output/sub_I.tsv   (ce_v3 + ce_xlmr_fr)
    python -m ber.experiments.overnight J tagA,tagB   custom CE set
Writes a markdown section to output/overnight_report.md.
"""
import hashlib
import os
import subprocess
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from .. import config, decode, evaluate, io, matcher2, split, submit
from ..stage3 import _logit

P = config.path
BASE_COLS = ["l1", "l2", "rank_p2_x", "max_p2_x", "sum_p2_x", "sum_p2_s_other", "n_conf_s_src_other", "gap_p2_x", "src_f", "ad_empty"]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=12, seed=1)
ROUNDS = 400
THRS = (0.7, 0.75, 0.8, 0.85)
REPORT = config.OUT / "overnight_report.md"
SEEDS = int(os.environ.get("BER_S3_SEEDS", 1))   # >1: average this many stage-3 fits (different seeds) for the test model


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(REPORT, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def ce_cols(tags):
    cols = []
    for t in tags:
        cols += [f"ce_{t}", f"cemax_{t}", f"cegap_{t}", f"l2ce_{t}"]
    return cols + (["l2ce_all"] if len(tags) > 1 else [])


def build(pred, split_name, tags, which):
    xn = pl.read_parquet(P("norm", f"{split_name}_x.parquet"), columns=["idx", "ad"]).rename({"idx": "x"})
    xn = xn.with_columns(ad_empty=(pl.col("ad").str.strip_chars() == "").cast(pl.Float32)).select("x", "ad_empty")
    d = pred.join(xn, on="x", how="left").with_columns(l1=_logit("p1"), l2=_logit("p2"))
    d = d.with_columns(
        rank_p2_x=pl.col("p2").rank("ordinal", descending=True).over("x").cast(pl.Float32),
        max_p2_x=pl.col("p2").max().over("x"), sum_p2_x=pl.col("p2").sum().over("x"),
        sum_p2_s_other=pl.col("p2").sum().over("s1") - pl.col("p2"),
        n_conf_s_src=(pl.col("p2") >= 0.8).sum().over(["s1", "src"]).cast(pl.Float32),
        conf_self=(pl.col("p2") >= 0.8).cast(pl.Float32), src_f=pl.col("src").cast(pl.Float32))
    d = d.with_columns(gap_p2_x=pl.col("p2") - pl.when(pl.col("p2") >= pl.col("max_p2_x")).then(0.0).otherwise(pl.col("max_p2_x")),
                       n_conf_s_src_other=pl.col("n_conf_s_src") - pl.col("conf_self"))
    for t in tags:
        ce = pl.read_parquet(P("preds", f"{t}_{which}.parquet")).rename({"ce": f"ce_{t}"})
        d = d.join(ce, on=["x", "s1"], how="left").with_columns(pl.col(f"ce_{t}").max().over("x").alias(f"cemax_{t}"))
        d = d.with_columns((pl.col(f"ce_{t}") - pl.col(f"cemax_{t}")).alias(f"cegap_{t}"),
                           (pl.col("l2") + pl.col(f"ce_{t}").fill_null(0.0)).alias(f"l2ce_{t}"))
    if len(tags) > 1:
        d = d.with_columns(l2ce_all=pl.col("l2") + sum(pl.col(f"ce_{t}").fill_null(0.0) for t in tags) / len(tags))
    return d


def raw_cols():
    return [c for c in matcher2.feat_cols("train_all_v2") if c != "src"]


# ------------------------------------------------------------------ holdout
def holdout_frames(tags):
    gt = io.load("train", "gt").with_columns(y=pl.lit(1, pl.UInt8))
    rc = raw_cols()
    hold = pl.DataFrame({"s1": split.holdout_s1()})
    G, raw = {}, None
    for g in ("full", "rm11", "rm22"):
        d = build(pl.read_parquet(P("preds", f"scratch_sim_{g}.parquet")), "train", tags, "holdout")
        if g == "rm22":                       # training-only graph: holdout-S1 rows suffice (features already computed on the graph)
            d = d.join(hold, on="s1", how="semi")
        keep = ["x", "s1", "src"] + BASE_COLS + ce_cols(tags)
        d = d.select(keep).with_columns([pl.col(c).cast(pl.Float32) for c in BASE_COLS + ce_cols(tags)])
        if raw is None:
            raw = pl.concat([pl.read_parquet(p, columns=["x", "s1"] + rc).join(d.select("x", "s1"), on=["x", "s1"], how="semi")
                             for p in matcher2.parts("train_all_v2")])
            raw = raw.with_columns([pl.col(c).cast(pl.Float32) for c in rc])
        d = d.join(raw, on=["x", "s1"], how="left").join(gt, on=["s1", "x"], how="left").with_columns(pl.col("y").fill_null(0))
        G[g] = d.with_columns(fold=((pl.col("s1").cast(pl.UInt64) * 40503 + 12345) % 1000003 % 5).cast(pl.Int32))
    return G


def cv_predict(G, cols, train_graphs, eval_graph, hold):
    te = G[eval_graph]
    Xt = te.select(cols).to_numpy().astype(np.float32); ft = te["fold"].to_numpy(); out = np.zeros(te.height)
    trs = [G[g].join(hold, on="s1", how="semi") for g in train_graphs]
    Xa = np.concatenate([t.select(cols).to_numpy().astype(np.float32) for t in trs])
    ya = np.concatenate([t["y"].to_numpy() for t in trs]); fa = np.concatenate([t["fold"].to_numpy() for t in trs])
    for k in range(5):
        m = lgb.train(PARAMS, lgb.Dataset(Xa[fa != k], ya[fa != k], feature_name=cols), ROUNDS)
        out[ft == k] = m.predict(Xt[ft == k])
    return te.select("x", "s1", "src").with_columns(p3=pl.Series(out.astype(np.float32)))


def select(pr, rule):
    base = decode.apply_caps(decode.assign_unique(pr.select("x", "s1", "src", pl.col("p3").alias("p")), "p").filter(pl.col("p") >= 0.02), "p")
    if rule == "expF":
        return decode.decode(base.select("x", "s1", "p"), "p", min_p=0.02, miss=0.0)
    return base.filter(pl.col("p") >= float(rule)).select("s1", "x")


def dup_keys(split_name):
    # full normalised name (NOT nm_core): nm_core drops stop words such as the country word, which would put a
    # "<name> France" decoy in the same group as its clean twin and bypass the country-word rule
    d = pl.read_parquet(P("norm", f"{split_name}_x.parquet"), columns=["idx", "country", "nm", "ad", "src"]).filter(pl.col("ad").fill_null("") != "")
    d = d.with_columns(key=pl.concat_str([pl.col("country"), pl.col("nm").fill_null(""), pl.col("ad")], separator="|"))
    g = d.group_by("key").agg(n=pl.len()).filter(pl.col("n") > 1)
    return d.join(g, on="key").select(pl.col("idx").alias("x"), "key", "src")


def propagate(sel, scores, K):
    """Exact-duplicate X groups share one S1 in train GT (99.7%): give every member the group's best accepted S1."""
    s = sel.select("s1", "x").join(scores.select("x", "s1", "p3", "src"), on=["x", "s1"], how="left")
    m = K.join(s.select("x", "s1", "p3"), on="x", how="left")
    best = m.drop_nulls("s1").sort("p3", descending=True).group_by("key").first().select("key", s_best="s1", p_best="p3")
    new = K.join(best, on="key").select("x", pl.col("s_best").alias("s1"), pl.col("p_best").alias("p3"), pl.col("src").cast(pl.UInt8))
    keep = s.join(new.select("x"), on="x", how="anti").select("x", "s1", "p3", pl.col("src").cast(pl.UInt8))
    return decode.apply_caps(pl.concat([keep, new]), "p3").select("s1", "x")


def choose(tags):
    log(f"## holdout evaluation, CE set {tags}")
    gt = io.load("train", "gt"); hold = pl.DataFrame({"s1": split.holdout_s1()})
    rm = io.load("train", "s1", ["idx"]).rename({"idx": "s1"}).sample(fraction=0.19, seed=11)
    kept = {"full": hold, "rm11": hold.join(rm, on="s1", how="anti")}
    G = holdout_frames(tags)
    cols = BASE_COLS + ce_cols(tags) + raw_cols()
    Kt = dup_keys("train")
    res = {}
    for tg in (("full",), ("full", "rm22")):
        for ev in ("full", "rm11"):
            pr = cv_predict(G, cols, tg, ev, hold)
            for rule in list(THRS) + ["expF"]:
                sel = select(pr, rule)
                res[(tg, ev, rule, False)] = evaluate.macro_f05(sel, gt, kept[ev]["s1"])
                if rule in (0.8, "expF"):
                    res[(tg, ev, rule, True)] = evaluate.macro_f05(propagate(sel, pr, Kt), gt, kept[ev]["s1"])
            log(f"   train on {'+'.join(tg):9s} eval {ev:5s}: " + "  ".join(f"{r}={res[(tg, ev, r, False)]:.5f}" for r in list(THRS) + ["expF"])
                + f"  | +dup: 0.8={res[(tg, ev, 0.8, True)]:.5f} expF={res[(tg, ev, 'expF', True)]:.5f}")
    best = max(((tg, rule, dup) for tg in (("full",), ("full", "rm22")) for rule in list(THRS) + ["expF"] for dup in (False, True)
                if (tg, "full", rule, dup) in res),
               key=lambda k: 0.5 * res[(k[0], "full", k[1], k[2])] + 0.5 * res[(k[0], "rm11", k[1], k[2])])
    log(f"   CHOSEN: train on {'+'.join(best[0])}, decoder {best[1]}, dup-propagation {best[2]}  -> full {res[(best[0], 'full', best[1], best[2])]:.5f}, "
        f"thinned {res[(best[0], 'rm11', best[1], best[2])]:.5f}   (shipped sub_F setting = ce_v2, full, 0.75: 0.99096 full)")
    return G, cols, best, res


def build_test(name, tags):
    t0 = time.time()
    G, cols, (tg, rule, dup), res = choose(tags)
    hold = pl.DataFrame({"s1": split.holdout_s1()})
    trs = [G[g].join(hold, on="s1", how="semi") for g in tg]
    X = np.concatenate([t.select(cols).to_numpy().astype(np.float32) for t in trs]); y = np.concatenate([t["y"].to_numpy() for t in trs])
    del G, trs
    ms = [lgb.train({**PARAMS, "seed": 1 + i}, lgb.Dataset(X, y, feature_name=cols), ROUNDS) for i in range(SEEDS)]
    ms[0].save_model(str(P("models", f"s3_{name}.txt")))
    for i, mi in enumerate(ms[1:], 2):
        mi.save_model(str(P("models", f"s3_{name}_seed{i}.txt")))
    del X
    te = pl.read_parquet(P("preds", "test_m_v2.parquet"))
    dt = build(te, "test", tags, "test").select(["x", "s1"] + BASE_COLS + ce_cols(tags) + ["p2", "country", "src"])
    outs = []
    for p in matcher2.parts("test_v2"):
        f = pl.read_parquet(p, columns=["x", "s1"] + raw_cols()).join(dt, on=["x", "s1"], how="inner")
        Xf = f.select(cols).to_numpy().astype(np.float32)
        p3 = np.mean([mi.predict(Xf, num_threads=12) for mi in ms], axis=0).astype(np.float32)
        outs.append(f.select("x", "s1", "src", "country", "p2", "ins_country_word").with_columns(p3=pl.Series(p3)))
    sc = pl.concat(outs)
    sc.write_parquet(P("preds", f"test_s3_{name}.parquet"))
    sc2 = sc.filter(pl.col("ins_country_word").fill_null(0) < 0.5)            # country-word decoy rule (as shipped)
    sel = select(sc2, rule)
    if dup:
        sel = propagate(sel, sc2, dup_keys("test"))
    out = config.OUT / f"sub_{name}.tsv"
    submit._write(sel.select("s1", "x"), "matched_entity_ids", out)
    tool = config.DATA.parent / "utils" / "validate_submission.py"
    r = subprocess.run([sys.executable, str(tool), "--matching", str(out), "--candidate", str(config.OUT / "candidate_pairs.tsv"),
                        "--test-dir", str(config.DATA / "test")], capture_output=True, text=True, encoding="utf-8", errors="replace")
    verdict = "PASS" if r.returncode == 0 else "FAIL"
    log(f"## sub_{name}: wrote {out.name} md5 {hashlib.md5(out.read_bytes()).hexdigest()}  validator {verdict}")
    if verdict != "PASS":
        log("   validator output: " + r.stdout[-1500:].replace("\n", " | "))
    compare(sel, name)
    log(f"   built in {(time.time() - t0) / 60:.0f} min")


def compare(sel, name):
    s1 = io.load("test", "s1", ["idx", "eid", "country"]); x = io.load("test", "x", ["idx", "eid"])
    sub = pl.read_csv(config.OUT / "matching_results.tsv", separator="\t", infer_schema=False)
    f = (sub.filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
         .with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
         .join(s1.select(pl.col("idx").alias("s1"), pl.col("eid").alias("source1_entity_id")), on="source1_entity_id")
         .join(x.select(pl.col("idx").alias("x"), pl.col("eid").alias("matched_entity_ids")), on="matched_entity_ids").select("s1", "x"))
    c = s1.select(pl.col("idx").alias("s1"), "country")
    n = dict(c.group_by("country").len().rows())
    add = sel.join(f, on=["s1", "x"], how="anti").join(c, on="s1").group_by("country").len()
    rem = f.join(sel, on=["s1", "x"], how="anti").join(c, on="s1").group_by("country").len()
    k_new = sel.join(c, on="s1").group_by("country").len()
    log(f"   vs sub_F: added {dict(add.rows())}  removed {dict(rem.rows())}  matches per S1 {({k: round(v / n[k], 3) for k, v in k_new.rows()})}")
    Xr = io.load("test", "x", ["idx", "name", "addr"]).rename({"idx": "x"}); Sr = io.load("test", "s1", ["idx", "name", "addr"]).rename({"idx": "s1"})
    for lab, d in (("ADDED", sel.join(f, on=["s1", "x"], how="anti")), ("REMOVED", f.join(sel, on=["s1", "x"], how="anti"))):
        dd = d.join(c, on="s1").filter(pl.col("country") == "France")
        if dd.height == 0:
            continue
        log(f"   France {lab} examples:")
        for r in dd.sample(min(8, dd.height), seed=0).join(Xr, on="x").join(Sr, on="s1", suffix="_s").iter_rows(named=True):
            log(f"      X : {r['name']} | {r['addr']}   ->   S1: {r['name_s']} | {r['addr_s']}")


if __name__ == "__main__":
    name = sys.argv[1]
    tags = sys.argv[2].split(",") if len(sys.argv) > 2 else {"H": ["ce_v3"], "I": ["ce_v3", "ce_xlmr_fr"]}[name]
    build_test(name, tags)
