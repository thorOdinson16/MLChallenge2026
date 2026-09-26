"""v2 matcher: full-graph context + edit-fingerprint features, RAM-safe streaming.

Fixes the v1 train/serve skew: v1 built stage-2 context inside a 2M-X training sample (each S1 saw ~45% of
its records); v2 featurises EVERY candidate pair of the split, so context is computed on the complete
candidate graph exactly as at test time. Training rows are still restricted to "clean" pairs (S1 in MATCH
folds, X unseen by the encoder); holdout (fold 0) rows are never trained on.

Artefacts (never overwrite v1):  feats/train_all_v2, feats/test_v2, models/m_v2_s{1,2}.txt,
preds/{train,test}_m_v2.parquet (x, s1, src, p1, p2).
"""
import shutil
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from . import config, features, io, normalize, split

TAG = "m_v2"
NF = 4
META = {"x", "s1", "y", "fold", "p0", "r0", "p1", "p2", "trainable", "country"}
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=255, min_data_in_leaf=100,
              feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0, verbose=-1,
              num_threads=14, deterministic=True, force_row_wise=True, seed=config.SEED)


# ------------------------------------------------------------------------------------------ features
def featurize(split_name: str, cand: str, out: str, chunk: int = 1_500_000):
    shutil.rmtree(config.path("feats", out, "_").parent, ignore_errors=True)
    config.path("feats", out, "_").parent.mkdir(parents=True, exist_ok=True)
    from . import candidates
    c = candidates.load(cand)
    s1 = normalize.load(split_name, "s1", features.S1_COLS + ["country"])
    xn = normalize.load(split_name, "x", features.X_COLS + ["country"])
    c = c.join(xn.select(pl.col("idx").alias("x"), "country"), on="x")
    part = 0
    for (country,), cc in c.group_by("country", maintain_order=True):
        t0 = time.time()
        ctx = features.FeatureContext(s1.filter(pl.col("country") == country),
                                      xn.filter(pl.col("country") == country), country=country)
        for i in range(0, cc.height, chunk):
            f = features.featurize(ctx, cc.slice(i, chunk).drop("country")).with_columns(country=pl.lit(country))
            f.write_parquet(config.path("feats", out, f"{part:04d}.parquet"))
            part += 1
        print(f"  featurized {split_name}/{out} {country}: {cc.height} pairs in {time.time() - t0:.0f}s", flush=True)
        del ctx


def parts(name):
    return sorted(config.path("feats", name).glob("*.parquet"))


def load_cols(name, cols, row_filter=None):
    out = []
    for p in parts(name):
        d = pl.read_parquet(p, columns=cols)
        if row_filter is not None:
            d = d.filter(row_filter)
        out.append(d)
    return pl.concat(out)


def feat_cols(name):
    sch = pl.read_parquet_schema(parts(name)[0])
    return [c for c, t in sch.items() if c not in META and not c.startswith("c_") and t.is_numeric()]


# ------------------------------------------------------------------------------------------ labels / folds
def train_keys(name="train_all_v2") -> pl.DataFrame:
    """x, s1, src, country, y, fold (grouped by true cluster), trainable."""
    k = load_cols(name, ["x", "s1", "src", "country"])
    gt = io.load("train", "gt")
    k = k.join(gt.with_columns(y=pl.lit(1, pl.UInt8)), on=["s1", "x"], how="left").with_columns(pl.col("y").fill_null(0))
    cl = gt.select("x", pl.col("s1").alias("true_s1"))
    k = k.join(cl, on="x", how="left").with_columns(
        cluster=pl.when(pl.col("true_s1").is_null()).then(pl.col("x").cast(pl.Int64) + 10_000_000)
        .otherwise(pl.col("true_s1").cast(pl.Int64)))
    k = k.with_columns(fold=((pl.col("cluster") * 2246822519) % 4294967296 % NF).cast(pl.UInt8)).drop("cluster", "true_s1")
    match_s1 = pl.DataFrame({"s1": split.s1_in(split.MATCH)})
    clean = split.clean_x()
    k = k.with_columns(trainable=pl.col("s1").is_in(match_s1["s1"].implode()) & pl.col("x").is_in(clean["x"].implode()))
    return k


# ------------------------------------------------------------------------------------------ context (full graph)
def context(k: pl.DataFrame, split_name: str, sim_chunk: int = 2_000_000) -> pl.DataFrame:
    """k: x, s1, src, p1 over the COMPLETE candidate graph of a split. Returns x, s1 + c_* columns."""
    fs = k.sort(["p1", "s1"], descending=[True, False], maintain_order=True)
    xs = fs.group_by("x", maintain_order=True).agg(
        _t1=pl.col("p1").first(), _t2=pl.col("p1").slice(1, 1).first().fill_null(0.0),
        c_sum_x=pl.col("p1").sum(), c_n_cand_x=pl.len().cast(pl.Float32))
    best = fs.group_by("s1", maintain_order=True).agg(
        b1=pl.col("x").first(), b1p=pl.col("p1").first(),
        b2=pl.col("x").slice(1, 1).first(), b2p=pl.col("p1").slice(1, 1).first())
    f = k.join(xs, on="x", how="left").with_columns(
        c_p1=pl.col("p1"),
        c_rank_x=(pl.col("p1").rank("ordinal", descending=True).over("x")).cast(pl.Float32),
        c_gap_x=pl.when(pl.col("p1") >= pl.col("_t1")).then(pl.col("p1") - pl.col("_t2"))
                  .otherwise(pl.col("p1") - pl.col("_t1")),
        c_rank_s=(pl.col("p1").rank("ordinal", descending=True).over("s1")).cast(pl.Float32),
        c_sum_s_other=pl.col("p1").sum().over("s1") - pl.col("p1"),
        c_n_strong_s=(pl.col("p1") > 0.5).sum().over("s1").cast(pl.Float32),
        c_n_cand_s=pl.len().over("s1").cast(pl.Float32),
        c_n_strong_same_src=((pl.col("p1") > 0.5).sum().over(["s1", "src"])).cast(pl.Float32),
    ).drop("_t1", "_t2")
    f = f.join(best, on="s1", how="left").with_columns(
        other=pl.when(pl.col("x") == pl.col("b1")).then(pl.col("b2")).otherwise(pl.col("b1")),
        c_other_p=pl.when(pl.col("x") == pl.col("b1")).then(pl.col("b2p")).otherwise(pl.col("b1p")),
    ).drop("b1", "b1p", "b2", "b2p")
    xn = normalize.load(split_name, "x", ["idx", "nm_core", "ad"])
    need = pl.concat([f.select(pl.col("x").alias("idx")), f.select(pl.col("other").alias("idx"))]).unique().drop_nulls()
    xn = xn.join(need, on="idx", how="semi")
    ns, ads = [], []
    for i in range(0, f.height, sim_chunk):
        a = (f.slice(i, sim_chunk).select("x", "other")
              .join(xn.rename({"idx": "x", "nm_core": "n_a", "ad": "a_a"}), on="x", how="left", maintain_order="left")
              .join(xn.rename({"idx": "other", "nm_core": "n_b", "ad": "a_b"}), on="other", how="left",
                    maintain_order="left").fill_null(""))
        ns.append(cpdist(a["n_a"].to_list(), a["n_b"].to_list(), scorer=fuzz.token_sort_ratio, workers=-1) / 100)
        ads.append(cpdist(a["a_a"].to_list(), a["a_b"].to_list(), scorer=fuzz.token_set_ratio, workers=-1) / 100)
    has = f["other"].is_not_null().to_numpy()
    f = f.with_columns(c_other_name_sim=pl.Series(np.where(has, np.concatenate(ns), -1).astype(np.float32)),
                       c_other_addr_sim=pl.Series(np.where(has, np.concatenate(ads), -1).astype(np.float32)))
    return f.select(["x", "s1"] + [c for c in f.columns if c.startswith("c_")])


# ------------------------------------------------------------------------------------------ train / predict
def _matrix(name, cols, rows: pl.DataFrame | None, extra: pl.DataFrame | None = None):
    """Stack feature rows (optionally only keys in `rows`, optionally joined with `extra` on x,s1)."""
    mats, keys = [], []
    for p in parts(name):
        d = pl.read_parquet(p, columns=["x", "s1"] + [c for c in cols if not c.startswith("c_")])
        if rows is not None:
            d = d.join(rows, on=["x", "s1"], how="semi")
        if extra is not None:
            d = d.join(extra, on=["x", "s1"], how="left")
        keys.append(d.select("x", "s1"))
        mats.append(d.select(cols).to_numpy().astype(np.float32))
    return pl.concat(keys), np.concatenate(mats)


def _predict_all(name, model, extra=None):
    cols = model.feature_name()
    out = []
    for p in parts(name):
        d = pl.read_parquet(p, columns=["x", "s1"] + [c for c in cols if not c.startswith("c_")])
        if extra is not None:
            d = d.join(extra, on=["x", "s1"], how="left")
        out.append(d.select("x", "s1").with_columns(
            p=pl.Series(model.predict(d.select(cols).to_numpy().astype(np.float32), num_threads=14).astype(np.float32))))
    return pl.concat(out)


def train(rounds1=700, rounds2=700, max_rows=6_000_000):
    t0 = time.time()
    k = train_keys()
    tr = k.filter(pl.col("trainable"))
    if tr.height > max_rows:   # sample whole clusters (by fold hash buckets) to stay group-consistent
        tr = tr.sample(max_rows, seed=config.SEED)
    print(f"keys {k.height}, trainable rows {tr.height} (pos {tr['y'].sum()})", flush=True)
    cols1 = feat_cols("train_all_v2")
    keys, X = _matrix("train_all_v2", cols1, tr.select("x", "s1"))
    lab = keys.join(tr.select("x", "s1", "y", "fold"), on=["x", "s1"], how="left")
    y, fold = lab["y"].to_numpy(), lab["fold"].to_numpy()
    oof = np.zeros(len(y), dtype=np.float32)
    for f_ in range(NF):
        m = lgb.train(PARAMS, lgb.Dataset(X[fold != f_], y[fold != f_], feature_name=cols1), rounds1)
        oof[fold == f_] = m.predict(X[fold == f_], num_threads=14)
        print(f"  stage-1 fold {f_} {time.time() - t0:.0f}s", flush=True)
    m1 = lgb.train(PARAMS, lgb.Dataset(X, y, feature_name=cols1), rounds1)
    m1.save_model(str(config.path("models", f"{TAG}_s1.txt")))
    del X
    # p1 for every pair: OOF on training rows, full model elsewhere
    p1 = _predict_all("train_all_v2", m1).rename({"p": "p1"})
    p1 = p1.join(keys.with_columns(oof=pl.Series(oof)), on=["x", "s1"], how="left").with_columns(
        p1=pl.coalesce("oof", "p1")).drop("oof")
    kk = k.select("x", "s1", "src", "country").join(p1, on=["x", "s1"])
    ctx = pl.concat([context(g.drop("country"), "train") for _, g in kk.group_by("country")])
    print(f"  context built {time.time() - t0:.0f}s", flush=True)
    cols2 = cols1 + [c for c in ctx.columns if c.startswith("c_")]
    keys2, X2 = _matrix("train_all_v2", cols2, tr.select("x", "s1"), extra=ctx)
    y2 = keys2.join(tr.select("x", "s1", "y"), on=["x", "s1"], how="left")["y"].to_numpy()
    m2 = lgb.train(PARAMS, lgb.Dataset(X2, y2, feature_name=cols2), rounds2)
    m2.save_model(str(config.path("models", f"{TAG}_s2.txt")))
    del X2
    imp = sorted(zip(m2.feature_importance("gain"), cols2), reverse=True)[:30]
    print("top stage-2:", [(c, round(g / 1e3)) for g, c in imp], flush=True)
    p2 = _predict_all("train_all_v2", m2, extra=ctx).rename({"p": "p2"})
    out = kk.join(p2, on=["x", "s1"])
    out.write_parquet(config.path("preds", f"train_{TAG}.parquet"))
    print(f"train done {time.time() - t0:.0f}s", flush=True)
    return out


def predict_test():
    t0 = time.time()
    m1 = lgb.Booster(model_file=str(config.path("models", f"{TAG}_s1.txt")))
    m2 = lgb.Booster(model_file=str(config.path("models", f"{TAG}_s2.txt")))
    k = load_cols("test_v2", ["x", "s1", "src", "country"])
    p1 = _predict_all("test_v2", m1).rename({"p": "p1"})
    kk = k.join(p1, on=["x", "s1"])
    ctx = pl.concat([context(g.drop("country"), "test") for _, g in kk.group_by("country")])
    p2 = _predict_all("test_v2", m2, extra=ctx).rename({"p": "p2"})
    out = kk.join(p2, on=["x", "s1"])
    out.write_parquet(config.path("preds", f"test_{TAG}.parquet"))
    print(f"test predicted {out.height} pairs in {time.time() - t0:.0f}s", flush=True)
    return out


# ------------------------------------------------------------------------------------------ stage 2 + cross-encoder
def _ce(split_name):
    p = config.path("preds", f"ce_{split_name}.parquet")
    return pl.read_parquet(p) if p.exists() else None


def _ctx_with_ce(kk: pl.DataFrame, split_name: str) -> pl.DataFrame:
    ctx = pl.concat([context(g.drop("country"), split_name) for _, g in kk.group_by("country")])
    ce = _ce(split_name)
    if ce is not None:
        ctx = ctx.join(ce, on=["x", "s1"], how="left").with_columns(c_ce=pl.col("ce").fill_null(-99.0)).drop("ce")
    return ctx


def retrain_stage2(tag_out="m_v2ce", rounds2=700, max_rows=6_000_000, seeds=(0,)):
    """Reuse v2 stage-1 (OOF p1 in preds/train_m_v2) and retrain stage 2 with the cross-encoder feature (c_ce).
    Several seeds are averaged (bagging)."""
    t0 = time.time()
    k = train_keys()
    tr = k.filter(pl.col("trainable"))
    if tr.height > max_rows:
        tr = tr.sample(max_rows, seed=config.SEED)
    p1 = pl.read_parquet(config.path("preds", "train_m_v2.parquet"), columns=["x", "s1", "src", "country", "p1"])
    ctx = _ctx_with_ce(p1, "train")
    print(f"  context+ce {time.time() - t0:.0f}s", flush=True)
    cols2 = feat_cols("train_all_v2") + [c for c in ctx.columns if c.startswith("c_")]
    keys2, X2 = _matrix("train_all_v2", cols2, tr.select("x", "s1"), extra=ctx)
    y2 = keys2.join(tr.select("x", "s1", "y"), on=["x", "s1"], how="left")["y"].to_numpy()
    models = []
    for sd in seeds:
        m = lgb.train({**PARAMS, "seed": config.SEED + sd, "bagging_seed": sd + 1, "feature_fraction_seed": sd + 7},
                      lgb.Dataset(X2, y2, feature_name=cols2), rounds2)
        m.save_model(str(config.path("models", f"{tag_out}_s2_{sd}.txt")))
        models.append(m)
        print(f"  stage-2 seed {sd} {time.time() - t0:.0f}s", flush=True)
    del X2
    imp = sorted(zip(models[0].feature_importance("gain"), cols2), reverse=True)[:15]
    print("top stage-2:", [(c, round(g / 1e3)) for g, c in imp], flush=True)
    for split_name, name in (("train", "train_all_v2"), ("test", "test_v2")):
        base = p1 if split_name == "train" else None
        if split_name == "test":
            m1 = lgb.Booster(model_file=str(config.path("models", f"{TAG}_s1.txt")))
            kk = load_cols("test_v2", ["x", "s1", "src", "country"]).join(
                _predict_all("test_v2", m1).rename({"p": "p1"}), on=["x", "s1"])
            c = _ctx_with_ce(kk, "test")
        else:
            kk, c = base, ctx
        ps = [_predict_all(name, m, extra=c).rename({"p": f"p_{i}"}) for i, m in enumerate(models)]
        out = kk
        for q in ps:
            out = out.join(q, on=["x", "s1"])
        out = out.with_columns(p2=pl.mean_horizontal([f"p_{i}" for i in range(len(models))]).cast(pl.Float32))
        out.select("x", "s1", "src", "country", "p1", "p2").write_parquet(config.path("preds", f"{split_name}_{tag_out}.parquet"))
        print(f"  predicted {split_name} {time.time() - t0:.0f}s", flush=True)
