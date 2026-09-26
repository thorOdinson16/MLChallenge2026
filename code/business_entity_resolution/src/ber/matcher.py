"""Two-stage matcher.

stage 1: LightGBM on pairwise features            -> p1  (out-of-fold on the training X set)
stage 2: LightGBM on pairwise + context features   -> p2
  context = how the pair ranks among the x's candidates and among the s1's candidates (by p1), the s1's
  number of strong matches, and similarity of x to the s1's strongest *other* match (cluster evidence).

X sets
  TRAIN_X: random train X records not in EVAL_X; training rows exclude every holdout S1.
  EVAL_X : all train X with a candidate in EVAL_S1 (a subset of the holdout S1 fold), so decoding for
           those S1 sees exactly what it would see at test time.
"""
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from . import candidates, config, features, io, normalize, split

NF = 4  # stage-1 OOF folds
META = {"x", "s1", "y", "fold", "p0", "r0", "country", "p1", "p2"}
P1 = dict(objective="binary", learning_rate=0.05, num_leaves=255, min_data_in_leaf=100,
          feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=14)


# ------------------------------------------------------------------------------------------ sets
def eval_s1() -> pl.Series:
    h = split.holdout_s1()
    return h.filter((h.cast(pl.UInt64) * 40503 % 3) == 0)


def make_sets(cand: str, n_train_x: int = 2_000_000):
    c = candidates.load(cand, ["x", "s1"])
    ev = c.join(pl.DataFrame({"s1": eval_s1()}), on="s1", how="semi").select("x").unique()
    pool = c.select("x").unique().join(ev, on="x", how="anti").join(split.clean_x(), on="x", how="semi")
    tr = pool.sample(min(n_train_x, pool.height), seed=config.SEED)
    ev.write_parquet(config.path("sets", "eval_x.parquet"))
    tr.write_parquet(config.path("sets", "train_x.parquet"))
    print("EVAL_X", ev.height, "TRAIN_X", tr.height)


# ------------------------------------------------------------------------------------------ features
def featurize(split_name: str, cand: str, x_ids: pl.DataFrame | None, out: str, chunk: int = 1_500_000):
    import shutil
    shutil.rmtree(config.path("feats", out, "_"), ignore_errors=True)
    shutil.rmtree(config.path("feats", out), ignore_errors=True)
    config.path("feats", out, "_").parent.mkdir(parents=True, exist_ok=True)
    c = candidates.load(cand)
    if x_ids is not None:
        c = c.join(x_ids, on="x", how="semi")
    s1 = normalize.load(split_name, "s1", features.S1_COLS + ["country"])
    xn = normalize.load(split_name, "x", features.X_COLS + ["country"])
    xn = xn.join(c.select(pl.col("x").alias("idx")).unique(), on="idx", how="semi")
    c = c.join(xn.select(pl.col("idx").alias("x"), "country"), on="x")
    part = 0
    for (country,), cc in c.group_by("country", maintain_order=True):
        t0 = time.time()
        ctx = features.FeatureContext(s1.filter(pl.col("country") == country), xn.filter(pl.col("country") == country))
        for i in range(0, cc.height, chunk):
            f = features.featurize(ctx, cc.slice(i, chunk).drop("country"))
            f.write_parquet(config.path("feats", out, f"{part:04d}.parquet"))
            part += 1
        print(f"  featurized {split_name}/{out} {country}: {cc.height} pairs in {time.time() - t0:.0f}s", flush=True)
        del ctx


def load_feats(out: str) -> pl.DataFrame:
    return pl.concat([pl.read_parquet(p) for p in sorted(config.path("feats", out).glob("*.parquet"))],
                     how="diagonal_relaxed")


def add_labels(f: pl.DataFrame) -> pl.DataFrame:
    gt = io.load("train", "gt").with_columns(y=pl.lit(1, pl.UInt8))
    return f.join(gt, on=["s1", "x"], how="left").with_columns(pl.col("y").fill_null(0))


def feat_cols(f: pl.DataFrame, stage: int):
    cols = [c for c in f.columns if c not in META and f[c].dtype.is_numeric()]
    if stage == 1:
        cols = [c for c in cols if not c.startswith("c_")]
    return cols


# ------------------------------------------------------------------------------------------ context
def context(f: pl.DataFrame, split_name: str) -> pl.DataFrame:
    """Context features from p1 over complete candidate lists."""
    fs = f.sort("p1", descending=True)
    xs = fs.group_by("x", maintain_order=True).agg(
        _t1=pl.col("p1").first(), _t2=pl.col("p1").slice(1, 1).first().fill_null(0.0),
        c_sum_x=pl.col("p1").sum(), c_n_cand_x=pl.len().cast(pl.Float32))
    # cluster evidence: the s1's two strongest x (to compare each x with the strongest *other* one)
    best = fs.group_by("s1", maintain_order=True).agg(
        b1=pl.col("x").first(), b1p=pl.col("p1").first(),
        b2=pl.col("x").slice(1, 1).first(), b2p=pl.col("p1").slice(1, 1).first())
    f = f.join(xs, on="x", how="left").with_columns(
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
    a = f.select("x", "other").join(xn.rename({"idx": "x", "nm_core": "n_a", "ad": "a_a"}), on="x", how="left").join(
        xn.rename({"idx": "other", "nm_core": "n_b", "ad": "a_b"}), on="other", how="left").fill_null("")
    has = f["other"].is_not_null().to_numpy()
    ns = cpdist(a["n_a"].to_list(), a["n_b"].to_list(), scorer=fuzz.token_sort_ratio, workers=-1) / 100
    ads = cpdist(a["a_a"].to_list(), a["a_b"].to_list(), scorer=fuzz.token_set_ratio, workers=-1) / 100
    return f.with_columns(c_other_name_sim=pl.Series(np.where(has, ns, -1).astype(np.float32)),
                          c_other_addr_sim=pl.Series(np.where(has, ads, -1).astype(np.float32))).drop("other")


# ------------------------------------------------------------------------------------------ training
def _fit(X, y, cols, rounds, params=P1, valid=None):
    ds = lgb.Dataset(X, y, feature_name=cols, free_raw_data=True)
    return lgb.train(params, ds, num_boost_round=rounds)


def train_and_eval(rounds1=600, rounds2=600, tag="m_v1"):
    hold = pl.DataFrame({"s1": split.holdout_s1()})
    tr = add_labels(load_feats("train_x"))
    tr = tr.with_columns(fold=((pl.col("x").cast(pl.UInt64) * 2246822519) % 4294967296 % NF).cast(pl.UInt8))
    # never learn from holdout S1 labels, nor from pairs whose S1 the encoder was trained on
    seen = pl.concat([hold["s1"], split.s1_in(split.ENC)])
    trainable = ~pl.col("s1").is_in(seen.implode())
    cols1 = feat_cols(tr, 1)
    print("stage-1 features:", len(cols1), "train pairs:", tr.height, "pos:", tr["y"].sum(), flush=True)
    # OOF p1 on TRAIN_X
    p1 = np.zeros(tr.height, dtype=np.float32)
    fold = tr["fold"].to_numpy()
    ok = tr.select(trainable).to_series().to_numpy()
    X1 = tr.select(cols1).to_numpy()
    y = tr["y"].to_numpy()
    for k in range(NF):
        t = time.time()
        m = _fit(X1[(fold != k) & ok], y[(fold != k) & ok], cols1, rounds1)
        p1[fold == k] = m.predict(X1[fold == k], num_threads=14)
        print(f"  stage-1 fold {k} done {time.time() - t:.0f}s", flush=True)
    m1 = _fit(X1[ok], y[ok], cols1, rounds1)
    m1.save_model(str(config.path("models", f"{tag}_s1.txt")))
    del X1
    tr = context(tr.with_columns(p1=pl.Series(p1)), "train")
    cols2 = feat_cols(tr, 2)
    ok = tr.select(trainable).to_series().to_numpy()
    m2 = _fit(tr.select(cols2).to_numpy()[ok], tr["y"].to_numpy()[ok], cols2, rounds2)
    m2.save_model(str(config.path("models", f"{tag}_s2.txt")))
    imp = sorted(zip(m2.feature_importance("gain"), cols2), reverse=True)[:25]
    print("top stage-2 features:", [(c, round(g / 1e3)) for g, c in imp], flush=True)
    del tr
    ev = predict("eval_x", "train", tag)
    ev.select("x", "s1", "p0", "p1", "p2").write_parquet(config.path("preds", f"eval_{tag}.parquet"))
    return ev


def predict(feats_name: str, split_name: str, tag: str) -> pl.DataFrame:
    m1 = lgb.Booster(model_file=str(config.path("models", f"{tag}_s1.txt")))
    m2 = lgb.Booster(model_file=str(config.path("models", f"{tag}_s2.txt")))
    f = load_feats(feats_name)
    f = f.with_columns(p1=pl.Series(m1.predict(f.select(m1.feature_name()).to_numpy(), num_threads=14)
                                    .astype(np.float32)))
    f = context(f, split_name)
    return f.with_columns(p2=pl.Series(m2.predict(f.select(m2.feature_name()).to_numpy(), num_threads=14)
                                       .astype(np.float32)))
