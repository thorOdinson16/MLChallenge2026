"""Stage 0: learned candidate pruning on retrieval signals only (cheap).

Features: each retriever's score/rank (missing -> sentinel), its gap to the x's best score for that
retriever, number of retrievers proposing the pair, best rank, candidate count. A small LightGBM predicts
p0; we keep the top-k per x. The kept set is the final candidate set (candidate_pairs.tsv).

Train it on an un-pruned candidate sample:  python -m ber.prune   (after `candidates` built 'prune_sample')
"""
import lightgbm as lgb
import numpy as np
import polars as pl

from . import candidates, config, io, split


def features(c: pl.DataFrame) -> pl.DataFrame:
    exprs = []
    for r in ("lsa_full", "lsa_addr", "lsa_name", "keys", "enc"):
        if f"{r}_score" not in c.columns:
            continue
        s = pl.col(f"{r}_score")
        exprs += [s.fill_null(-1.0).alias(f"f0_{r}_score"),
                  pl.col(f"{r}_rank").cast(pl.Float32).fill_null(99.0).alias(f"f0_{r}_rank"),
                  (s - s.max().over("x")).fill_null(-9.0).alias(f"f0_{r}_gap")]
    exprs += [pl.col("n_ret").cast(pl.Float32).alias("f0_n_ret"),
              pl.col("best_rank").cast(pl.Float32).alias("f0_best_rank"),
              pl.len().over("x").cast(pl.Float32).alias("f0_n_cands")]
    return c.with_columns(exprs)


def load_model() -> lgb.Booster:
    return lgb.Booster(model_file=str(config.path("models", "prune.txt")))


def prune(u: pl.DataFrame, model: lgb.Booster, k: int, min_p: float = 1e-4) -> pl.DataFrame:
    f = features(u)
    p0 = model.predict(f.select(model.feature_name()).to_numpy(), num_threads=14).astype(np.float32)
    return (u.with_columns(p0=pl.Series(p0))
             .with_columns(r0=(pl.col("p0").rank("ordinal", descending=True).over("x") - 1).cast(pl.UInt8))
             .filter((pl.col("r0") < k) & (pl.col("p0") >= min_p)))


def train(sample_dir: str = "prune_sample") -> lgb.Booster:
    c = features(candidates.load(sample_dir))
    gt = io.load("train", "gt").with_columns(y=pl.lit(1, pl.UInt8))
    c = c.join(gt, on=["s1", "x"], how="left").with_columns(pl.col("y").fill_null(0))
    hold = pl.DataFrame({"s1": split.holdout_s1()})
    # train on "clean" X only (true S1 unseen by the encoder), and never on holdout S1 pairs
    tr = c.join(split.clean_x(), on="x", how="semi").join(hold, on="s1", how="anti")
    cols = [col for col in c.columns if col.startswith("f0_")]
    params = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=200,
                  feature_fraction=0.9, bagging_fraction=0.5, bagging_freq=1, verbose=-1, num_threads=14)
    m = lgb.train(params, lgb.Dataset(tr.select(cols).to_numpy(), tr["y"].to_numpy(), feature_name=cols),
                  num_boost_round=300)
    m.save_model(str(config.path("models", "prune.txt")))
    # recall of the pruned set on holdout pairs, for several k
    ho = c.join(hold, on="s1", how="semi")
    n_true_total = io.load("train", "gt").join(hold, on="s1", how="semi").join(
        c.select("x").unique(), on="x", how="semi").height
    raw = c.drop([col for col in c.columns if col.startswith("f0_")])
    for k in (5, 8, 10, 12, 15, 20):
        kept = prune(raw, m, k)
        kh = kept.join(hold, on="s1", how="semi")
        print(f"prune k={k:2d}: holdout recall {kh['y'].sum() / max(n_true_total, 1):.5f}  "
              f"(pre-prune {ho['y'].sum() / max(n_true_total, 1):.5f})  pairs/x {kept.height / raw['x'].n_unique():.2f}",
              flush=True)
    return m


if __name__ == "__main__":
    train()
