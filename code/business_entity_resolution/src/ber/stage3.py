"""Stage 3: X-competition re-scorer trained on the labeled holdout.

v2's p1/p2 are out-of-sample on the holdout S1 (fold 0 was never trained on), so those rows can honestly train one
more small model on top of v2. Its main input is the gap between a pair's final probability p2 and the strongest
competing pair of the same X, i.e. one more round of collective inference using final rather than stage-1
probabilities. Grouped 5-fold CV on the holdout: 0.98885 -> 0.98919 without the cross-encoder, -> 0.99054 with the first cross-encoder, -> 0.99096 with ce_v2 (thr 0.7-0.8 plateau, US+India).

    python -m ber.stage3 train    fit on all holdout rows -> models/m_v2_s3.txt
    python -m ber.stage3 test     score the test graph    -> preds/test_m_v2_s3.parquet
    python -m ber.stage3 submit   decode + write output/sub_stage3_*.tsv (US/India only, and all countries)

Features avoid anything tied to the size of the S1 pool (e.g. namesake counts) because the test pool is smaller.
"""
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

from . import config, decode, io, matcher2, split, submit

TAG = "m_v2_s3ce2"
CE_TAG = "ce_v2"     # cross-encoder run (ber.crossenc): all non-holdout folds, band p1 in [0.01, 0.99], top-4 per X
THR = 0.75
STAGE3_COLS = ["l1", "l2", "rank_p2_x", "max_p2_x", "sum_p2_x", "sum_p2_s_other", "n_conf_s_src_other", "gap_p2_x",
               "src_f", "ad_empty", "ce", "ce_max_x", "ce_gap", "l2_plus_ce"]   # last four: cross-encoder (ber.crossenc)
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=14, seed=1)
ROUNDS = 400


def _logit(c: str) -> pl.Expr:
    p = pl.col(c).clip(1e-6, 1 - 1e-6)
    return (p / (1 - p)).log()


def build(pred: pl.DataFrame, split_name: str, ce: pl.DataFrame) -> pl.DataFrame:
    """pred: x, s1, src, country, p1, p2 over the COMPLETE candidate graph of a split.
    ce: x, s1, ce cross-encoder scores for the uncertain band (null elsewhere)."""
    xn = pl.read_parquet(config.path("norm", f"{split_name}_x.parquet"), columns=["idx", "ad"]).rename({"idx": "x"})
    xn = xn.with_columns(ad_empty=(pl.col("ad").str.strip_chars() == "").cast(pl.Float32)).select("x", "ad_empty")
    d = pred.join(xn, on="x", how="left").join(ce, on=["x", "s1"], how="left").with_columns(l1=_logit("p1"), l2=_logit("p2"))
    d = d.with_columns(
        rank_p2_x=pl.col("p2").rank("ordinal", descending=True).over("x").cast(pl.Float32),
        max_p2_x=pl.col("p2").max().over("x"), sum_p2_x=pl.col("p2").sum().over("x"),
        sum_p2_s_other=pl.col("p2").sum().over("s1") - pl.col("p2"),
        n_conf_s_src=(pl.col("p2") >= 0.8).sum().over(["s1", "src"]).cast(pl.Float32),
        conf_self=(pl.col("p2") >= 0.8).cast(pl.Float32), src_f=pl.col("src").cast(pl.Float32))
    d = d.with_columns(ce_max_x=pl.col("ce").max().over("x"))
    return d.with_columns(
        gap_p2_x=pl.col("p2") - pl.when(pl.col("p2") >= pl.col("max_p2_x")).then(0.0).otherwise(pl.col("max_p2_x")),
        n_conf_s_src_other=pl.col("n_conf_s_src") - pl.col("conf_self"),
        ce_gap=pl.col("ce") - pl.col("ce_max_x"), l2_plus_ce=pl.col("l2") + pl.col("ce").fill_null(0.0))


def raw_cols() -> list:
    return [c for c in matcher2.feat_cols("train_all_v2") if c != "src"]


def train():
    tr = pl.read_parquet(config.path("preds", "train_m_v2.parquet"))
    hold = pl.DataFrame({"s1": split.holdout_s1()})
    ev_x = tr.join(hold, on="s1", how="semi").select("x").unique()
    ce = pl.read_parquet(config.path("preds", f"{CE_TAG}_holdout.parquet"))
    d = build(tr.join(ev_x, on="x", how="semi"), "train", ce)
    gt = io.load("train", "gt").with_columns(y=pl.lit(1, pl.UInt8))
    d = d.join(gt, on=["s1", "x"], how="left").with_columns(pl.col("y").fill_null(0)).join(hold, on="s1", how="semi")
    rc = raw_cols()
    raw = pl.concat([pl.read_parquet(p, columns=["x", "s1"] + rc).join(d.select("x", "s1"), on=["x", "s1"], how="semi")
                     for p in matcher2.parts("train_all_v2")])
    d = d.join(raw, on=["x", "s1"], how="left")
    cols = STAGE3_COLS + rc
    X = d.select(cols).to_numpy().astype(np.float32)
    m = lgb.train(PARAMS, lgb.Dataset(X, d["y"].to_numpy(), feature_name=cols), ROUNDS)
    m.save_model(str(config.path("models", f"{TAG}.txt")))
    print(f"stage-3 trained on {d.height} out-of-sample holdout rows", flush=True)


def predict_test():
    m = lgb.Booster(model_file=str(config.path("models", f"{TAG}.txt")))
    cols = STAGE3_COLS + raw_cols()
    assert m.feature_name() == cols, "feature order mismatch"
    te = pl.read_parquet(config.path("preds", "test_m_v2.parquet"))
    ce = pl.read_parquet(config.path("preds", f"{CE_TAG}_test.parquet"))
    dt = build(te, "test", ce).select(["x", "s1"] + STAGE3_COLS + ["p2", "country", "src"])
    outs = []
    for p in matcher2.parts("test_v2"):
        f = pl.read_parquet(p, columns=["x", "s1"] + raw_cols()).join(dt, on=["x", "s1"], how="inner")
        p3 = m.predict(f.select(cols).to_numpy().astype(np.float32), num_threads=14).astype(np.float32)
        outs.append(f.select("x", "s1", "src", "country", "p2", "ins_country_word").with_columns(p3=pl.Series(p3)))
    out = pl.concat(outs)
    out.write_parquet(config.path("preds", f"test_{TAG}.parquet"))
    print(f"scored {out.height} test pairs (input graph had {te.height})", flush=True)


def decode_stage3(d: pl.DataFrame, thr: float = THR) -> pl.DataFrame:
    """Same country-word rule and source caps as v2, unique assignment + threshold on p3."""
    d = d.filter(pl.col("ins_country_word").fill_null(0) < 0.5)
    sel = decode.assign_unique(d.select("x", "s1", "src", "country", "p3"), "p3").filter(pl.col("p3") >= thr)
    return decode.apply_caps(sel, "p3")


def submit_stage3(v2_main_pairs: pl.DataFrame | None = None):
    """Writes sub_stage3_usin.tsv (US/India from stage 3, France from v2_main_pairs if given) and sub_stage3_all.tsv."""
    sel = decode_stage3(pl.read_parquet(config.path("preds", f"test_{TAG}.parquet")))
    usin = sel.filter(pl.col("country") != "France").select("s1", "x")
    fr = sel.filter(pl.col("country") == "France").select("s1", "x")
    submit._write(pl.concat([usin, fr]), "matched_entity_ids", config.OUT / "sub_stage3_all.tsv")
    if v2_main_pairs is not None:
        submit._write(pl.concat([usin, v2_main_pairs]), "matched_entity_ids", config.OUT / "sub_stage3_usin.tsv")


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    for s in (["train", "test", "submit"] if step == "all" else [step]):
        {"train": train, "test": predict_test, "submit": submit_stage3}[s]()
