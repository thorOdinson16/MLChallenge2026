"""US/India LLM-judge kit: slices defined on the REAL stage 3 (p3), gate from out-of-fold holdout p3 (with labels).
    python -m ber.experiments.llm_kit_usin      (needs preds/s3oof_full.parquet from post_eval)
  add  : X unassigned (top p3 < 0.8), top p3 in [0.3, 0.8), top share >= 0.6, no country word
  veto : top p3 in [0.8, 0.97), share >= 0.6, no country word
-> work/llm_kit/gate_usin.parquet (labelled, 3,000 per slice), work/llm_kit/usin_slices.parquet (test, <= 25k per slice)
"""
import polars as pl

from .. import config, io
from . import llm_kit as K

CAP_GATE, CAP_TEST = 3000, 25000


def slices(d):
    ok = (pl.col("share") >= 0.6) & ~pl.col("cw")
    return pl.concat([d.filter(ok & (pl.col("p") >= 0.3) & (pl.col("p") < 0.8)).with_columns(slice=pl.lit("add")),
                      d.filter(ok & (pl.col("p") >= 0.8) & (pl.col("p") < 0.97)).with_columns(slice=pl.lit("veto"))])


if __name__ == "__main__":
    h = pl.read_parquet(config.path("preds", "s3oof_full.parquet"), columns=["x", "s1", "p3"])
    h = K.country_word(K.texts(K.top2(h, "p3"), "train"))
    h = slices(h).join(io.load("train", "gt").with_columns(y=pl.lit(1, pl.UInt8)), on=["s1", "x"], how="left").with_columns(
        pl.col("y").fill_null(0))
    parts = []
    for sl in ("add", "veto"):
        g = h.filter(pl.col("slice") == sl)
        print(f"holdout p3 {sl}: {g.height} pairs, precision {g['y'].mean():.3f}", flush=True)
        parts.append(g.sample(n=min(CAP_GATE, g.height), seed=5))
    gate = pl.concat(parts).with_row_index("id").select("id", "slice", "x", "s1", "p", "share", "xn", "xa", "sn", "sa",
                                                       "country", "y")
    gate.write_parquet(K.OUT / "gate_usin.parquet")
    t = pl.scan_parquet(config.path("preds", "test_s3_K.parquet")).select("x", "s1", "country", "p3", "ins_country_word")
    t = t.filter((pl.col("country") != "France") & (pl.col("ins_country_word").fill_null(0) < 0.5)).select("x", "s1", "p3").collect()
    t = slices(K.country_word(K.texts(K.top2(t, "p3"), "test")))
    parts = []
    for sl in ("add", "veto"):
        g = t.filter(pl.col("slice") == sl)
        print(f"test US/IN {sl}: {g.height} pairs", flush=True)
        parts.append(g.sort("p", descending=(sl == "add")).head(CAP_TEST))
    te = pl.concat(parts).with_row_index("id").select("id", "slice", "x", "s1", "p", "share", "xn", "xa", "sn", "sa", "country")
    te.write_parquet(K.OUT / "usin_slices.parquet")
    print("gate", gate.height, "test", te.height, flush=True)
