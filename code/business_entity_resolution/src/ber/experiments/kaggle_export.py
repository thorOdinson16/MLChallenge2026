"""Export the cross-encoder band pairs with text for the cloud trainer (kaggle/train_ce_big.py).
Same selection as `python -m ber.crossenc <tag> 0.01 0.99 4 <epochs> all`: stage-1 p1 in [0.01, 0.99], top-4 per record.

    python -m ber.experiments.kaggle_export        -> <project>/kaggle_ce/{ce_train,ce_holdout,ce_test}.parquet
"""
import sys
import polars as pl
from ber import config, io, split
from ber.crossenc import band, add_text
sys.stdout.reconfigure(encoding="utf-8")
out = config.OUT.parent / "kaggle_ce"
out.mkdir(exist_ok=True)
tr = pl.read_parquet(config.path("preds", "train_m_v2.parquet"), columns=["x", "s1", "src", "country", "p1", "p2"])
te = pl.read_parquet(config.path("preds", "test_m_v2.parquet"), columns=["x", "s1", "src", "country", "p1", "p2"])
gt = io.load("train", "gt").with_columns(y=pl.lit(1, pl.UInt8))
hold = pl.DataFrame({"s1": split.holdout_s1()})
rows = tr.join(hold, on="s1", how="anti")
trn = band(rows, 0.01, 0.99, 4).join(gt, on=["s1", "x"], how="left").with_columns(pl.col("y").fill_null(0))
hh = band(tr.join(hold, on="s1", how="semi"), 0.01, 0.99, 4).join(gt, on=["s1", "x"], how="left").with_columns(pl.col("y").fill_null(0))
tt = band(te, 0.01, 0.99, 4).with_columns(y=pl.lit(0, pl.UInt8))
for name, d, sp in (("ce_train", trn, "train"), ("ce_holdout", hh, "train"), ("ce_test", tt, "test")):
    d = add_text(d, sp).select("x", "s1", "country", "p2", "y", "ta", "tb")
    d.write_parquet(out / f"{name}.parquet", compression="zstd")
    print(name, d.height, f"pos {d['y'].mean():.3f}", (out / f"{name}.parquet").stat().st_size // 2**20, "MB")
