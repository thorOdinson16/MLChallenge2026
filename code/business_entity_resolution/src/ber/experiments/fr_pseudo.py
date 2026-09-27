"""France pseudo-labels for cross-encoder domain adaptation.

Takes the cross-encoder band of the TEST graph (same selection as ber.crossenc: stage-1 p1 in [0.01, 0.99], top-4 per X),
keeps France pairs where the stage-3 probability is confident (p3 >= HI -> 1, p3 <= LO -> 0) and writes them as extra
training rows.  Validated on a US->India proxy: pseudo-labels recover ~97% of the gain of true labels for an unseen country.

    python -m ber.experiments.fr_pseudo [scores.parquet] [out_name]    default: test_m_v2_s3ce2 -> fr_pseudo
Also writes kaggle_ce/<out_name>_kaggle.parquet (with text) for the cloud trainer (train_ce_big.py --pseudo).
"""
import sys

import polars as pl

from .. import config
from ..crossenc import add_text, band

HI, LO = 0.9, 0.05


def main(src="test_m_v2_s3ce2.parquet", out="fr_pseudo"):
    te = pl.read_parquet(config.path("preds", "test_m_v2.parquet"), columns=["x", "s1", "src", "country", "p1", "p2"])
    p3 = pl.read_parquet(config.path("preds", src), columns=["x", "s1", "p3"])
    b = band(te.filter(pl.col("country") == "France"), 0.01, 0.99, 4).join(p3, on=["x", "s1"])
    ps = b.filter((pl.col("p3") >= HI) | (pl.col("p3") <= LO)).with_columns(y=(pl.col("p3") >= HI).cast(pl.UInt8))
    print(f"[{src}] France band pairs {b.height}; pseudo-labelled {ps.height} (pos {ps['y'].mean():.3f}); "
          f"dropped ambiguous {b.height - ps.height}", flush=True)
    ps.drop("p3").write_parquet(config.path("preds", f"{out}.parquet"))
    kdir = config.OUT.parent / "kaggle_ce"
    kdir.mkdir(exist_ok=True)
    add_text(ps.drop("p3"), "test").select("x", "s1", "country", "p2", "y", "ta", "tb").write_parquet(
        kdir / f"{out}_kaggle.parquet", compression="zstd")


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
