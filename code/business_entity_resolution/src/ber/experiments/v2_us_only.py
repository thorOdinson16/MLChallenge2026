"""Unseen-country proxy: train the v2 matcher on US rows only, so India plays the role of France.

Same code path as ber.matcher2.train(), with training rows restricted to US and artefacts tagged m_v2us
(models/m_v2us_s{1,2}.txt, preds/train_m_v2us.parquet).  Nothing of the shipped v2 is overwritten.

    python -m ber.experiments.v2_us_only
"""
import sys

import polars as pl

from .. import matcher2


def main():
    orig = matcher2.train_keys

    def us_keys(name="train_all_v2"):
        k = orig(name)
        return k.with_columns(trainable=pl.col("trainable") & (pl.col("country") == "US"))

    matcher2.train_keys = us_keys
    matcher2.TAG = "m_v2us"
    matcher2.train()


if __name__ == "__main__":
    sys.exit(main())
