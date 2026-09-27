"""Light holdout check of the fuzzy-twin rule (US/India labels) on the m_v2ce p2 base, comparing name scorers.
    python -m ber.experiments.twin_light
Memory-light: lazy scans restricted to X touching holdout S1."""
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from .. import config, decode, evaluate, io, split


def words_changed(a, b):
    ta, tb = a.split(), b.split()
    ex = sum(1 for t in ta if t not in tb and max([fuzz.ratio(t, u) for u in tb] or [0]) < 75)
    mi = sum(1 for t in tb if t not in ta and max([fuzz.ratio(t, u) for u in ta] or [0]) < 75)
    return ex + mi


if __name__ == "__main__":
    hold = pl.DataFrame({"s1": split.holdout_s1()}).with_columns(half=((pl.col("s1") // 10) % 2).cast(pl.UInt8))
    lp = pl.scan_parquet(config.path("preds", "train_m_v2ce.parquet")).select("x", "s1", "src", "p2")
    ev_x = lp.join(hold.lazy(), on="s1", how="semi").select("x").unique().collect()
    p = lp.join(ev_x.lazy(), on="x", how="semi").collect()
    gt = io.load("train", "gt")
    base = decode.apply_caps(decode.assign_unique(p, "p2").filter(pl.col("p2") >= 0.85), "p2")
    xn = (pl.scan_parquet(config.path("norm", "train_x.parquet")).select(pl.col("idx").alias("x"), "nm_core", "ad")
          .join(ev_x.lazy(), on="x", how="semi").filter(pl.col("ad").fill_null("") != "")
          .with_columns(pl.col("nm_core").fill_null("")).collect())
    acc = base.select("x", "s1").join(xn, on="x").select(pl.col("x").alias("xa"), "s1", "ad", pl.col("nm_core").alias("nm_a"))
    un = xn.join(base.select("x"), on="x", how="anti")
    j = un.join(acc, on="ad").join(p.select("x", "s1", "p2"), on=["x", "s1"]).filter(pl.col("p2") >= 0.1)
    A, B = j["nm_core"].to_list(), j["nm_a"].to_list()
    j = j.with_columns(ts=pl.Series(cpdist(A, B, scorer=fuzz.token_set_ratio, workers=4)),
                       r=pl.Series(cpdist(A, B, scorer=fuzz.ratio, workers=4)),
                       wc=pl.Series([words_changed(a, b) for a, b in zip(A, B)]))
    j = j.join(gt.with_columns(y=pl.lit(1)), on=["s1", "x"], how="left").with_columns(pl.col("y").fill_null(0))
    halves = [hold.filter(pl.col("half") == h)["s1"] for h in (0, 1)]
    b = [evaluate.macro_f05(base.select("s1", "x"), gt, h) for h in halves]
    print(f"base A {b[0]:.5f} B {b[1]:.5f}", flush=True)
    rules = {"token_set>=90": pl.col("ts") >= 90, "ratio>=90": pl.col("r") >= 90,
             "token_set>=90 & no word change": (pl.col("ts") >= 90) & (pl.col("wc") == 0),
             "token_set>=90 & word change": (pl.col("ts") >= 90) & (pl.col("wc") > 0)}
    for name, cond in rules.items():
        for pmin in (0.1, 0.3, 0.5):
            add = (j.filter(cond & (pl.col("p2") >= pmin)).sort(["x", "ts", "p2"], descending=[False, True, True])
                    .unique("x", keep="first"))
            both = pl.concat([base.select("s1", "x", "src", pl.col("p2") + 2.0),
                              add.join(p.select("x", "src").unique("x"), on="x").select("s1", "x", "src", "p2")])
            sel = decode.apply_caps(both, "p2").select("s1", "x")
            f = [evaluate.macro_f05(sel, gt, h) for h in halves]
            print(f"{name:34s} p>={pmin}: add {add.height:5d} prec {add['y'].mean() if add.height else 0:.3f}"
                  f"  A {f[0] - b[0]:+.5f}  B {f[1] - b[1]:+.5f}", flush=True)
