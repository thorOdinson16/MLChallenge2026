"""Experiment: does calibrating p2 let the expected-F decoder beat the flat-threshold decoder?

ARCHITECTURE.md's own log flags this as unresolved: the expected-F decoder (decode.decode,
a proper Poisson-binomial optimum) scored 0.9848 macro-F0.5 on holdout, worse than a flat
threshold at 0.9856, because p2 isn't calibrated. This fits an isotonic map (global, and
separately per country) and re-scores both decoders on a held-out half of EVAL_X so the
reported numbers aren't inflated by fitting and scoring the calibration on the same rows.

Needs work/preds/eval_m_v1.parquet (out-of-fold holdout predictions for EVAL_X: x, s1, p2).

    cd code/business_entity_resolution/src
    python -m ber.experiments.calibration
"""
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

from ber import config, decode, evaluate, io, split

TAG = "m_v1"

ev = pl.read_parquet(config.path("preds", f"eval_{TAG}.parquet"))
gt = io.load("train", "gt")
s1_meta = io.load("train", "s1", ["idx", "country"]).rename({"idx": "s1"})

# EVAL_X's S1 universe. NOTE: matcher.eval_s1()'s own filter (`s1 * 40503 % 3 == 0`) is a
# no-op -- 40503 is divisible by 3, so that condition is always true and eval_s1() actually
# returns the *entire* holdout set, not the ~1/3 subset its code implies. Doesn't invalidate
# anything (more eval data is fine); just means EVAL_X == holdout_s1() in practice. We use
# the full holdout directly here rather than reproducing the (inert) filter.
eval_s1 = split.holdout_s1()

# split EVAL_X's S1 entities in half: fit calibration on one half, score on the other.
# NOTE: don't reuse split.py's fold-hash constant (2654435761) for this -- it's odd, and
# N_FOLDS=10 is even, so that hash preserves input parity end-to-end. HOLDOUT (fold 0, even)
# is therefore silently all-even-indexed S1 rows; a mod-2 split with the same constant on an
# already-all-even set is degenerate (confirmed by a smoke test: every row landed in one
# half). A plain random sample split has no such correlation.
fit_s1 = eval_s1.sample(fraction=0.5, seed=42, shuffle=True)
score_s1 = eval_s1.filter(~eval_s1.is_in(fit_s1))


def meta_for(ids: pl.Series) -> pl.DataFrame:
    return s1_meta.join(pl.DataFrame({"s1": ids}), on="s1", how="semi")


ev_lab = ev.join(gt.with_columns(y=pl.lit(1, pl.UInt8)), on=["s1", "x"], how="left").with_columns(
    pl.col("y").fill_null(0))
fit_rows = ev_lab.join(pl.DataFrame({"s1": fit_s1}), on="s1", how="semi")
score_rows = ev_lab.join(pl.DataFrame({"s1": score_s1}), on="s1", how="semi")
print(f"fit half: {fit_rows.height:,} rows, {int(fit_rows['y'].sum()):,} positive")
print(f"score half: {score_rows.height:,} rows, {int(score_rows['y'].sum()):,} positive")


def sweep(rows: pl.DataFrame, prob_col: str, label: str, thrs=(0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9),
          misses=(0.0, 0.03, 0.05, 0.08)):
    best = (0.0, "")
    for thr in thrs:
        r = evaluate.report(decode.decode_threshold(rows, thr, prob_col), gt, meta_for(score_s1),
                            f"{label} thr={thr}")
        if r["all"] > best[0]:
            best = (r["all"], f"thr={thr}")
    for miss in misses:
        r = evaluate.report(decode.decode(rows, prob_col, miss=miss), gt, meta_for(score_s1),
                            f"{label} expF miss={miss}")
        if r["all"] > best[0]:
            best = (r["all"], f"expF miss={miss}")
    return best


print("\n--- baseline: uncalibrated p2 ---")
best_base = sweep(score_rows, "p2", "base")

print("\n--- global isotonic calibration (fit on the other half) ---")
iso = IsotonicRegression(out_of_bounds="clip").fit(fit_rows["p2"].to_numpy(), fit_rows["y"].to_numpy())
score_g = score_rows.with_columns(p2c=pl.Series(iso.predict(score_rows["p2"].to_numpy()).astype(np.float32)))
best_global = sweep(score_g, "p2c", "global-cal")

print("\n--- per-country isotonic calibration (fit on the other half, per country) ---")
parts = []
for country in s1_meta["country"].unique().to_list():
    fit_c = fit_rows.join(meta_for(fit_s1).filter(pl.col("country") == country).select("s1"), on="s1", how="semi")
    sc_c = score_rows.join(meta_for(score_s1).filter(pl.col("country") == country).select("s1"), on="s1", how="semi")
    if fit_c.height == 0 or sc_c.height == 0:
        continue
    isoc = IsotonicRegression(out_of_bounds="clip").fit(fit_c["p2"].to_numpy(), fit_c["y"].to_numpy())
    parts.append(sc_c.with_columns(p2c=pl.Series(isoc.predict(sc_c["p2"].to_numpy()).astype(np.float32))))
score_pc = pl.concat(parts)
best_pc = sweep(score_pc, "p2c", "percountry-cal")

print("\n===== SUMMARY (all numbers on the held-out score half, not seen by calibration fitting) =====")
print(f"  baseline (uncalibrated) : {best_base}")
print(f"  global isotonic         : {best_global}")
print(f"  per-country isotonic    : {best_pc}")
