"""Exact re-implementation of the leaderboard metric plus blocking diagnostics.

Macro F0.5 over S1 entities. Per entity with tp true positives, np predictions and nt truths:
    F0.5 = 1.25*tp / (0.25*nt + np)      (and 1.0 when nt == np == 0)
"""
import polars as pl


def per_entity_f05(pred: pl.DataFrame, gt: pl.DataFrame, s1_universe: pl.Series) -> pl.DataFrame:
    """pred/gt: frames with u32 columns s1, x. Returns one row per S1 in the universe."""
    u = pl.DataFrame({"s1": s1_universe.cast(pl.UInt32)})
    gt = gt.join(u, on="s1", how="semi")
    pred = pred.join(u, on="s1", how="semi").unique(["s1", "x"])
    tp = pred.join(gt, on=["s1", "x"], how="semi").group_by("s1").len("tp")
    npred = pred.group_by("s1").len("np")
    nt = gt.group_by("s1").len("nt")
    r = (u.join(tp, on="s1", how="left").join(npred, on="s1", how="left").join(nt, on="s1", how="left")
          .fill_null(0))
    return r.with_columns(
        f05=pl.when((pl.col("nt") == 0) & (pl.col("np") == 0)).then(1.0)
              .otherwise(1.25 * pl.col("tp") / (0.25 * pl.col("nt") + pl.col("np")).clip(1e-9)))


def macro_f05(pred, gt, s1_universe) -> float:
    return per_entity_f05(pred, gt, s1_universe)["f05"].mean()


def report(pred, gt, s1_meta: pl.DataFrame, label="") -> dict:
    """s1_meta: s1 (u32) + country for the evaluated universe. Prints breakdowns."""
    r = per_entity_f05(pred, gt, s1_meta["s1"]).join(s1_meta.select("s1", "country"), on="s1")
    out = {"all": r["f05"].mean()}
    tot = r.select(pl.col("tp").sum(), pl.col("np").sum(), pl.col("nt").sum()).row(0)
    out["micro_P"] = tot[0] / max(tot[1], 1)
    out["micro_R"] = tot[0] / max(tot[2], 1)
    for c, g in r.group_by("country"):
        out[c[0]] = g["f05"].mean()
    single = r.filter(pl.col("nt") == 0)
    out["singleton_acc"] = single["f05"].mean() if single.height else float("nan")
    print(f"[eval {label}] " + "  ".join(f"{k}={(v if v is not None else float('nan')):.5f}" for k, v in out.items()))
    return out


def blocking_recall(cands: pl.DataFrame, gt: pl.DataFrame, x_universe: pl.Series | None = None) -> float:
    """Share of true (s1, x) links present in the candidate set (restricted to x_universe)."""
    if x_universe is not None:
        gt = gt.join(pl.DataFrame({"x": x_universe.cast(pl.UInt32)}), on="x", how="semi")
    hit = gt.join(cands.select("s1", "x"), on=["s1", "x"], how="semi").height
    return hit / max(gt.height, 1)
