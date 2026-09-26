"""v2 driver:  python -m ber.v2 feats | train | eval | test | submit | all

Decoder selection is tuned on holdout half A and reported on half B (v1 tuned and scored on the same fold).
Post-filters are data-derived rules computed from unlabeled inputs (documented in README):
  country_word : drop pairs whose X name inserts the country word absent from the S1 name (decoy signature;
                 holdout India: 127 decoys, 0 true)
  fr_swap      : (A/B variant, France only) drop pairs where one real word is swapped for another real word
"""
import json
import subprocess
import sys

import numpy as np
import polars as pl

from . import candidates, config, decode, evaluate, io, matcher2, split, submit

CAND_TRAIN, CAND_TEST = "train_pruned", "test_pruned"
RULE_COLS = ["x", "s1", "ins_country_word", "swap_real", "ins_lr_real_max"]


def feats():
    matcher2.featurize("train", CAND_TRAIN, "train_all_v2")
    matcher2.featurize("test", CAND_TEST, "test_v2")


def _rules(name):
    return matcher2.load_cols(name, RULE_COLS + ["country"])


def decode_cfg(pred: pl.DataFrame, rules: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """pred: x, s1, src, country, p2. Returns selected (s1, x)."""
    p = pred.join(rules.drop("country"), on=["x", "s1"], how="left")
    if cfg.get("country_word"):
        p = p.filter(pl.col("ins_country_word").fill_null(0) < 0.5)
    if cfg.get("fr_swap"):
        p = p.filter(~((pl.col("country") == "France") & (pl.col("swap_real") > 0.5)
                       & (pl.col("ins_lr_real_max") < 1.0)))
    if cfg["dec"] == "thr":
        thr = pl.lit(cfg["thr"])
        for c, t in cfg.get("thr_by_country", {}).items():   # e.g. {"France": 0.5}: label-free prior matching
            thr = pl.when(pl.col("country") == c).then(pl.lit(t)).otherwise(thr)
        sel = decode.assign_unique(p, "p2").filter(pl.col("p2") >= thr)
    else:
        s = decode.decode(p.select("x", "s1", "p2"), "p2", miss=cfg.get("miss", 0.0))
        sel = s.join(p.select("x", "s1", "src", "p2"), on=["x", "s1"])
    if cfg.get("caps"):
        sel = decode.apply_caps(sel, "p2")
    return sel.select("s1", "x")


def evaluate_v2():
    hold = pl.DataFrame({"s1": split.holdout_s1()})
    pred = pl.read_parquet(config.path("preds", "train_m_v2.parquet"))
    ev_x = pred.join(hold, on="s1", how="semi").select("x").unique()
    pred = pred.join(ev_x, on="x", how="semi")          # all candidates of X touching holdout S1
    rules = _rules("train_all_v2").join(ev_x, on="x", how="semi")
    gt = io.load("train", "gt")
    hold = hold.with_columns(
        half=((pl.col("s1").cast(pl.UInt64) * 3266489917) % 4294967296 % 2).cast(pl.UInt8))
    meta = io.load("train", "s1", ["idx", "country"]).rename({"idx": "s1"})
    A = hold.filter(pl.col("half") == 0).join(meta, on="s1")
    B = hold.filter(pl.col("half") == 1).join(meta, on="s1")
    cfgs = [{"dec": "thr", "thr": t, "caps": c, "country_word": w}
            for t in (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9) for c in (False, True) for w in (False, True)]
    cfgs += [{"dec": "expf", "miss": m, "caps": True, "country_word": True} for m in (0.0, 0.05)]
    res = []
    for cfg in cfgs:
        sel = decode_cfg(pred, rules, cfg)
        fa = evaluate.macro_f05(sel, gt, A["s1"])
        res.append((fa, cfg))
        print(f"  half A {fa:.5f}  {cfg}", flush=True)
    best = max(res, key=lambda r: r[0])[1]
    sel = decode_cfg(pred, rules, best)
    print("BEST (chosen on A):", best, flush=True)
    rb = evaluate.report(sel, gt, B, "v2 half B")
    rall = evaluate.report(sel, gt, pl.concat([A, B]), "v2 full holdout")
    v1like = decode_cfg(pred, rules, {"dec": "thr", "thr": 0.8})
    evaluate.report(v1like, gt, pl.concat([A, B]), "v2 thr0.8 no rules (v1 decoder)")
    json.dump({"best": best, "half_B": rb, "full": rall}, open(config.path("models", "m_v2_decoder.json"), "w"), indent=1)
    return best


def _write_variant(sel: pl.DataFrame, folder: str, cands: pl.DataFrame):
    d = config.OUT.parent / "submissions" / folder
    submit._write(sel, "matched_entity_ids", d / "matching_results.tsv")
    submit._write(cands, "candidate_entity_ids", d / "candidate_pairs.tsv")
    tool = config.DATA.parent / "utils" / "validate_submission.py"
    r = subprocess.run([sys.executable, str(tool), "--matching", str(d / "matching_results.tsv"),
                        "--candidate", str(d / "candidate_pairs.tsv"), "--test-dir", str(config.DATA / "test")],
                       capture_output=True, text=True)
    print(folder, "validator:", "PASS" if r.returncode == 0 else r.stdout[-1500:], flush=True)


def france_proxy(sel: pl.DataFrame, label: str):
    """Label-free check: matches per S1 and empty share per country (train prior: 3.46 and 5.6%)."""
    s1 = io.load("test", "s1", ["idx", "country"]).rename({"idx": "s1"})
    per = s1.join(sel.group_by("s1").len("n"), on="s1", how="left").fill_null(0)
    print(label, per.group_by("country").agg(mean_matches=pl.col("n").mean(), share_empty=(pl.col("n") == 0).mean())
          .sort("country"), flush=True)


def submit_v2():
    best = json.load(open(config.path("models", "m_v2_decoder.json")))["best"]
    pred = pl.read_parquet(config.path("preds", "test_m_v2.parquet"))
    rules = _rules("test_v2")
    cands = candidates.load(CAND_TEST, ["x", "s1"])
    main = decode_cfg(pred, rules, {**best, "country_word": True})
    france_proxy(main, "v2 main")
    _write_variant(main, "v2_main", cands)
    swap = decode_cfg(pred, rules, {**best, "country_word": True, "fr_swap": True})
    france_proxy(swap, "v2 + France swap rule")
    _write_variant(swap, "v2_fr_swap", cands)
    # v1 + country-word rule only (cheap France-only A/B vs v1)
    v1 = pl.read_parquet(config.path("preds", "test_m_v1.parquet")).select("x", "s1", "p2")
    tx = io.load("test", "x", ["idx", "src", "country"]).rename({"idx": "x"})
    v1 = v1.join(tx, on="x")
    v1cw = decode_cfg(v1, rules, {"dec": "thr", "thr": 0.8, "country_word": True})
    france_proxy(v1cw, "v1 + country-word rule")
    _write_variant(v1cw, "v1_country_word", cands)


def submit_fr_thr(t_fr: float = 0.5):
    """v2_main with a France-only threshold (US/India rows byte-identical to v2_main)."""
    best = json.load(open(config.path("models", "m_v2_decoder.json")))["best"]
    pred = pl.read_parquet(config.path("preds", "test_m_v2.parquet"))
    sel = decode_cfg(pred, _rules("test_v2"), {**best, "country_word": True, "thr_by_country": {"France": t_fr}})
    france_proxy(sel, f"v2 France thr {t_fr}")
    _write_variant(sel, f"v2_fr{int(t_fr * 100):02d}", candidates.load(CAND_TEST, ["x", "s1"]))


def eval_and_submit(tag: str, t_fr: float = 0.5):
    """Holdout eval of preds/train_{tag} (thr chosen on half A, reported on half B), then test submissions:
    {tag}_main (same thr everywhere) and {tag}_fr (France-only lower threshold)."""
    hold = pl.DataFrame({"s1": split.holdout_s1()}).with_columns(half=((pl.col("s1") // 10) % 2).cast(pl.UInt8))
    pred = pl.read_parquet(config.path("preds", f"train_{tag}.parquet"))
    ev_x = pred.join(hold, on="s1", how="semi").select("x").unique()
    pred = pred.join(ev_x, on="x", how="semi")
    rules = _rules("train_all_v2").join(ev_x, on="x", how="semi")
    gt = io.load("train", "gt")
    meta = io.load("train", "s1", ["idx", "country"]).rename({"idx": "s1"})
    A = hold.filter(pl.col("half") == 0).join(meta, on="s1")
    B = hold.filter(pl.col("half") == 1).join(meta, on="s1")
    res = []
    for t in (0.6, 0.7, 0.75, 0.8, 0.85, 0.9):
        cfg = {"dec": "thr", "thr": t, "caps": True, "country_word": True}
        sel = decode_cfg(pred, rules, cfg)
        res.append((evaluate.macro_f05(sel, gt, A["s1"]), t))
    fa, thr = max(res)
    print(f"{tag}: half-A sweep {[(t, round(f, 5)) for f, t in res]} -> thr {thr}", flush=True)
    cfg = {"dec": "thr", "thr": thr, "caps": True, "country_word": True}
    sel = decode_cfg(pred, rules, cfg)
    evaluate.report(sel, gt, B, f"{tag} half B")
    evaluate.report(sel, gt, pl.concat([A, B]), f"{tag} full holdout")
    t = pl.read_parquet(config.path("preds", f"test_{tag}.parquet"))
    trules = _rules("test_v2")
    cands = candidates.load(CAND_TEST, ["x", "s1"])
    main = decode_cfg(t, trules, cfg)
    france_proxy(main, f"{tag} main thr {thr}")
    _write_variant(main, f"{tag}_main", cands)
    fr = decode_cfg(t, trules, {**cfg, "thr_by_country": {"France": t_fr}})
    france_proxy(fr, f"{tag} France thr {t_fr}")
    _write_variant(fr, f"{tag}_fr{int(t_fr * 100):02d}", cands)


if __name__ == "__main__":
    steps = sys.argv[1:] or ["all"]
    if steps == ["all"]:
        steps = ["feats", "train", "eval", "test", "submit"]
    for s in steps:
        print(f"===== v2 {s}", flush=True)
        {"feats": feats, "train": matcher2.train, "eval": evaluate_v2, "test": matcher2.predict_test,
         "submit": submit_v2, "fr50": lambda: submit_fr_thr(0.5),
         "ce": lambda: (matcher2.retrain_stage2("m_v2ce", seeds=(0, 1, 2)), eval_and_submit("m_v2ce"))}[s]()
        print(f"===== v2 {s} done", flush=True)
