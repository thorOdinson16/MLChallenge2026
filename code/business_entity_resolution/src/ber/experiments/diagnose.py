"""Read-only diagnostics for planning v2: where does macro-F0.5 go, and how does France differ?

Writes WORK/diag/diagnostics.md. Uses only cached predictions/features (no retraining except a tiny
leave-one-country-out probe on stage-1 features).
"""
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from ber import config, decode, evaluate, io, matcher

OUT = config.path("diag", "diagnostics.md")
lines = []


def w(s=""):
    print(s, flush=True)
    lines.append(s)


def md(df: pl.DataFrame, fmt=4):
    df = df.with_columns([pl.col(c).round(fmt) for c, t in df.schema.items() if t in (pl.Float32, pl.Float64)])
    cols = df.columns
    w("| " + " | ".join(cols) + " |")
    w("|" + "---|" * len(cols))
    for r in df.iter_rows():
        w("| " + " | ".join(str(v) for v in r) + " |")
    w()


THR = 0.8
t0 = time.time()
gt_all = io.load("train", "gt")
es = pl.DataFrame({"s1": matcher.eval_s1()})
gt = gt_all.join(es, on="s1", how="semi")
ev = pl.read_parquet(config.path("preds", "eval_m_v1.parquet"))
s1meta = io.load("train", "s1", ["idx", "country"]).rename({"idx": "s1"})
xraw = io.load("train", "x", ["idx", "src", "name", "addr"]).rename({"idx": "x"})

# ------------------------------------------------------------------ D1 holdout loss decomposition
w("# Diagnostics for v2 planning\n")
w(f"Holdout = EVAL_S1 ({es.height} S1 entities of fold 0, never trained on). Decoder: unique + thr {THR} on p2.\n")
pred = decode.decode_threshold(ev, THR, "p2").join(es, on="s1", how="semi")
r = evaluate.per_entity_f05(pred, gt, es["s1"]).join(s1meta, on="s1")
w(f"macro F0.5 = {r['f05'].mean():.5f}; true pairs = {gt.height}; predicted = {pred.height}; "
  f"TP = {r['tp'].sum()}; FP = {r['np'].sum() - r['tp'].sum()}; FN = {r['nt'].sum() - r['tp'].sum()}\n")
cat = r.with_columns(
    kind=pl.when((pl.col("nt") == 0) & (pl.col("np") > 0)).then(pl.lit("singleton_with_FP"))
    .when(pl.col("nt") == 0).then(pl.lit("singleton_ok"))
    .when(pl.col("np") == 0).then(pl.lit("missed_entirely(pred empty)"))
    .when((pl.col("np") > pl.col("tp")) & (pl.col("nt") > pl.col("tp"))).then(pl.lit("FP_and_FN"))
    .when(pl.col("np") > pl.col("tp")).then(pl.lit("FP_only"))
    .when(pl.col("nt") > pl.col("tp")).then(pl.lit("FN_only"))
    .otherwise(pl.lit("perfect")))
w("## D1. Macro-F0.5 loss by entity outcome (loss = share of total 1-F mass)\n")
tot_loss = (1 - cat["f05"]).sum()
md(cat.group_by("kind").agg(n=pl.len(), mean_f=pl.col("f05").mean(), loss_pts=(1 - pl.col("f05")).sum() / cat.height)
   .with_columns(loss_share=pl.col("loss_pts") / (tot_loss / cat.height)).sort("loss_pts", descending=True))
w("By country:\n")
md(cat.group_by("country").agg(n=pl.len(), macro_f=pl.col("f05").mean(),
                               P=pl.col("tp").sum() / pl.col("np").sum(), R=pl.col("tp").sum() / pl.col("nt").sum()))

# FN reasons
cand = ev.select("s1", "x", "p2")
best_x = ev.sort("p2", descending=True).unique("x", keep="first").select("x", pl.col("s1").alias("best_s1"),
                                                                        pl.col("p2").alias("best_p2"))
fn = gt.join(pred, on=["s1", "x"], how="anti")
fn = fn.join(cand, on=["s1", "x"], how="left").join(best_x, on="x", how="left").with_columns(
    reason=pl.when(pl.col("p2").is_null()).then(pl.lit("not_in_candidates"))
    .when(pl.col("best_s1") != pl.col("s1")).then(pl.lit("lost_to_other_s1"))
    .otherwise(pl.lit("below_threshold")))
w("## D2. False negatives by reason\n")
md(fn.group_by("reason").agg(n=pl.len(), mean_p2=pl.col("p2").mean()).with_columns(share=pl.col("n") / fn.height))
fnb = fn.filter(pl.col("reason") == "below_threshold")
w("Below-threshold FN, p2 histogram:\n")
md(fnb.with_columns(b=(pl.col("p2") * 10).floor() / 10).group_by("b").len().sort("b"))
fp = pred.join(gt, on=["s1", "x"], how="anti")
x_true = gt_all.select("x", pl.col("s1").alias("true_s1"))
fp = fp.join(x_true, on="x", how="left").join(ev.select("s1", "x", "p2"), on=["s1", "x"], how="left")
w("## D3. False positives\n")
md(fp.with_columns(kind=pl.when(pl.col("true_s1").is_null()).then(pl.lit("x_matches_no_s1(decoy)"))
                   .otherwise(pl.lit("x_belongs_to_other_s1"))).group_by("kind").agg(n=pl.len(), mean_p2=pl.col("p2").mean()))

# slices: loss by x attributes (FN + FP counts per slice relative to truth volume)
xa = xraw.with_columns(addr_empty=(pl.col("addr").str.strip_chars() == "") | pl.col("addr").str.to_lowercase().is_in(["null", "n/a", "none"]),
                       indic=pl.col("name").str.contains(r"[ऀ-෿]"),
                       url=pl.col("name").str.contains(r"\.(com|in|net|org|co|fr)\b")).select("x", "src", "addr_empty", "indic", "url")
g2 = gt.join(xa, on="x").join(pred.with_columns(hit=pl.lit(1)), on=["s1", "x"], how="left")
w("## D4. Recall by slice of the true X record\n")
for col in ("src", "addr_empty", "indic", "url"):
    md(g2.group_by(col).agg(n=pl.len(), recall=pl.col("hit").is_not_null().mean()).sort(col))
fpa = fp.join(xa, on="x")
w("FP by slice (count):\n")
md(fpa.group_by("addr_empty", "src").len().sort("len", descending=True))

# oracle ceilings
w("## D5. Ceilings on holdout\n")
in_c = gt.join(cand, on=["s1", "x"], how="semi")
orc = evaluate.per_entity_f05(in_c.select("s1", "x"), gt, es["s1"])["f05"].mean()
w(f"- perfect matcher on current candidates (blocking ceiling): macro F0.5 = {orc:.5f}")
w(f"- current: {r['f05'].mean():.5f}\n")

# ------------------------------------------------------------------ D6 test-time country comparison
w("## D6. Test vs holdout: score distributions per country (distribution-shift check)\n")
tp_ = pl.read_parquet(config.path("preds", "test_m_v1.parquet"))
tx = io.load("test", "x", ["idx", "country"]).rename({"idx": "x"})
tp_ = tp_.join(tx, on="x")
evc = ev.join(io.load("train", "x", ["idx", "country"]).rename({"idx": "x"}), on="x")


def dist(df, label):
    top = df.sort("p2", descending=True).unique("x", keep="first")
    return top.group_by("country").agg(
        n_x=pl.len(),
        top_p2_ge_0_8=(pl.col("p2") >= 0.8).mean(), top_p2_0_2_to_0_8=((pl.col("p2") >= 0.2) & (pl.col("p2") < 0.8)).mean(),
        top_p2_lt_0_2=(pl.col("p2") < 0.2).mean(), mean_top_p2=pl.col("p2").mean()).with_columns(set=pl.lit(label))


cpx = lambda df: df.group_by("country").agg(cands_per_x=pl.len() / pl.col("x").n_unique())  # noqa: E731
md(pl.concat([dist(evc, "holdout(EVAL_X)").join(cpx(evc), on="country"),
              dist(tp_, "test").join(cpx(tp_), on="country")]).sort("set", "country"))
w("Note: EVAL_X is biased toward records near EVAL_S1 entities, so compare shapes rather than absolute levels.\n")

# feature shift on top-1 pairs (test) per country
w("## D7. Feature means on each record's top-1 candidate (test), per country\n")
fcols = ["enc_score", "lsa_full_score", "lsa_name_score", "keys_score", "p0", "core_ratio", "core_tset",
         "ntok_wjacc", "ad_tset", "atok_wjacc", "num_first_eq", "state_eq", "s1_core_freq", "s1_ad_freq",
         "x_core_freq_in_s1", "ad_x_empty", "legal_eq", "x_is_url", "c_gap_x", "c_n_strong_s", "c_other_addr_sim"]
tf = matcher.load_feats("test").select(["x", "s1"] + [c for c in fcols if c not in ("c_gap_x", "c_n_strong_s", "c_other_addr_sim")])
tf = tf.join(tp_.select("x", "s1", "p2", "country"), on=["x", "s1"])
top = tf.sort("p2", descending=True).unique("x", keep="first")
agg = top.group_by("country").agg([pl.col(c).mean().alias(c) for c in tf.columns if c not in ("x", "s1", "country")])
md(agg)
del tf, top

# ------------------------------------------------------------------ D8 leave-one-country-out probe
w("## D8. Leave-one-country-out probe (proxy for unseen France)\n")
w("Stage-1 LightGBM (300 rounds), trained on TRAIN_X pairs of one country, AUC/logloss and best-F0.5 threshold on EVAL_X "
  "pairs of the other country, compared with in-country training.\n")
trf = matcher.add_labels(matcher.load_feats("train_x")).join(io.load("train", "x", ["idx", "country"]).rename({"idx": "x"}), on="x")
evf = matcher.add_labels(matcher.load_feats("eval_x")).join(io.load("train", "x", ["idx", "country"]).rename({"idx": "x"}), on="x")
base_cols = matcher.feat_cols(trf.drop("country"), 1)
retr = [c for c in base_cols if c.endswith("_score") or c.endswith("_rank") or c in ("p0", "r0")]
freq = ["s1_core_freq", "s1_ad_freq", "x_core_freq_in_s1"]
variants = {"all": base_cols, "no_retrieval_scores": [c for c in base_cols if c not in retr],
            "no_retrieval_no_freq": [c for c in base_cols if c not in retr + freq]}
P = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
         bagging_fraction=0.7, bagging_freq=1, verbose=-1, num_threads=14)
from sklearn.metrics import roc_auc_score, log_loss  # noqa: E402
rows = []
for vname, cols in variants.items():
    for tr_c in ("US", "India"):
        a = trf.filter(pl.col("country") == tr_c)
        m = lgb.train(P, lgb.Dataset(a.select(cols).to_numpy(), a["y"].to_numpy()), 300)
        for te_c in ("US", "India"):
            b = evf.filter(pl.col("country") == te_c)
            p = m.predict(b.select(cols).to_numpy(), num_threads=14)
            bb = b.select("x", "s1").with_columns(p2=pl.Series(p))
            e_s1 = es.join(s1meta.filter(pl.col("country") == te_c), on="s1", how="semi")
            best = max((evaluate.macro_f05(decode.decode_threshold(bb, t, "p2").join(e_s1, on="s1", how="semi"),
                                           gt.join(e_s1, on="s1", how="semi"), e_s1["s1"]), t) for t in (0.5, 0.7, 0.8, 0.9))
            f_at_08 = evaluate.macro_f05(decode.decode_threshold(bb, 0.8, "p2").join(e_s1, on="s1", how="semi"),
                                         gt.join(e_s1, on="s1", how="semi"), e_s1["s1"])
            rows.append({"features": vname, "train": tr_c, "test": te_c, "auc": roc_auc_score(b["y"], p),
                         "logloss": log_loss(b["y"], np.clip(p, 1e-6, 1 - 1e-6)),
                         "macroF_thr0.8": f_at_08, "best_macroF": best[0], "best_thr": best[1]})
            print(rows[-1], flush=True)
md(pl.DataFrame(rows))
w(f"_diagnostics runtime {time.time() - t0:.0f}s_")
OUT.write_text("\n".join(lines), encoding="utf-8")
print("written", OUT)
