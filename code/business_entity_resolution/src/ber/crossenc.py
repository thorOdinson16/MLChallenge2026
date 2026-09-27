"""Cross-encoder for the uncertain band (Apache-2.0 multilingual MiniLM, 118M params, fine-tuned here).

Reads an (X record, S1 record) text pair jointly and outputs a logit. It is used only where the matcher is unsure
(stage-1 probability p1 in [lo, hi], top-k candidates per X); its score feeds the stage-3 stacker (ber.stage3).
Inside that band the cross-encoder alone is weaker than v2 but the two are complementary: holdout-band AUC
p2 0.939 -> p2 + ce 0.955 (India 0.949 -> 0.972), and the stacker's holdout macro-F0.5 rises 0.98919 -> 0.99054.

The cross-encoder never sees encoder features, so unlike the matcher it can train on every non-holdout fold
(the matcher must avoid the folds the bi-encoder was trained on).  Holdout (fold 0) is never used for training.

    python -m ber.crossenc <tag> [lo hi k epochs train_folds]     e.g.  python -m ber.crossenc ce_v2 0.01 0.99 4 2 all
Outputs: models/<tag>.pt, preds/<tag>_holdout.parquet, preds/<tag>_test.parquet  (x, s1, ce)
"""
import os
import sys
import time

import numpy as np
import polars as pl
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from . import config, io, split

BASE = os.environ.get("BER_CE_BASE", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
MAXLEN, BS, LR = 128, int(os.environ.get("BER_CE_BS", 64)), float(os.environ.get("BER_CE_LR", 3e-5))
FRAC = float(os.environ.get("BER_CE_FRAC", 1.0))          # train on a random subset (pilots)
SKIP_TEST = os.environ.get("BER_CE_SKIP_TEST") == "1"
EXTRA = os.environ.get("BER_CE_EXTRA", "")                # optional extra TEST-split training rows (x, s1, src, country, p1, p2, y), e.g. pseudo-labels
DEV = "cuda"


def band(d: pl.DataFrame, lo: float, hi: float, k: int) -> pl.DataFrame:
    d = d.filter((pl.col("p1") >= lo) & (pl.col("p1") <= hi))
    return d.with_columns(r=pl.col("p1").rank("ordinal", descending=True).over("x")).filter(pl.col("r") <= k).drop("r")


def add_text(d: pl.DataFrame, split_name: str) -> pl.DataFrame:
    x = io.load(split_name, "x", ["idx", "name", "addr"]).rename({"idx": "x", "name": "xn", "addr": "xa"})
    s = io.load(split_name, "s1", ["idx", "name", "addr"]).rename({"idx": "s1", "name": "sn", "addr": "sa"})
    d = d.join(x, on="x", how="left").join(s, on="s1", how="left")
    return d.with_columns(ta=(pl.col("xn") + " ; " + pl.col("xa")).str.strip_chars(),
                          tb=(pl.col("sn") + " ; " + pl.col("sa")).str.strip_chars())


def run(tag: str, lo=0.02, hi=0.98, k=3, epochs=2, train_folds="match"):
    torch.manual_seed(0)
    np.random.seed(0)
    tr = pl.read_parquet(config.path("preds", "train_m_v2.parquet"), columns=["x", "s1", "src", "country", "p1", "p2"])
    te = pl.read_parquet(config.path("preds", "test_m_v2.parquet"), columns=["x", "s1", "src", "country", "p1", "p2"])
    gt = io.load("train", "gt").with_columns(y=pl.lit(1, pl.UInt8))
    hold = pl.DataFrame({"s1": split.holdout_s1()})
    if train_folds == "match":     # v1: folds the matcher trained on
        rows = tr.join(pl.DataFrame({"s1": split.s1_in(split.MATCH)}), on="s1", how="semi").join(split.clean_x(), on="x", how="semi")
    else:                          # all non-holdout folds
        rows = tr.join(hold, on="s1", how="anti")
    trn = band(rows, lo, hi, k).join(gt, on=["s1", "x"], how="left").with_columns(pl.col("y").fill_null(0))
    hh = band(tr.join(hold, on="s1", how="semi"), lo, hi, k).join(gt, on=["s1", "x"], how="left").with_columns(pl.col("y").fill_null(0))
    tt = band(te, lo, hi, k)
    if FRAC < 1.0:
        trn = trn.sample(fraction=FRAC, seed=0)
    trn, hh, tt = add_text(trn, "train"), add_text(hh, "train"), add_text(tt, "test")
    if EXTRA:
        ex = add_text(pl.read_parquet(EXTRA), "test").select(trn.columns)
        print(f"extra training rows {ex.height} (pos {ex['y'].mean():.3f}) from {EXTRA}", flush=True)
        trn = pl.concat([trn, ex.cast(trn.schema)])
    print(f"pairs train {trn.height} (pos {trn['y'].mean():.3f}) holdout {hh.height} test {tt.height}", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1).to(DEV)

    def batches(d, bs, shuffle):
        idx = np.random.permutation(d.height) if shuffle else np.arange(d.height)
        ta, tb = d["ta"].to_list(), d["tb"].to_list()
        for i in range(0, len(idx), bs):
            j = idx[i:i + bs]
            enc = tok([ta[q] for q in j], [tb[q] for q in j], padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt")
            yield j, {a: b.to(DEV) for a, b in enc.items()}

    @torch.inference_mode()
    def score(d, name):
        model.eval()
        out = np.zeros(d.height, dtype=np.float32)
        t0 = time.time()
        for n, (j, enc) in enumerate(batches(d, 256, False)):
            with torch.autocast("cuda", dtype=torch.float16):
                out[j] = model(**enc).logits.squeeze(-1).float().cpu().numpy()
            if n % 1000 == 0:
                print(f"  scoring {name}: {min((n + 1) * 256, d.height)}/{d.height} {time.time() - t0:.0f}s", flush=True)
        return out

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = epochs * ((trn.height + BS - 1) // BS)
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    scaler = torch.amp.GradScaler("cuda")
    y_tr = torch.tensor(trn["y"].to_numpy(), dtype=torch.float32)
    t0, step = time.time(), 0
    for ep in range(1, epochs + 1):
        model.train()
        run_loss = 0.0
        for j, enc in batches(trn, BS, True):
            with torch.autocast("cuda", dtype=torch.float16):
                logit = model(**enc).logits.squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit.float(), y_tr[j].to(DEV))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss = 0.98 * run_loss + 0.02 * loss.item() if step else loss.item()
            step += 1
            if step % 500 == 0:
                print(f"ep {ep} step {step}/{steps} loss {run_loss:.4f} {(time.time() - t0) / 60:.1f} min", flush=True)
        s_h = score(hh, "holdout")
        p = np.clip(hh["p2"].to_numpy(), 1e-6, 1 - 1e-6)
        l2, y = np.log(p / (1 - p)), hh["y"].to_numpy()
        print(f"[epoch {ep}] holdout-band AUC: ce {roc_auc_score(y, s_h):.4f} | p2 {roc_auc_score(y, l2):.4f} | p2+ce {roc_auc_score(y, l2 + s_h):.4f}", flush=True)
        torch.save(model.state_dict(), str(config.path("models", f"{tag}.pt")))
    hh.select("x", "s1").with_columns(ce=pl.Series(s_h)).write_parquet(config.path("preds", f"{tag}_holdout.parquet"))
    if not SKIP_TEST:
        tt.select("x", "s1").with_columns(ce=pl.Series(score(tt, "test"))).write_parquet(config.path("preds", f"{tag}_test.parquet"))
    print(f"DONE {tag} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    run(a[0] if a else "ce_v2", float(a[1]) if len(a) > 1 else 0.02, float(a[2]) if len(a) > 2 else 0.98,
        int(a[3]) if len(a) > 3 else 3, int(a[4]) if len(a) > 4 else 2, a[5] if len(a) > 5 else "match")
