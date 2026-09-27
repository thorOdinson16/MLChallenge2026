"""Unseen-country proxy, cross-encoder part (India plays France; matcher = US-only v2, preds/train_m_v2us.parquet).

Trains two MiniLM cross-encoders on the band of the US-only matcher:
  prx_us : US band rows only (India never seen)                       -> like ce_v2 on France
  prx_ps : US band rows + India pseudo-labels from the US-only matcher -> like the France-adapted CEs
and scores the holdout band.  Output: preds/prx_{us,ps}_holdout.parquet (x, s1, ce).

    python -m ber.experiments.trust_proxy_ce
"""
import sys
import time

import numpy as np
import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .. import config, io, split
from ..crossenc import add_text, band

BASE = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
HI, LO = 0.9, 0.05


def train_score(trn, hh, tag):
    torch.manual_seed(0); np.random.seed(0)
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1).to("cuda")
    ta, tb, yy = trn["ta"].to_list(), trn["tb"].to_list(), torch.tensor(trn["y"].to_numpy(), dtype=torch.float32)
    bs, epochs = 64, 2
    steps = epochs * ((len(ta) + bs - 1) // bs)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    scaler = torch.amp.GradScaler("cuda")
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        idx = np.random.permutation(len(ta))
        for st in range(0, len(idx), bs):
            j = idx[st:st + bs]
            enc = tok([ta[q] for q in j], [tb[q] for q in j], padding=True, truncation=True, max_length=128, return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                lo = model(**enc).logits.squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(lo.float(), yy[j].to("cuda"))
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sch.step()
        print(f"  {tag} epoch {ep + 1} done {time.time() - t0:.0f}s", flush=True)
    model.eval()
    out = np.zeros(hh.height, dtype=np.float32)
    A, B = hh["ta"].to_list(), hh["tb"].to_list()
    with torch.inference_mode():
        for st in range(0, hh.height, 512):
            enc = tok(A[st:st + 512], B[st:st + 512], padding=True, truncation=True, max_length=128, return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                out[st:st + 512] = model(**enc).logits.squeeze(-1).float().cpu().numpy()
    hh.select("x", "s1").with_columns(ce=pl.Series(out)).write_parquet(config.path("preds", f"{tag}_holdout.parquet"))
    del model; torch.cuda.empty_cache()


def main():
    tr = pl.read_parquet(config.path("preds", "train_m_v2us.parquet"), columns=["x", "s1", "src", "country", "p1", "p2"])
    gt = io.load("train", "gt").with_columns(y=pl.lit(1, pl.UInt8))
    hold = pl.DataFrame({"s1": split.holdout_s1()})
    b = band(tr.join(hold, on="s1", how="anti"), 0.01, 0.99, 4).join(gt, on=["s1", "x"], how="left").with_columns(pl.col("y").fill_null(0))
    us = b.filter(pl.col("country") == "US").sample(400_000, seed=0)
    ind = b.filter(pl.col("country") == "India")
    ps = ind.filter((pl.col("p2") >= HI) | (pl.col("p2") <= LO)).with_columns(y=(pl.col("p2") >= HI).cast(pl.UInt8)).sample(250_000, seed=1)
    hh = band(tr.join(hold, on="s1", how="semi"), 0.01, 0.99, 4)
    us, ps, hh = add_text(us, "train"), add_text(ps, "train"), add_text(hh, "train")
    print(f"US rows {us.height}, India pseudo rows {ps.height}, holdout band {hh.height}", flush=True)
    train_score(us, hh, "prx_us")
    train_score(pl.concat([us.select("ta", "tb", "y"), ps.select("ta", "tb", "y")]).sample(fraction=1.0, seed=3), hh, "prx_ps")


if __name__ == "__main__":
    sys.exit(main())
