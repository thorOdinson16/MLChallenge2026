"""Cross-encoder re-ranker (joint S1/X attention) used as an extra stage-2 feature.

Leakage-free by construction: trained ONLY on pairs whose S1 is in the ENC folds (1-5). The matcher trains on
MATCH-fold pairs, and holdout/test are never seen, so the cross-encoder score looks the same everywhere.
Scored only in the uncertain band 0.01 <= p1 <= 0.99 (~10% of pairs); elsewhere the feature is -1.

  python -m ber.cross train | score
"""
import math
import sys
import time

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer

from . import config, io, normalize, split
from .blocking import encoder

TAG = "ce_v1"
MAX_LEN = 128
BAND = (0.01, 0.99)
DEVICE = encoder.DEVICE


class Cross(nn.Module):
    def __init__(self):
        super().__init__()
        enc = encoder.Encoder()
        ck = torch.load(config.path("models", "enc_v2.pt"), map_location="cpu")
        enc.load_state_dict(ck["state"])          # start from the domain-tuned bi-encoder
        self.bert = enc.bert
        self.head = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, ids, mask, tt=None):
        h = self.bert(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).to(h.dtype)
        return self.head((h * m).sum(1) / m.sum(1).clamp(min=1)).squeeze(-1)


def _texts(split_name: str, pairs: pl.DataFrame):
    s1 = normalize.load(split_name, "s1", ["idx", "nm", "ad", "ad_state"])
    x = normalize.load(split_name, "x", ["idx", "nm", "ad", "ad_state"])
    s1 = s1.join(pairs.select(pl.col("s1").alias("idx")).unique(), on="idx", how="semi")
    x = x.join(pairs.select(pl.col("x").alias("idx")).unique(), on="idx", how="semi")
    s1t = s1.select(pl.col("idx").alias("s1"), ta=pl.Series(encoder.record_text(s1)))
    xt = x.select(pl.col("idx").alias("x"), tb=pl.Series(encoder.record_text(x)))
    return pairs.join(s1t, on="s1", how="left", maintain_order="left").join(xt, on="x", how="left",
                                                                              maintain_order="left")


def train(n_band=1_300_000, n_easy=300_000, bs=128, lr=3e-5):
    torch.cuda.set_per_process_memory_fraction(0.9)
    p = pl.read_parquet(config.path("preds", "train_m_v2.parquet"), columns=["x", "s1", "p1"])
    p = p.join(pl.DataFrame({"s1": split.s1_in(split.ENC)}), on="s1", how="semi")
    gt = io.load("train", "gt").with_columns(y=pl.lit(1, pl.UInt8))
    p = p.join(gt, on=["s1", "x"], how="left").with_columns(pl.col("y").fill_null(0))
    band = p.filter(pl.col("p1").is_between(0.003, 0.997))
    easy = p.filter(~pl.col("p1").is_between(0.003, 0.997))
    d = pl.concat([band.sample(min(n_band, band.height), seed=1), easy.sample(min(n_easy, easy.height), seed=2)])
    d = _texts("train", d.sample(fraction=1.0, shuffle=True, seed=3))
    print(f"CE train pairs {d.height}, pos {d['y'].mean():.3f}", flush=True)
    ta, tb, y = d["ta"].to_list(), d["tb"].to_list(), d["y"].to_numpy().astype(np.float32)
    tok = AutoTokenizer.from_pretrained(encoder.BASE)
    model = Cross().to(DEVICE)
    model.bert.embeddings.word_embeddings.weight.requires_grad_(False)
    model.bert.gradient_checkpointing_enable()
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=lr, weight_decay=0.01)
    total = len(y) // bs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / 300) * max(0.0, 0.5 * (1 + math.cos(math.pi * s / total))))
    scaler = torch.amp.GradScaler()
    model.train()
    t0 = time.time()
    for step in range(total):
        sl = slice(step * bs, (step + 1) * bs)
        e = tok(ta[sl], tb[sl], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        with torch.autocast("cuda", dtype=torch.float16):
            logit = model(e["input_ids"].to(DEVICE), e["attention_mask"].to(DEVICE))
        loss = F.binary_cross_entropy_with_logits(logit.float(), torch.from_numpy(y[sl]).to(DEVICE))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if (step + 1) % 500 == 0:
            print(f"step {step + 1}/{total} loss {loss.item():.4f} {(time.time() - t0) / (step + 1):.3f}s/it", flush=True)
    torch.save(model.state_dict(), config.path("models", f"{TAG}.pt"))


@torch.inference_mode()
def score(split_name: str, bs: int = 512):
    p = pl.read_parquet(config.path("preds", f"{split_name}_m_v2.parquet"), columns=["x", "s1", "p1"])
    p = p.filter(pl.col("p1").is_between(*BAND)).select("x", "s1")
    d = _texts(split_name, p)
    tok = AutoTokenizer.from_pretrained(encoder.BASE)
    model = Cross()
    model.load_state_dict(torch.load(config.path("models", f"{TAG}.pt"), map_location="cpu"))
    model = model.to(DEVICE).eval().half()
    ta, tb = d["ta"].to_list(), d["tb"].to_list()
    order = np.argsort([len(a) + len(b) for a, b in zip(ta, tb)])
    out = np.empty(len(ta), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(order), bs):
        ix = order[i:i + bs]
        e = tok([ta[j] for j in ix], [tb[j] for j in ix], padding=True, truncation=True, max_length=MAX_LEN,
                return_tensors="pt")
        out[ix] = model(e["input_ids"].to(DEVICE), e["attention_mask"].to(DEVICE)).float().cpu().numpy()
        if (i // bs) % 500 == 0:
            print(f"  scored {i}/{len(order)} {time.time() - t0:.0f}s", flush=True)
    d.select("x", "s1").with_columns(ce=pl.Series(out)).write_parquet(config.path("preds", f"ce_{split_name}.parquet"))
    print(f"scored {split_name}: {len(out)} pairs in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    for s in sys.argv[1:]:
        if s == "train":
            train()
        elif s == "score":
            score("train")
            score("test")
