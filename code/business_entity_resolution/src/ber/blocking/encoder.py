"""Fine-tuned multilingual bi-encoder for candidate retrieval.

Base: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0, 118M params).
Training: contrastive (InfoNCE / multiple-negatives ranking) on (S1, X) ground-truth pairs; every batch
is drawn from a single country so in-batch negatives are realistic. Text = normalised name ; address.
"""
import math
import time

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer

from .. import config, io, normalize

BASE = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MAX_LEN = 64
DIM = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def record_text(d: pl.DataFrame) -> list:
    return (d["nm"] + " ; " + d["ad"] + " " + d["ad_state"]).str.strip_chars().to_list()


class Encoder(nn.Module):
    def __init__(self, base=BASE, n_layers=None):
        super().__init__()
        self.bert = AutoModel.from_pretrained(base)
        if n_layers:
            self.bert.encoder.layer = self.bert.encoder.layer[:n_layers]
        self.proj = nn.Linear(self.bert.config.hidden_size, DIM)

    def forward(self, ids, mask):
        h = self.bert(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return F.normalize(self.proj(pooled), dim=-1)


def _batches(pairs: pl.DataFrame, bs: int, seed: int):
    """Index arrays for one epoch. Hard-negative batching: pairs are sorted by a locality key
    (same city/state, or same first name word) so in-batch negatives are confusable businesses.
    One X per S1 per batch-run, so no in-batch false negatives."""
    rng = np.random.default_rng(seed)
    g = pairs.sample(fraction=1.0, shuffle=True, seed=int(rng.integers(1 << 30)))
    g = g.unique("s1", keep="first", maintain_order=True)
    g = g.with_columns(r=pl.Series(rng.random(g.height)), use_name=pl.Series(rng.random(g.height) < 0.4))
    g = g.with_columns(key=pl.when(pl.col("use_name")).then("n:" + pl.col("k_name")).otherwise("a:" + pl.col("k_addr")))
    g = g.sort(["country", "key", "r"])
    idx = g["row"].to_numpy()
    out = [idx[i:i + bs] for i in range(0, len(idx) - bs + 1, bs)]
    rng.shuffle(out)
    return out


def train(n_pairs=1_600_000, epochs=1, bs=192, lr=5e-5, n_layers=None, tag="enc_v1", exclude_s1=None):
    tok = AutoTokenizer.from_pretrained(BASE)
    model = Encoder(n_layers=n_layers).to(DEVICE)
    s1 = normalize.load("train", "s1", ["idx", "country", "nm", "ad", "ad_state"])
    x = normalize.load("train", "x", ["idx", "nm", "ad", "ad_state"])
    gt = io.load("train", "gt")
    if exclude_s1 is not None:
        gt = gt.join(pl.DataFrame({"s1": exclude_s1}).cast(pl.UInt32), on="s1", how="anti")
    gt = gt.sample(min(n_pairs, gt.height), seed=config.SEED)
    p = (gt.join(s1.rename({"idx": "s1"}), on="s1")
           .join(x.rename({"idx": "x", "nm": "nm_x", "ad": "ad_x", "ad_state": "ad_state_x"}), on="x"))
    t1 = record_text(p)
    t2 = record_text(p.select(nm="nm_x", ad="ad_x", ad_state="ad_state_x"))
    p = p.with_columns(
        k_addr=pl.col("ad_state") + " " + pl.col("ad").str.split(" ").list.last().fill_null(""),
        k_name=pl.col("nm").str.split(" ").list.first().fill_null(""),
    ).select("s1", "country", "k_addr", "k_name").with_row_index("row")
    del s1, x
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps_per_epoch = len(_batches(p, bs, 0))
    total = steps_per_epoch * epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / 500) * max(0.0, 0.5 * (1 + math.cos(math.pi * s / total))))
    scaler = torch.amp.GradScaler()
    scale = nn.Parameter(torch.tensor(math.log(20.0), device=DEVICE))
    opt.add_param_group({"params": [scale], "lr": 1e-3, "weight_decay": 0})
    step, t0 = 0, time.time()
    model.train()
    for ep in range(epochs):
        for b in _batches(p, bs, ep):
            ea = tok([t1[i] for i in b], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
            eb = tok([t2[i] for i in b], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
            with torch.autocast("cuda", dtype=torch.float16):
                za = model(ea["input_ids"].to(DEVICE), ea["attention_mask"].to(DEVICE))
                zb = model(eb["input_ids"].to(DEVICE), eb["attention_mask"].to(DEVICE))
            logits = (za.float() @ zb.float().T) * scale.exp().clamp(max=100)
            lbl = torch.arange(len(b), device=DEVICE)
            loss = (F.cross_entropy(logits, lbl) + F.cross_entropy(logits.T, lbl)) / 2
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 200 == 0:
                print(f"step {step}/{total} loss {loss.item():.4f} scale {scale.exp().item():.1f} "
                      f"{(time.time() - t0) / step:.3f}s/it", flush=True)
    torch.save({"state": model.state_dict(), "n_layers": n_layers}, config.path("models", f"{tag}.pt"))
    return model


def load(tag="enc_v1") -> Encoder:
    ck = torch.load(config.path("models", f"{tag}.pt"), map_location="cpu")
    m = Encoder(n_layers=ck["n_layers"])
    m.load_state_dict(ck["state"])
    return m.to(DEVICE).eval().half()


@torch.inference_mode()
def encode(model: Encoder, texts: list, bs: int = 1024) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained(BASE)
    order = np.argsort([len(t) for t in texts])         # length bucketing -> less padding
    out = np.empty((len(texts), DIM), dtype=np.float16)
    for i in range(0, len(texts), bs):
        ix = order[i:i + bs]
        e = tok([texts[j] for j in ix], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        z = model(e["input_ids"].to(DEVICE), e["attention_mask"].to(DEVICE))
        out[ix] = z.cpu().numpy().astype(np.float16)
    return out


if __name__ == "__main__":
    import sys
    from .. import split
    tag = sys.argv[1] if len(sys.argv) > 1 else "enc_v2"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    # train only on ENC-fold entities; MATCH folds stay unseen so downstream models see test-like scores
    train(tag=tag, epochs=epochs, n_pairs=10_000_000, exclude_s1=split.s1_in(split.HOLDOUT | split.MATCH))
    m = load(tag)
    texts = record_text(normalize.load("test", "x", ["nm", "ad", "ad_state"]).head(100_000))
    t = time.time()
    encode(m, texts)
    print(f"encode throughput: {len(texts) / (time.time() - t):.0f} texts/s", flush=True)
