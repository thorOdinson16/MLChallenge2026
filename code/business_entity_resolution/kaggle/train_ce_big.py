"""Bigger cross-encoder for the AnchorMatch uncertain band — runs unchanged on Kaggle (2x T4) or an AWS GPU instance.

Inputs (same folder or --data): ce_train.parquet, ce_holdout.parquet, ce_test.parquet  [+ optional ce_fr_pseudo.parquet]
   columns: x, s1, country, p2, y, ta (X text), tb (S1 text)
Outputs (--out): ce_big_holdout.parquet, ce_big_test.parquet  (x, s1, ce)   + model_big.pt, log lines on stdout

    python train_ce_big.py --data . --out . --model xlm-roberta-large --hours 6
Only the provided challenge data is used; the base model (MIT) is downloaded once from the Hugging Face hub.
"""
import argparse, os, time
import numpy as np, polars as pl, torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=".")
ap.add_argument("--out", default=".")
ap.add_argument("--model", default="xlm-roberta-large")
ap.add_argument("--hours", type=float, default=6.0, help="training time budget; scoring runs after it")
ap.add_argument("--epochs", type=float, default=2.0)
ap.add_argument("--bs", type=int, default=32, help="per-step batch (split across GPUs)")
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--maxlen", type=int, default=128)
ap.add_argument("--pseudo", default="", help="optional extra training parquet (e.g. France pseudo-labels)")
ap.add_argument("--seed", type=int, default=0, help="random seed (weights init of the head, data order, subsample)")
ap.add_argument("--limit", type=int, default=0, help="smoke test: use only the first N rows of each file")
a = ap.parse_args()
t_start = time.time()
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = "cuda"
ngpu = torch.cuda.device_count()
print(f"GPUs: {ngpu} x {torch.cuda.get_device_name(0)}", flush=True)
bf16 = torch.cuda.is_bf16_supported()
dtype = torch.bfloat16 if bf16 else torch.float16
rd = lambda n: pl.read_parquet(os.path.join(a.data, n))
trn, hh, tt = rd("ce_train.parquet"), rd("ce_holdout.parquet"), rd("ce_test.parquet")
if a.limit:                                     # smoke test
    trn, hh, tt = trn.head(a.limit), hh.head(a.limit), tt.head(a.limit)
if a.pseudo:
    ps = pl.read_parquet(a.pseudo).select(trn.columns)
    trn = pl.concat([trn, ps])
    print(f"added {ps.height} pseudo-labelled rows", flush=True)
print(f"train {trn.height}  holdout {hh.height}  test {tt.height}", flush=True)

tok = AutoTokenizer.from_pretrained(a.model)
base = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=1)
class Wrap(torch.nn.Module):          # returns a plain tensor so DataParallel can gather it
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, **kw): return self.m(**kw).logits.squeeze(-1)
model = Wrap(base).to(dev)
if ngpu > 1:
    model = torch.nn.DataParallel(model)

def encode(A, B):
    e = tok(A, B, padding=True, truncation=True, max_length=a.maxlen, return_tensors="pt")
    return {k: v.to(dev) for k, v in e.items()}

@torch.inference_mode()
def score(d, name):
    model.eval()
    A, B = d["ta"].to_list(), d["tb"].to_list()
    order = np.argsort([len(x) + len(y) for x, y in zip(A, B)])      # length-sorted batches: much faster
    out = np.zeros(d.height, dtype=np.float32); t0 = time.time(); bs = 256 * max(ngpu, 1)
    for n, st in enumerate(range(0, len(order), bs)):
        j = order[st:st + bs]
        with torch.autocast("cuda", dtype=dtype):
            out[j] = model(**encode([A[q] for q in j], [B[q] for q in j])).float().cpu().numpy()
        if n % 200 == 0:
            print(f"  scoring {name}: {min(st + bs, len(order))}/{len(order)}  {time.time() - t0:.0f}s", flush=True)
    return out

def report(s, tag):
    p = np.clip(hh["p2"].to_numpy(), 1e-6, 1 - 1e-6); l2 = np.log(p / (1 - p)); y = hh["y"].to_numpy()
    print(f"[{tag}] holdout-band AUC: ce {roc_auc_score(y, s):.4f} | p2 {roc_auc_score(y, l2):.4f} | p2+ce {roc_auc_score(y, l2 + s):.4f}"
          f"   (MiniLM reference: ce 0.9452, p2+ce 0.9724)", flush=True)

opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
steps_per_epoch = (trn.height + a.bs - 1) // a.bs
total = int(a.epochs * steps_per_epoch)
sch = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
scaler = torch.amp.GradScaler("cuda", enabled=not bf16)
A, B, Y = trn["ta"].to_list(), trn["tb"].to_list(), torch.tensor(trn["y"].to_numpy(), dtype=torch.float32)
budget = a.hours * 3600
step, stop, ep = 0, False, 0
print(f"training: {total} steps planned, budget {a.hours} h", flush=True)
while not stop and step < total:
    ep += 1
    model.train(); idx = np.random.permutation(trn.height); run = None
    for st in range(0, len(idx), a.bs):
        j = idx[st:st + a.bs]
        with torch.autocast("cuda", dtype=dtype):
            logit = model(**encode([A[q] for q in j], [B[q] for q in j]))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit.float(), Y[j].to(dev))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sch.step(); step += 1
        run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
        if step % 500 == 0:
            el = time.time() - t_start
            print(f"ep {ep} step {step}/{total} loss {run:.4f}  {el / 60:.1f} min  (eta train {el / step * (total - step) / 60:.0f} min)", flush=True)
        if time.time() - t_start > budget or step >= total:
            stop = time.time() - t_start > budget
            break
    print(f"epoch {ep} done at step {step} ({(time.time() - t_start) / 60:.1f} min){' - time budget reached' if stop else ''}", flush=True)
torch.save((model.module if ngpu > 1 else model).m.state_dict(), os.path.join(a.out, "model_big.pt"))
s_h = score(hh, "holdout"); report(s_h, "final")
hh.select("x", "s1").with_columns(ce=pl.Series(s_h)).write_parquet(os.path.join(a.out, "ce_big_holdout.parquet"))
s_t = score(tt, "test")
tt.select("x", "s1").with_columns(ce=pl.Series(s_t)).write_parquet(os.path.join(a.out, "ce_big_test.parquet"))
print(f"DONE in {(time.time() - t_start) / 3600:.2f} h -> ce_big_holdout.parquet, ce_big_test.parquet", flush=True)
