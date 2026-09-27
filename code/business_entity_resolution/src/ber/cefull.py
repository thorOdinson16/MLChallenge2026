"""ce_full: full-coverage cross-encoder (plan v4, T1).

Every earlier cross-encoder re-reads only the uncertain band (p1 in [0.01, 0.99], top-4 per X). Pairs the US/India
GBDT is confident about (p1 > 0.99 or < 0.01) were never re-read by a text model, so confident decoy false positives
and France's confident errors survive. ce_full is trained on band + confident pairs + test pseudo-labels (France over
the whole graph, US/India for the thinned test pool), and scores every candidate pair (top-K by p1 per X).

Training rows never include an X that touches a holdout (fold-0) S1, so holdout scores are out-of-sample for stage 3.
Recipe = ce_v2 / crossenc (raw 'name ; addr' text, MiniLM base, sequence-classification head, BCE, fp16).

    python -m ber.cefull train [tag]           -> models/<tag>.pt
    python -m ber.cefull score [tag]           -> preds/<tag>_holdout.parquet, preds/<tag>_test.parquet  (x, s1, ce)
Outputs are plug-compatible with ber.experiments.overnight (use <tag> in its CE list).
"""
import os
import sys
import time

import numpy as np
import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from . import config, io, split
from .crossenc import add_text, band

BASE = os.environ.get("BER_CE_BASE", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
MAXLEN, BS, LR = 128, int(os.environ.get("BER_CE_BS", 64)), float(os.environ.get("BER_CE_LR", 3e-5))
EPOCHS = int(os.environ.get("BER_CE_EPOCHS", 1))
TOPK_SCORE = int(os.environ.get("BER_CE_TOPK", 6))
CORRECTED = os.environ.get("BER_CE_CORRECTED") == "1"
# ce_full2 is used for France only (per-country final); override with BER_CE_SCORE_COUNTRY="" to score all countries
SCORE_COUNTRY = os.environ.get("BER_CE_SCORE_COUNTRY", "France" if CORRECTED else "")
# round-4 mode: train on train band (US/IN labels) + a friend label file (France), score the friend's band pair set
R4 = os.environ.get("BER_CE_R4", "")                      # e.g. fr_pseudo4.parquet
R4_BAND = int(os.environ.get("BER_CE_R4_BAND", 300_000))
SCORE_SET = os.environ.get("BER_CE_SCORE_SET", "")        # "band": score exactly the (x, s1) of preds/ce_v4_{holdout,test}
SEED = int(os.environ.get("BER_CE_SEED", 0))
BUDGET_MIN = float(os.environ.get("BER_CE_BUDGET_MIN", 0))   # >0: shrink the LR schedule to finish within this many minutes
DEV = "cuda"
P = config.path
COLS = ["x", "s1", "src", "country", "p1", "p2", "y", "w"]


def _rank(d):
    return d.with_columns(r=pl.col("p1").rank("ordinal", descending=True).over("x"))


def _label_file(name):
    """Pseudo-label file (x, s1, y [+ src, country, p1, p2]); missing columns are taken from test_m_v2."""
    d = pl.read_parquet(P("preds", name))
    need = [c for c in ("src", "country", "p1", "p2") if c not in d.columns]
    if need:
        te = pl.read_parquet(P("preds", "test_m_v2.parquet"), columns=["x", "s1"] + need)
        d = d.join(te, on=["x", "s1"], how="left")
    return d.select("x", "s1", pl.col("src").cast(pl.Float32), "country", pl.col("p1").cast(pl.Float32),
                    pl.col("p2").cast(pl.Float32), pl.col("y").cast(pl.UInt8)).drop_nulls(["country"])


def r4_rows(seed=0):
    gt = io.load("train", "gt").with_columns(y=pl.lit(1, pl.UInt8))
    tr = pl.read_parquet(P("preds", "train_m_v2.parquet"), columns=["x", "s1", "src", "country", "p1", "p2"])
    hold = pl.DataFrame({"s1": split.holdout_s1()})
    tr = tr.join(tr.join(hold, on="s1", how="semi").select("x").unique(), on="x", how="anti")
    b = band(tr, 0.01, 0.99, 4).join(gt, on=["s1", "x"], how="left").with_columns(pl.col("y").fill_null(0).cast(pl.UInt8))
    parts = {"band": b.sample(n=min(R4_BAND, b.height), seed=seed + 11, shuffle=True).with_columns(w=pl.lit("train")),
             "r4": _label_file(R4).with_columns(w=pl.lit("test"))}
    parts = {k: v.select(COLS) for k, v in parts.items()}
    for k, v in parts.items():
        print(f"  {k:12s} {v.height:9d}  pos {v['y'].mean():.3f}", flush=True)
    return parts


def training_rows(seed=0):
    if R4:
        return r4_rows(seed)
    rng = seed
    gt = io.load("train", "gt").with_columns(y=pl.lit(1, pl.UInt8))
    tr = pl.read_parquet(P("preds", "train_m_v2.parquet"), columns=["x", "s1", "src", "country", "p1", "p2"])
    hold = pl.DataFrame({"s1": split.holdout_s1()})
    ev_x = tr.join(hold, on="s1", how="semi").select("x").unique()
    tr = tr.join(ev_x, on="x", how="anti")                           # X-disjoint from every holdout-touching X
    tr = _rank(tr).join(gt, on=["s1", "x"], how="left").with_columns(pl.col("y").fill_null(0).cast(pl.UInt8))
    parts = {
        "band": band(tr.drop("r"), 0.01, 0.99, 4).join(gt, on=["s1", "x"], how="left").with_columns(
            pl.col("y").fill_null(0).cast(pl.UInt8)).sample(n=700_000, seed=rng, shuffle=True),
        "conf_hi_neg": tr.filter((pl.col("p1") > 0.99) & (pl.col("y") == 0)),       # confident decoys/other-S1: the veto
        "conf_hi_pos": tr.filter((pl.col("p1") > 0.99) & (pl.col("y") == 1)).sample(n=250_000, seed=rng + 1),
        "conf_lo": tr.filter((pl.col("p1") < 0.01) & (pl.col("r") <= 2)).sample(n=150_000, seed=rng + 2),
    }
    parts = {k: v.select(["x", "s1", "src", "country", "p1", "p2", "y"]).with_columns(w=pl.lit("train")) for k, v in parts.items()}
    # test pseudo-labels from the best stage 3 (sub_K): France over the WHOLE graph, US/India for the thinned pool
    te = pl.read_parquet(P("preds", "test_m_v2.parquet"), columns=["x", "s1", "src", "country", "p1", "p2"])
    p3 = pl.read_parquet(P("preds", "test_s3_K.parquet"), columns=["x", "s1", "p3"])
    te = te.join(p3, on=["x", "s1"], how="left").with_columns(pl.col("p3").fill_null(0.0))
    conf = te.filter((pl.col("p3") >= 0.97) | (pl.col("p3") <= 0.02)).with_columns(y=(pl.col("p3") >= 0.97).cast(pl.UInt8))
    if CORRECTED:
        # LB-validated France rescues (noise words, category swaps, dropped word) are matches: flip them to y=1 and
        # drop any other confident positive of the same X (an X matches at most one S1)
        x_ids = io.load("test", "x", ["idx", "eid"]).rename({"idx": "x", "eid": "xe"})
        s_ids = io.load("test", "s1", ["idx", "eid"]).rename({"idx": "s1", "eid": "source1_entity_id"})
        resc = pl.concat([pl.read_parquet(P("preds", f)) for f in ("cn_rescue_pairs.parquet",
                                                                   )]).unique()   # LB: cn +0.00054; nz_swap+del_only -0.00088 (dropped)
        resc = resc.join(x_ids, on="xe").join(s_ids, on="source1_entity_id").select("x", "s1")
        resc = resc.join(te.select("x", "s1", "src", "country", "p1", "p2"), on=["x", "s1"]).with_columns(
            p3=pl.lit(1.0), y=pl.lit(1, pl.UInt8))
        conf = pl.concat([conf.join(resc.select("x"), on="x", how="anti"), resc.select(conf.columns).cast(conf.schema)])
        print(f"  corrected: {resc.height} rescued France pairs set to y=1", flush=True)
    fr = conf.filter(pl.col("country") == "France")
    fr_hard = fr.filter(((pl.col("p1") > 0.99) & (pl.col("y") == 0)) | ((pl.col("p1") < 0.5) & (pl.col("y") == 1)))   # s3 overruled v2
    fr_rest = fr.join(fr_hard.select("x", "s1"), on=["x", "s1"], how="anti")
    parts["fr_hard"] = fr_hard
    parts["fr_pos"] = fr_rest.filter(pl.col("y") == 1).sample(n=min(250_000, fr_rest.filter(pl.col("y") == 1).height), seed=rng + 3)
    parts["fr_neg"] = fr_rest.filter(pl.col("y") == 0).sample(n=min(150_000, fr_rest.filter(pl.col("y") == 0).height), seed=rng + 4)
    ui = conf.filter(pl.col("country") != "France")
    parts["ui_pseudo"] = ui.sample(n=min(200_000, ui.height), seed=rng + 5)
    if CORRECTED:   # the friend's corrected band labels (rescued pairs = 1)
        fc = pl.read_parquet(P("preds", "fr_pseudo_corrected.parquet")).select("x", "s1", "src", "country", "p1", "p2", "y")
        parts["fr_band_corr"] = fc.cast({"y": pl.UInt8})
    for k in ("fr_hard", "fr_pos", "fr_neg", "ui_pseudo", "fr_band_corr"):
        if k not in parts:
            continue
        parts[k] = parts[k].select(["x", "s1", "src", "country", "p1", "p2", "y"]).with_columns(w=pl.lit("test"))
    for k, v in parts.items():
        print(f"  {k:12s} {v.height:9d}  pos {v['y'].mean():.3f}", flush=True)
    return parts


def _prefetch(gen, depth=6):
    """Run a batch generator in a background thread so CPU tokenisation overlaps GPU compute."""
    import queue
    import threading
    q = queue.Queue(maxsize=depth)
    END = object()

    def work():
        for item in gen:
            q.put(item)
        q.put(END)
    threading.Thread(target=work, daemon=True).start()
    while True:
        item = q.get()
        if item is END:
            return
        yield item


def _texts(d, split_name):
    return add_text(d.drop("w", strict=False), split_name)


def train(tag="ce_full"):
    torch.manual_seed(SEED); np.random.seed(SEED)
    torch.cuda.set_per_process_memory_fraction(0.9)
    parts = training_rows(SEED)
    trd = [_texts(v, "train") for k, v in parts.items() if v["w"][0] == "train"]
    ted = [_texts(v, "test") for k, v in parts.items() if v["w"][0] == "test"]
    d = pl.concat([t.select("x", "s1", "ta", "tb", "y") for t in trd] + [t.select("x", "s1", "ta", "tb", "y") for t in ted[::-1]])
    d = d.unique(["x", "s1", "ta", "tb"], keep="first").sample(fraction=1.0, shuffle=True, seed=7 + SEED)
    ta, tb, y = d["ta"].to_list(), d["tb"].to_list(), torch.tensor(d["y"].to_numpy(), dtype=torch.float32)
    print(f"training pairs {len(ta)} pos {float(y.mean()):.3f}", flush=True)
    del d, trd, ted, parts
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * ((len(ta) + BS - 1) // BS)
    total = [steps]                                   # mutable: BUDGET_MIN may shrink it after timing 300 steps
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: s / warm if s < warm else max(0.0, (total[0] - s) / max(1, total[0] - warm)))
    scaler = torch.amp.GradScaler("cuda")
    t0, step, run = time.time(), 0, 0.0
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(len(ta))

        def gen():
            for i in range(0, len(perm), BS):
                j = perm[i:i + BS]
                yield j, tok([ta[q] for q in j], [tb[q] for q in j], padding=True, truncation=True, max_length=MAXLEN,
                             return_tensors="pt")
        for j, enc in _prefetch(gen()):
            enc = enc.to(DEV)
            with torch.autocast("cuda", dtype=torch.float16):
                logit = model(**enc).logits.squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit.float(), y[j].to(DEV))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
            step += 1
            if BUDGET_MIN and step == 300:
                fit = int(BUDGET_MIN * 60 / ((time.time() - t0) / 300))
                if fit < total[0]:
                    total[0] = max(fit, warm + 1)
                    print(f"budget: schedule shrunk to {total[0]} of {steps} steps", flush=True)
            if step >= total[0]:
                break
            if step % 1000 == 0:
                print(f"step {step}/{total[0]} loss {run:.4f} {(time.time() - t0) / 60:.1f} min", flush=True)
        torch.save(model.state_dict(), str(P("models", f"{tag}.pt")))
    print(f"TRAINED {tag} in {(time.time() - t0) / 60:.1f} min", flush=True)


@torch.inference_mode()
def _score_frame(model, tok, d, split_name, chunk=400_000, bs=512):
    out = []
    t0 = time.time()
    for i in range(0, d.height, chunk):
        c = _texts(d.slice(i, chunk), split_name)
        ta, tb = c["ta"].to_list(), c["tb"].to_list()
        order = np.argsort([len(a) + len(b) for a, b in zip(ta, tb)])
        s = np.empty(len(ta), dtype=np.float32)

        def gen():
            for k in range(0, len(order), bs):
                ix = order[k:k + bs]
                yield ix, tok([ta[q] for q in ix], [tb[q] for q in ix], padding=True, truncation=True, max_length=MAXLEN,
                              return_tensors="pt")
        for ix, enc in _prefetch(gen()):
            enc = enc.to(DEV)
            with torch.autocast("cuda", dtype=torch.float16):
                s[ix] = model(**enc).logits.squeeze(-1).float().cpu().numpy()
        out.append(c.select("x", "s1").with_columns(ce=pl.Series(s)))
        print(f"  scored {split_name} {min(i + chunk, d.height)}/{d.height} {(time.time() - t0) / 60:.1f} min", flush=True)
    return pl.concat(out)


def score(tag="ce_full"):
    torch.cuda.set_per_process_memory_fraction(0.9)
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1)
    model.load_state_dict(torch.load(str(P("models", f"{tag}.pt")), map_location="cpu"))
    model = model.to(DEV).eval()
    # holdout universe = every pair of every X touching a fold-0 S1 (same pairs as scratch_sim_full)
    if SCORE_SET == "band":                 # same pair set as the band cross-encoders (ce_v4 / ce_xlmr_fr / ...)
        h = pl.read_parquet(P("preds", "ce_v4_holdout.parquet"), columns=["x", "s1"])
    else:
        h = pl.read_parquet(P("preds", "scratch_sim_full.parquet"), columns=["x", "s1", "p1"])
        h = _rank(h).filter(pl.col("r") <= TOPK_SCORE).select("x", "s1")
    _score_frame(model, tok, h, "train").write_parquet(P("preds", f"{tag}_holdout.parquet"))
    if SCORE_SET == "band":
        t = pl.read_parquet(P("preds", "ce_v4_test.parquet"), columns=["x", "s1"])
    else:
        t = pl.read_parquet(P("preds", "test_m_v2.parquet"), columns=["x", "s1", "p1", "country"])
        if SCORE_COUNTRY:                   # per-country final: only the countries this CE is used for
            t = t.filter(pl.col("country").is_in(SCORE_COUNTRY.split(",")))
        t = _rank(t).filter(pl.col("r") <= TOPK_SCORE).select("x", "s1")
    _score_frame(model, tok, t, "test").write_parquet(P("preds", f"{tag}_test.parquet"))
    print(f"SCORED {tag}", flush=True)


if __name__ == "__main__":
    step = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else "ce_full"
    {"train": train, "score": score, "rows": lambda t: training_rows()}[step](tag)
