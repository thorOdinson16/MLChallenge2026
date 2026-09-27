"""Score the test band with a saved cross-encoder checkpoint (recovery when training finished but test scoring died).
    python -m ber.experiments.ce_score <tag> [lo=0.01] [hi=0.99] [k=4]      -> preds/<tag>_test.parquet
"""
import sys, time

import numpy as np
import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .. import config
from ..crossenc import BASE, DEV, MAXLEN, add_text, band

tag, lo, hi, k = sys.argv[1], *(float(v) for v in sys.argv[2:4]), int(sys.argv[4])
te = pl.read_parquet(config.path("preds", "test_m_v2.parquet"), columns=["x", "s1", "src", "country", "p1", "p2"])
tt = add_text(band(te, lo, hi, k), "test")
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1)
model.load_state_dict(torch.load(str(config.path("models", f"{tag}.pt")), map_location="cpu"))
model.to(DEV).eval()
ta, tb = tt["ta"].to_list(), tt["tb"].to_list()
out = np.zeros(tt.height, dtype=np.float32); t0 = time.time()
with torch.inference_mode():
    for n, i in enumerate(range(0, tt.height, 256)):
        enc = tok(ta[i:i + 256], tb[i:i + 256], padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt")
        with torch.autocast("cuda", dtype=torch.float16):
            out[i:i + 256] = model(**{a: b.to(DEV) for a, b in enc.items()}).logits.squeeze(-1).float().cpu().numpy()
        if n % 1000 == 0:
            print(f"  scoring test: {i}/{tt.height} {time.time() - t0:.0f}s", flush=True)
tt.select("x", "s1").with_columns(ce=pl.Series(out)).write_parquet(config.path("preds", f"{tag}_test.parquet"))
print(f"DONE {tag} test {tt.height}", flush=True)
