"""Zero-shot LLM judge: P("Yes") that two business records are the same real-world business.

    python judge.py <in.parquet> <out.parquet> [--shard K --nshards N] [--model M] [--batch B] [--dtype fp16|bf16|fp32] [--limit L]
    python judge.py merge <out_prefix> <nshards>          -> <out_prefix>.parquet   (concat of the shard files)
Input columns: id, xn, xa, sn, sa (record A = noisy S2/S3 copy, record B = S1 reference). Output: id, p_yes.
Model default: Qwen/Qwen3-4B-Instruct-2507 (Apache-2.0, 4.0B). One forward pass per pair, no generation: the
probability is softmax over the "Yes" / "No" logits at the first assistant token.
Kaggle 2xT4 (fp16):
    CUDA_VISIBLE_DEVICES=0 python judge.py in.parquet out --shard 0 --nshards 2 &
    CUDA_VISIBLE_DEVICES=1 python judge.py in.parquet out --shard 1 --nshards 2 ; wait
    python judge.py merge out 2
"""
import argparse
import sys
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM = ("You compare two business records taken from noisy business registries and decide whether they describe "
          "the SAME real-world business. Record A is a noisy copy, record B is the reference record. Noise that does NOT "
          "change the business: typos, missing or extra letters, accents, letter case, abbreviations (Rue/R., Avenue/AV, "
          "Street/St), legal-form changes (SARL, SAS, SASU, EURL, SA, LLC, Inc, Ltd, Pvt Ltd), reordered words, "
          "website-style names, transliteration, a missing or partial address, different address formatting. Different "
          "businesses can share an address or have similar names. Answer with exactly one word: Yes or No.")


SHOTS = [  # (record A, record B, answer): conventions of this dataset, each validated on labelled training/holdout data
    ("Jhonson Plumbing Svcs | 12 Main St, Austin", "Johnson Plumbing Services LLC | 12 Main Street, Austin, TX", "Yes"),
    ("Johnson Plumbing Holdings | 12 Main St, Austin", "Johnson Plumbing LLC | 12 Main Street, Austin, TX", "No"),
    ("Dupont & Fils SARL | 3 R. Victor Hugo, Lille", "Dupont SARL | 3 Rue Victor Hugo, Lille, Hauts-de-France", "Yes"),
    ("Nantes Union SAS | 5 Rue Crebillon, Nantes", "Nantes Club SAS | 5 Rue Crebillon, Nantes, Pays de la Loire", "No"),
    ("RC | 46 R Henri Loridant, Tourcoing", "Rassemblement Club SASU | 46 Rue Henri Loridant, Tourcoing", "Yes"),
    ("Sharma Traders India | Plot 7, Sector 5, Noida", "Sharma Traders | Plot 7, Sector 5, Noida, Uttar Pradesh", "No"),
]


def prompt(tok, r, shots=False):
    user = (f"Record A: {r.xn} | {r.xa or '(no address)'}\n"
            f"Record B: {r.sn} | {r.sa or '(no address)'}\n"
            "Same real-world business?")
    msgs = [{"role": "system", "content": SYSTEM}]
    if shots:
        for a, b, y in SHOTS:
            msgs += [{"role": "user", "content": f"Record A: {a}\nRecord B: {b}\nSame real-world business?"},
                     {"role": "assistant", "content": y}]
    msgs.append({"role": "user", "content": user})
    # enable_thinking=False: hybrid Qwen3 models (e.g. the Qwen3-1.7B fallback) answer directly; ignored by Instruct-2507
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shots", action="store_true", help="few-shot examples of this dataset's conventions")
    a = ap.parse_args()
    df = pd.read_parquet(a.inp)
    df = df[df["id"] % a.nshards == a.shard].reset_index(drop=True)
    if a.limit:
        df = df.head(a.limit)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if a.dtype == "fp32" and torch.cuda.device_count() > 1:     # 4B in fp32 = 16 GB: spread over both T4s (run ONE process)
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt, device_map="auto").eval()
        dev = "cuda:0"
    else:
        # load straight onto the device (no full CPU copy: keeps host RAM low)
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt, device_map=dev, low_cpu_mem_usage=True).eval()
    yes = tok.encode("Yes", add_special_tokens=False)[0]
    no = tok.encode("No", add_special_tokens=False)[0]
    texts = [prompt(tok, r, a.shots) for r in df.itertuples()]
    order = np.argsort([len(t) for t in texts])
    p = np.full(len(texts), np.nan, dtype=np.float32)
    t0 = time.time()
    with torch.inference_mode():
        for k in range(0, len(order), a.batch):
            ix = order[k:k + a.batch]
            enc = tok([texts[i] for i in ix], return_tensors="pt", padding=True).to(dev)
            h = model.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state[:, -1]
            lg = model.lm_head(h.to(model.lm_head.weight.device)).float()
            p[ix] = torch.softmax(lg[:, [yes, no]], dim=-1)[:, 0].cpu().numpy()
            if k == 0 and np.isnan(p[ix]).any():
                sys.exit("NaN in the first batch: rerun as ONE process with --dtype fp32 --batch 8 (spreads over 2 GPUs), "
                         "or --model Qwen/Qwen3-1.7B --dtype fp32")
            if (k // a.batch) % 50 == 0:
                print(f"shard {a.shard}: {k + len(ix)}/{len(order)}  {(time.time() - t0) / 60:.1f} min", flush=True)
    out = pd.DataFrame({"id": df["id"].values, "p_yes": p})
    out.to_parquet(f"{a.out}_shard{a.shard}.parquet" if a.nshards > 1 else a.out)
    print(f"done shard {a.shard}: {len(out)} rows, NaN {int(np.isnan(p).sum())}, mean p_yes {np.nanmean(p):.3f}, "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "merge":
        pre, n = sys.argv[2], int(sys.argv[3])
        pd.concat([pd.read_parquet(f"{pre}_shard{k}.parquet") for k in range(n)]).to_parquet(f"{pre}.parquet")
        print("merged ->", f"{pre}.parquet")
    else:
        main()
