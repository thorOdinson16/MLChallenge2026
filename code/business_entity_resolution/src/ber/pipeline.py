"""End-to-end driver. Each step caches its output under WORK and can be re-run on its own.

  python -m ber.pipeline prepare        raw TSV -> parquet, transliteration dict, normalisation
  python -m ber.pipeline encoder        fine-tune the bi-encoder (holdout S1 excluded)
  python -m ber.pipeline prune          un-pruned candidate sample -> stage-0 pruner
  python -m ber.pipeline cands          pruned candidates for train and test
  python -m ber.pipeline feats          features for TRAIN_X / EVAL_X / test
  python -m ber.pipeline train          matcher (stage 1 + 2), holdout evaluation
  python -m ber.pipeline submit         test prediction -> decode -> output TSVs -> validator
  python -m ber.pipeline all            everything in order
"""
import sys
import time

import polars as pl

from . import candidates, config, decode, evaluate, io, matcher, normalize, prune, split, submit, translit

TAG = "m_v1"
CAND = "{split}_pruned"
K0 = 12


def prepare():
    for s in config.SPLITS:
        io.prepare_raw(s)
    translit.build()
    for s in config.SPLITS:
        for t in ("s1", "x"):
            normalize.run(s, t)


def encoder_step():
    from .blocking import encoder
    encoder.train(tag=candidates.ENC_TAG, epochs=2, n_pairs=10_000_000,
                  exclude_s1=split.s1_in(split.HOLDOUT | split.MATCH))


LEX = ("lsa_full", "lsa_name", "keys")


def retrieve_step():
    """Unpruned retriever passes, stored separately so each can be re-run on its own."""
    for s in config.SPLITS:
        candidates.generate(s, f"{s}_lex", LEX)
        candidates.generate(s, f"{s}_enc", ("enc",))


def prune_step():
    xs = io.load("train", "x", ["idx"]).sample(600_000, seed=config.SEED).rename({"idx": "x"})
    candidates.merge(["train_lex", "train_enc"], "prune_sample", x_subset=xs)
    prune.train()


def cands_step():
    for s in config.SPLITS:
        candidates.merge([f"{s}_lex", f"{s}_enc"], CAND.format(split=s), prune_k=K0)
    gt = io.load("train", "gt")
    c = candidates.load(CAND.format(split="train"), ["x", "s1"])
    hold_gt = gt.join(pl.DataFrame({"s1": split.holdout_s1()}), on="s1", how="semi")
    print("holdout blocking recall:", round(evaluate.blocking_recall(c, hold_gt), 5), flush=True)


def feats_step():
    matcher.make_sets(CAND.format(split="train"))
    matcher.featurize("train", CAND.format(split="train"), pl.read_parquet(config.path("sets", "train_x.parquet")),
                      "train_x")
    matcher.featurize("train", CAND.format(split="train"), pl.read_parquet(config.path("sets", "eval_x.parquet")),
                      "eval_x")
    matcher.featurize("test", CAND.format(split="test"), None, "test")


def evaluate_preds(ev: pl.DataFrame, label=""):
    gt = io.load("train", "gt")
    ev_s1 = matcher.eval_s1()
    meta = io.load("train", "s1", ["idx", "country"]).rename({"idx": "s1"}).join(
        pl.DataFrame({"s1": ev_s1}), on="s1", how="semi")
    best = None
    for col in ("p1", "p2"):
        for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
            r = evaluate.report(decode.decode_threshold(ev, thr, col), gt, meta, f"{label} {col} thr={thr}")
            if best is None or r["all"] > best[0]:
                best = (r["all"], f"{col} thr={thr}")
    for miss in (0.0, 0.05):
        r = evaluate.report(decode.decode(ev, "p2", miss=miss), gt, meta, f"{label} p2 expected-F miss={miss}")
        if r["all"] > best[0]:
            best = (r["all"], f"p2 expF miss={miss}")
    print("BEST:", best, flush=True)
    return best


def train_step():
    ev = matcher.train_and_eval(tag=TAG)
    evaluate_preds(ev, TAG)


def submit_step(decoder="expf"):
    t = matcher.predict("test", "test", TAG)
    t.select("x", "s1", "p0", "p1", "p2").write_parquet(config.path("preds", f"test_{TAG}.parquet"))
    m = decode.decode(t, "p2") if decoder == "expf" else decode.decode_threshold(t, 0.5, "p2")
    submit.write(m, candidates.load(CAND.format(split="test"), ["x", "s1"]))
    submit.validate()


STEPS = {"prepare": prepare, "encoder": encoder_step, "retrieve": retrieve_step, "prune": prune_step,
         "cands": cands_step,
         "feats": feats_step, "train": train_step, "submit": submit_step}

if __name__ == "__main__":
    todo = sys.argv[1:] or ["all"]
    if todo == ["all"]:
        todo = list(STEPS)
    for s in todo:
        t = time.time()
        print(f"===== {s}", flush=True)
        STEPS[s]()
        print(f"===== {s} done in {time.time() - t:.0f}s", flush=True)
