"""France trust rule (validated on the unseen-country proxy, ber.experiments.trust_proxy_s3: India 0.9600 -> 0.9630).

For a country the matcher never saw, stage 3 over-trusts the matcher probability. France decisions use a 50/50 logit
blend of the normal stage 3 (B) and a stage 3 without matcher-probability features (C), decoded at threshold 0.85.
US/India rows are copied from a base file.  Output: output/<name>.tsv (then apply ber.cn_rescue as the last step).

    python -m ber.experiments.fr_trust <name> <base_tsv> <ce_tag,...>
"""
import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

from .. import config, decode, io, matcher2, split, submit
from .overnight import BASE_COLS, PARAMS, ROUNDS, build, ce_cols, holdout_frames, raw_cols

P2COLS = ["l1", "l2", "rank_p2_x", "max_p2_x", "sum_p2_x", "sum_p2_s_other", "n_conf_s_src_other", "gap_p2_x"]
W, THR = 0.5, 0.85
SEEDS = int(os.environ.get("BER_S3_SEEDS", 1))


def main(name, base_tsv, tags):
    tags = tags.split(",")
    rc = raw_cols()
    colsB = BASE_COLS + ce_cols(tags) + rc
    colsC = [c for c in BASE_COLS if c not in P2COLS] + [f"{p}_{t}" for t in tags for p in ("ce", "cemax", "cegap")] + rc
    hold = pl.DataFrame({"s1": split.holdout_s1()})
    G = holdout_frames(tags)
    tr = pl.concat([G[g].join(hold, on="s1", how="semi") for g in ("full", "rm22")])
    del G
    y = tr["y"].to_numpy()
    XB, XC = tr.select(colsB).to_numpy().astype(np.float32), tr.select(colsC).to_numpy().astype(np.float32)
    mB = [lgb.train({**PARAMS, "seed": 1 + i}, lgb.Dataset(XB, y, feature_name=colsB), ROUNDS) for i in range(SEEDS)]
    mC = [lgb.train({**PARAMS, "seed": 1 + i}, lgb.Dataset(XC, y, feature_name=colsC), ROUNDS) for i in range(SEEDS)]
    del XB, XC
    del tr
    te = pl.read_parquet(config.path("preds", "test_m_v2.parquet"))
    dt = build(te, "test", tags, "test").filter(pl.col("country") == "France")
    dt = dt.select(["x", "s1"] + [c for c in set(colsB + colsC) if c not in rc and c in dt.columns] + ["src", "country"])
    outs = []
    for p in matcher2.parts("test_v2"):
        f = pl.read_parquet(p, columns=["x", "s1"] + rc).join(dt, on=["x", "s1"], how="inner")
        if f.height == 0:
            continue
        FB, FC = f.select(colsB).to_numpy().astype(np.float32), f.select(colsC).to_numpy().astype(np.float32)
        pB = np.mean([m.predict(FB, num_threads=12) for m in mB], axis=0)
        pC = np.mean([m.predict(FC, num_threads=12) for m in mC], axis=0)
        lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        pD = 1 / (1 + np.exp(-((1 - W) * lg(pB) + W * lg(pC))))
        outs.append(f.select("x", "s1", "src", "ins_country_word").with_columns(pB=pl.Series(pB.astype(np.float32)), pD=pl.Series(pD.astype(np.float32))))
    sc = pl.concat(outs)
    sc.write_parquet(config.path("preds", f"test_fr_trust_{name}.parquet"))
    sc = sc.filter(pl.col("ins_country_word").fill_null(0) < 0.5)
    fr = decode.apply_caps(decode.assign_unique(sc.select("x", "s1", "src", pl.col("pD").alias("p")), "p").filter(pl.col("p") >= THR), "p")
    frB = decode.apply_caps(decode.assign_unique(sc.select("x", "s1", "src", pl.col("pB").alias("p")), "p").filter(pl.col("p") >= 0.75), "p")
    # US/India from the base file
    s1 = io.load("test", "s1", ["idx", "eid", "country"]); x = io.load("test", "x", ["idx", "eid"])
    sub = pl.read_csv(base_tsv, separator="\t", infer_schema=False)
    bp = (sub.filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
          .with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
          .join(s1.select(pl.col("idx").alias("s1"), pl.col("eid").alias("source1_entity_id"), "country"), on="source1_entity_id")
          .join(x.select(pl.col("idx").alias("x"), pl.col("eid").alias("matched_entity_ids")), on="matched_entity_ids").select("s1", "x", "country"))
    base_fr = bp.filter(pl.col("country") == "France").select("s1", "x")
    nfr = s1.filter(pl.col("country") == "France").height
    print(f"France matches per S1: base {base_fr.height / nfr:.3f} | stage-3 B thr .75 {frB.height / nfr:.3f} | trust blend thr .85 {fr.height / nfr:.3f}")
    print(f"trust vs base France: +{fr.select('s1', 'x').join(base_fr, on=['s1', 'x'], how='anti').height} / -{base_fr.join(fr.select('s1', 'x'), on=['s1', 'x'], how='anti').height}")
    out = pl.concat([bp.filter(pl.col("country") != "France").select("s1", "x"), fr.select("s1", "x")])
    submit._write(out, "matched_entity_ids", config.OUT / f"{name}.tsv")


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
