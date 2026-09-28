"""France pseudo-labels, rounds 3-5 (rounds 1-2: ber.experiments.fr_pseudo).

correct  - round-3 correction. Pairs a LB-verified France rule added on top of a base file become positives (the
           self-trained labels had them as confident negatives: confirmation bias). Other positives of those records
           are dropped, because a record has at most one true S1.
round    - rounds 4 and 5. Labels come from the France trust-rule probabilities (pD >= 0.9 -> 1, <= 0.05 -> 0) on the
           cross-encoder band. Country-word decoys are forced to 0, the noise-word rescue pairs to 1 and the LB-negative
           category-swap / dropped-word lists to 0.

    python -m ber.experiments.fr_labels correct <pseudo_name> <base.tsv> <base_plus_rule.tsv>
        e.g. correct fr_pseudo3 output/sub_K.tsv output/sub_K_cnrescue.tsv    (in place; original -> <name>_orig)
    python -m ber.experiments.fr_labels round <france_probs.parquet> <out_name>
        round 4: test_fr_trust_W1_fr.parquet  -> fr_pseudo4      round 5: test_fr_trust_L3d_fr.parquet -> fr_pseudo5
Both write preds/<name>.parquet (for ber.crossenc BER_CE_EXTRA) and kaggle_ce/<name>_kaggle.parquet (with text, for
kaggle/train_ce_big.py --pseudo). The rule lists are in code/business_entity_resolution/data/ (copy them to work/preds).
"""
import shutil
import sys

import polars as pl

from .. import config, io
from ..crossenc import add_text, band

P = config.path
COLS = ["x", "s1", "src", "country", "p1", "p2", "y"]


def _ids():
    s1 = io.load("test", "s1", ["idx", "eid"]).rename({"idx": "s1", "eid": "source1_entity_id"})
    x = io.load("test", "x", ["idx", "eid"]).rename({"idx": "x", "eid": "xe"})
    return s1, x


def _pairs(path, s1, x):
    sub = pl.read_csv(path, separator="\t", infer_schema=False).with_columns(pl.col("matched_entity_ids").fill_null(""))
    return (sub.filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
               .explode("matched_entity_ids").rename({"matched_entity_ids": "xe"})
               .join(s1, on="source1_entity_id").join(x, on="xe").select("s1", "x"))


def _write(out, name):
    out.write_parquet(P("preds", f"{name}.parquet"))
    kdir = config.OUT.parent / "kaggle_ce"
    kdir.mkdir(exist_ok=True)
    add_text(out, "test").select("x", "s1", "country", "p2", "y", "ta", "tb").write_parquet(
        kdir / f"{name}_kaggle.parquet", compression="zstd")


def correct(name, base_tsv, rule_tsv):
    s1, x = _ids()
    rescue = _pairs(rule_tsv, s1, x).join(_pairs(base_tsv, s1, x), on=["s1", "x"], how="anti")
    src = P("preds", f"{name}.parquet")
    shutil.copy(src, P("preds", f"{name}_orig.parquet"))
    ps = pl.read_parquet(src)
    te = pl.read_parquet(P("preds", "test_m_v2.parquet"), columns=COLS[:-1])
    ps2 = ps.join(rescue, on=["x", "s1"], how="anti")
    ps2 = ps2.filter(~(pl.col("x").is_in(rescue["x"].implode()) & (pl.col("y") == 1)))
    extra = te.join(rescue, on=["x", "s1"], how="semi").with_columns(y=pl.lit(1, pl.UInt8)).select(ps.columns)
    out = pl.concat([ps2, extra.cast(ps.schema)])
    _write(out, name)
    print(f"{name}: rule pairs {rescue.height}; corrected set {out.height} rows, pos {out['y'].mean():.3f} "
          f"(was {ps['y'].mean():.3f})", flush=True)


def round_labels(prob_file, name):
    s1, x = _ids()
    te = pl.read_parquet(P("preds", "test_m_v2.parquet"), columns=COLS[:-1])
    sc = pl.read_parquet(prob_file, columns=["x", "s1", "pD", "ins_country_word"])
    b = band(te.filter(pl.col("country") == "France"), 0.01, 0.99, 4).join(sc, on=["x", "s1"])
    ps = b.filter((pl.col("pD") >= 0.9) | (pl.col("pD") <= 0.05)).with_columns(y=(pl.col("pD") >= 0.9).cast(pl.UInt8))
    ps = ps.with_columns(y=pl.when(pl.col("ins_country_word").fill_null(0) >= 0.5).then(0).otherwise(pl.col("y")).cast(pl.UInt8))
    ids = lambda f: pl.read_parquet(P("preds", f)).join(s1, on="source1_entity_id").join(x, on="xe").select("x", "s1")
    pos = ids("cn_rescue_pairs_full.parquet")
    neg = pl.concat([ids("nz_swap_pairs.parquet"), ids("del_only_pairs.parquet")]).unique().join(pos, on=["x", "s1"], how="anti")
    ps = (ps.join(pos, on=["x", "s1"], how="anti").join(neg, on=["x", "s1"], how="anti")
            .filter(~(pl.col("x").is_in(pos["x"].implode()) & (pl.col("y") == 1))))
    p1 = te.join(pos, on=["x", "s1"], how="semi").with_columns(y=pl.lit(1, pl.UInt8)).select(COLS)
    n0 = te.join(neg, on=["x", "s1"], how="semi").with_columns(y=pl.lit(0, pl.UInt8)).select(COLS)
    sch = ps.select(COLS).schema
    out = pl.concat([ps.select(COLS), p1.cast(sch), n0.cast(sch)])
    _write(out, name)
    print(f"{name}: {out.height} rows, pos {out['y'].mean():.3f}; noise-word positives {p1.height}; "
          f"swap/deletion forced negatives {n0.height}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "correct":
        correct(*sys.argv[2:5])
    else:
        round_labels(*sys.argv[2:4])
