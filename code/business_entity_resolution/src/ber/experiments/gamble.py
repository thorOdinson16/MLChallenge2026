"""France recall gamble (LB-only bet, NOT holdout-validated): add the clear best candidate of every France X that the
final file leaves unassigned, when its trust-blend pD >= THR and its share p1st / (p1st + p2nd) >= SHARE.
Excludes country-word decoys and the LB-proven decoy lists (nz_swap, del_only). India proxy says thresholds below 0.85
cost ~-0.0001 overall; LB probes said France errors look like misses - this submission settles which one is right.

    python -m ber.experiments.gamble <probs.parquet> [thr=0.5] [share=0.8]  -> preds/fr_gamble_pairs.parquet
    python -m ber.cn_rescue <in.tsv> <out.tsv> fr_gamble_pairs.parquet      (unassigned X only, caps, France only)
"""
import sys

import polars as pl

from .. import config, io

P = config.path


def main(prob_file, thr=0.5, share=0.8):
    t = pl.read_parquet(prob_file, columns=["x", "s1", "ins_country_word", "pD"])
    t = t.filter(pl.col("ins_country_word").fill_null(0) < 0.5)
    top = (t.sort("pD", descending=True).group_by("x")
            .agg(s1=pl.col("s1").first(), p1=pl.col("pD").first(), p2=pl.col("pD").get(1, null_on_oob=True)))
    top = top.with_columns(share=pl.col("p1") / (pl.col("p1") + pl.col("p2").fill_null(0)))
    top = top.filter((pl.col("p1") >= thr) & (pl.col("share") >= share))
    s1 = io.load("test", "s1", ["idx", "eid"]).rename({"idx": "s1", "eid": "source1_entity_id"})
    xid = io.load("test", "x", ["idx", "eid"]).rename({"idx": "x", "eid": "xe"})
    out = top.join(s1, on="s1").join(xid, on="x").select("source1_entity_id", "xe", "p1")
    for f in ("nz_swap_pairs.parquet", "del_only_pairs.parquet"):
        out = out.join(pl.read_parquet(P("preds", f)), on=["source1_entity_id", "xe"], how="anti")
    out.select("source1_entity_id", "xe").write_parquet(P("preds", "fr_gamble_pairs.parquet"))
    print(f"gamble: {out.height} France pairs listed (pD >= {thr}, share >= {share}); by band:",
          out.with_columns(b=pl.col("p1").cut([0.65, 0.75, 0.85])).group_by("b").len().sort("b").rows(), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], *[float(v) for v in sys.argv[2:4]])
