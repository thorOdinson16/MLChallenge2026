"""France recall gamble (LB-only bet): entity-aware adds on top of a finished submission.

For every France X the submission leaves unassigned, take its best candidate by trust-blend pD when that candidate is a
clear winner (share p1st / (p1st + p2nd) >= SHARE) and
  * its S1 has no match in the submission and pD >= EMPTY_THR    (empty S1: F0.5 break-even ~0.5), or
  * its S1 already has matches and pD >= FILLED_THR               (break-even ~0.75).
Country-word decoys and the LB-proven decoy lists (nz_swap, del_only) are excluded.
Evidence: US/IN holdout +0.0004 (at 0.5 / 0.75 over a flat 0.85); India unseen-country proxy -0.0005 at 0.6 / 0.8
(the least negative setting). The original clear-winner-at-0.3 version lost 0.003 on holdout and is not used.

    python -m ber.experiments.gamble <probs.parquet> <submission.tsv> [empty=0.6] [filled=0.8] [share=0.75]
        -> preds/fr_gamble_pairs.parquet ; then  python -m ber.cn_rescue <in.tsv> <out.tsv> fr_gamble_pairs.parquet
"""
import sys

import polars as pl

from .. import config, io

P = config.path


def main(prob_file, sub_file, empty_thr=0.6, filled_thr=0.8, share=0.75):
    t = pl.read_parquet(prob_file, columns=["x", "s1", "ins_country_word", "pD"])
    t = t.filter(pl.col("ins_country_word").fill_null(0) < 0.5)
    top = (t.sort("pD", descending=True).group_by("x")
            .agg(s1=pl.col("s1").first(), p1=pl.col("pD").first(), p2=pl.col("pD").get(1, null_on_oob=True)))
    top = top.with_columns(share=pl.col("p1") / (pl.col("p1") + pl.col("p2").fill_null(0)))
    s1 = io.load("test", "s1", ["idx", "eid"]).rename({"idx": "s1", "eid": "source1_entity_id"})
    xid = io.load("test", "x", ["idx", "eid"]).rename({"idx": "x", "eid": "xe"})
    sub = pl.read_csv(sub_file, separator="\t", infer_schema=False).with_columns(pl.col("matched_entity_ids").fill_null(""))
    filled = sub.filter(pl.col("matched_entity_ids") != "").select("source1_entity_id")
    out = top.join(s1, on="s1").join(xid, on="x").with_columns(
        emp=~pl.col("source1_entity_id").is_in(filled["source1_entity_id"].implode()))
    out = out.filter((pl.col("share") >= share) & (((pl.col("p1") >= empty_thr) & pl.col("emp")) |
                                                  ((pl.col("p1") >= filled_thr) & ~pl.col("emp"))))
    for f in ("nz_swap_pairs.parquet", "del_only_pairs.parquet"):
        out = out.join(pl.read_parquet(P("preds", f)), on=["source1_entity_id", "xe"], how="anti")
    out.select("source1_entity_id", "xe").write_parquet(P("preds", "fr_gamble_pairs.parquet"))
    print(f"gamble: {out.height} France pairs listed (empty S1 pD >= {empty_thr}, filled S1 pD >= {filled_thr}, "
          f"share >= {share}); empty-S1 part {out['emp'].sum()}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *[float(v) for v in sys.argv[3:6]])
