"""Builds the France rule lists used by the post-steps (label-free "unseen-country" protocol).

    python -m ber.rule_lists [out_dir]      (default: work/rules/; the shipped copies live in work/preds/)

1. Word roles (per country, label-free). For every token inserted into an X's core name relative to its top S1
   (argmax stage-2 p2), two signatures over the country's own test pairs:
     up_ratio = P(first house number shifted +1..+9 | token inserted) / P(shift | any top pair)   (decoy signature)
     strong   = P(address token_set >= 90 and same first house number | token inserted)
   Tokens with >= 300 insertions get a class: CN (up_ratio < 0.6, strong >= 0.5), NZ (up_ratio < 1.2, strong >= 0.2),
   DC (up_ratio >= 2), AM otherwise. On US/India train the same table is checked against labels.
2. Cells. Each X's top pair gets a country-agnostic descriptor:
     addr  : EMPTY / STRONG (same first number, token_set >= 90) / SAMENUM_lo / UP / DOWN / NONUM_* / DIFFNUM_*
     name  : "<#inserted>i:<classes of inserted tokens>|<#deleted>d:<rw|ty of deleted tokens>"
             (rw = real word, used by >= 5 S1 names; ty = rare token, i.e. a typo)
     rival : runner-up p2 >= 0.1;  acc : p2 >= 0.8
3. Lists (France, all restricted to STRONG address and no rival), written as (source1_entity_id, xe):
     cn_rescue_pairs.parquet       name starts "1i:CN|", not accepted by stage 2, X unassigned in sub_K   (LB +0.00054)
     cn_rescue_pairs_full.parquet  cn_rescue_pairs  U  every STRONG / "1i:CN|" / no-rival pair           (used in S1+)
     nz_swap_pairs.parquet         name == "1i:NZ|1d:rw"  (one word swapped for another real word)      (LB-negative)
     del_only_pairs.parquet        name == "0i:-|1d:rw"   (one real word dropped)                       (LB-negative)
   The last two are kept only as decoy references (ber.experiments.guard); they must never be added to a submission.
Inputs: work/preds/test_m_v2.parquet, work/norm/test_{x,s1}.parquet, output/sub_K.tsv, work/raw/test_{x,s1}.parquet.
"""
import sys
from pathlib import Path

import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from . import config

P = config.path
MIN_INS, SAMPLE = 300, 1_500_000


def _top_pairs(split_name, country, keep_second=False):
    pred = "test_m_v2.parquet" if split_name == "test" else "train_m_v2.parquet"
    p = pl.read_parquet(P("preds", pred), columns=["x", "s1", "p2", "country"]).filter(pl.col("country") == country).drop("country")
    p = p.sort(["x", "p2", "s1"], descending=[False, True, False])
    if not keep_second:
        return p.unique("x", keep="first")
    p = p.with_columns(rk=pl.int_range(pl.len()).over("x"))
    top = p.filter(pl.col("rk") == 0).join(p.filter(pl.col("rk") == 1).select("x", pl.col("p2").alias("p2b")), on="x", how="left")
    return top.with_columns(pl.col("p2b").fill_null(0)).drop("rk")


def _with_text(top, split_name):
    X = pl.scan_parquet(P("norm", f"{split_name}_x.parquet")).select("idx", "nm_core", "ad", "ad_nums").join(
        top.select(pl.col("x").alias("idx")).lazy(), on="idx", how="semi").collect()
    S = pl.scan_parquet(P("norm", f"{split_name}_s1.parquet")).select("idx", "nm_core", "ad", "ad_nums").join(
        top.select(pl.col("s1").alias("idx")).unique().lazy(), on="idx", how="semi").collect()
    d = top.join(X.rename({"idx": "x", "nm_core": "xc", "ad": "xa", "ad_nums": "xn"}), on="x").join(
        S.rename({"idx": "s1", "nm_core": "sc", "ad": "sa", "ad_nums": "sn"}), on="s1")
    d = d.with_columns(pl.Series("atset", cpdist(d["xa"].to_list(), d["sa"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)))
    split = lambda c: pl.col(c).str.split(" ")  # noqa: E731
    return d.with_columns(
        ins=split("xc").list.set_difference(split("sc")).list.eval(pl.element().filter(pl.element() != "")),
        dele=split("sc").list.set_difference(split("xc")).list.eval(pl.element().filter(pl.element() != "")))


def word_roles(split_name="test", country="France"):
    """Step 1: label-free role table of inserted tokens."""
    top = _top_pairs(split_name, country)
    if top.height > SAMPLE:
        top = top.sample(SAMPLE, seed=11)
    d = _with_text(top, split_name)
    off = pl.col("xn").list.first().cast(pl.Int64, strict=False) - pl.col("sn").list.first().cast(pl.Int64, strict=False)
    d = d.with_columns(up=off.is_between(1, 9).fill_null(False),
                       strong=((pl.col("atset") >= 90) & (pl.col("xn").list.first() == pl.col("sn").list.first()).fill_null(False)
                               & (pl.col("xa") != "")))
    base_up = d["up"].mean()
    e = d.select("x", "ins").explode("ins").drop_nulls().rename({"ins": "t"}).unique().join(d.select("x", "up", "strong"), on="x")
    r = e.group_by("t").agg(n=pl.len(), up_ratio=pl.col("up").mean() / base_up, strong=pl.col("strong").mean())
    return r.filter(pl.col("n") >= MIN_INS)


def cells(roles, split_name="test", country="France"):
    """Step 2: address class x name-edit profile x rival for every X's top pair."""
    cls = roles.select("t", cls=pl.when((pl.col("up_ratio") < 0.6) & (pl.col("strong") >= 0.5)).then(pl.lit("CN"))
                       .when((pl.col("up_ratio") < 1.2) & (pl.col("strong") >= 0.2)).then(pl.lit("NZ"))
                       .when(pl.col("up_ratio") >= 2).then(pl.lit("DC")).otherwise(pl.lit("AM")))
    names = pl.scan_parquet(P("norm", f"{split_name}_s1.parquet")).filter(pl.col("country") == country).select("nm_core").collect()["nm_core"]
    dfs = (pl.DataFrame({"t": names.str.split(" ")}).with_row_index("r").explode("t").filter(pl.col("t") != "")
             .unique(["r", "t"]).group_by("t").len("df_s"))
    d = _with_text(_top_pairs(split_name, country, keep_second=True), split_name)
    x1, s1n = pl.col("xn").list.first(), pl.col("sn").list.first()
    off = x1.cast(pl.Int64, strict=False) - s1n.cast(pl.Int64, strict=False)
    d = d.with_columns(addr=pl.when(pl.col("xa") == "").then(pl.lit("EMPTY"))
                       .when(x1.is_null() | s1n.is_null()).then(pl.when(pl.col("atset") >= 90).then(pl.lit("NONUM_hi")).otherwise(pl.lit("NONUM_lo")))
                       .when(x1 == s1n).then(pl.when(pl.col("atset") >= 90).then(pl.lit("STRONG")).otherwise(pl.lit("SAMENUM_lo")))
                       .when(off.is_between(1, 9)).then(pl.lit("UP")).when(off.is_between(-9, -1)).then(pl.lit("DOWN"))
                       .otherwise(pl.when(pl.col("atset") >= 90).then(pl.lit("DIFFNUM_hi")).otherwise(pl.lit("DIFFNUM_lo"))))
    e = d.select("x", "ins").explode("ins").drop_nulls().rename({"ins": "t"}).join(cls, on="t", how="left").join(dfs, on="t", how="left")
    e = e.with_columns(c=pl.when(pl.col("cls").is_not_null()).then(pl.col("cls"))
                       .when(pl.col("df_s").fill_null(0) >= 5).then(pl.lit("RW")).otherwise(pl.lit("TY")))
    ei = e.group_by("x").agg(ins_prof=pl.col("c").unique().sort().str.join("+"))
    e2 = d.select("x", "dele").explode("dele").drop_nulls().rename({"dele": "t"}).join(dfs, on="t", how="left").with_columns(
        c=pl.when(pl.col("df_s").fill_null(0) >= 5).then(pl.lit("rw")).otherwise(pl.lit("ty")))
    ed = e2.group_by("x").agg(del_prof=pl.col("c").unique().sort().str.join("+"))
    d = d.join(ei, on="x", how="left").join(ed, on="x", how="left").with_columns(
        ni=pl.col("ins").list.len().clip(0, 2).cast(pl.Utf8), nd=pl.col("dele").list.len().clip(0, 2).cast(pl.Utf8))
    d = d.with_columns(name=pl.col("ni") + "i:" + pl.col("ins_prof").fill_null("-") + "|" + pl.col("nd") + "d:" + pl.col("del_prof").fill_null("-"),
                       rival=pl.col("p2b") >= 0.1, acc=pl.col("p2") >= 0.8)
    return d.select("x", "s1", "p2", "p2b", "addr", "name", "rival", "acc")


def lists(c, out_dir):
    """Step 3: the four France lists."""
    s1 = pl.read_parquet(P("raw", "test_s1.parquet"), columns=["idx", "eid"]).rename({"idx": "s1", "eid": "source1_entity_id"})
    xi = pl.read_parquet(P("raw", "test_x.parquet"), columns=["idx", "eid"]).rename({"idx": "x", "eid": "xe"})
    ids = lambda d: d.join(s1, on="s1").join(xi, on="x").select("source1_entity_id", "xe")  # noqa: E731
    base = c.filter((pl.col("addr") == "STRONG") & ~pl.col("rival"))
    cn_cell = base.filter(pl.col("name").str.starts_with("1i:CN|"))
    # residual on sub_K: CN pairs stage 2 rejected whose X sub_K left unassigned
    sub = pl.read_csv(config.OUT / "sub_K.tsv", separator="\t", infer_schema=False).filter(pl.col("matched_entity_ids").is_not_null())
    sub = (sub.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
              .rename({"matched_entity_ids": "xe"}).join(xi, on="xe").select("x"))
    cn = ids(cn_cell.filter(~pl.col("acc")).join(sub.unique(), on="x", how="anti"))
    out = {"cn_rescue_pairs.parquet": cn,
           "cn_rescue_pairs_full.parquet": pl.concat([cn, ids(cn_cell)]).unique(),
           "nz_swap_pairs.parquet": ids(base.filter(pl.col("name") == "1i:NZ|1d:rw")),
           "del_only_pairs.parquet": ids(base.filter(pl.col("name") == "0i:-|1d:rw"))}
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, d in out.items():
        d.write_parquet(out_dir / name)
        print(f"{name}: {d.height} pairs", flush=True)


if __name__ == "__main__":
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else config.WORK / "rules"
    roles = word_roles("test", "France")
    print("France word roles:", roles.height, "tokens with >=", MIN_INS, "insertions", flush=True)
    lists(cells(roles, "test", "France"), out_dir)
