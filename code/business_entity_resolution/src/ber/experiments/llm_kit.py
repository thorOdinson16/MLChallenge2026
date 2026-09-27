"""Build the LLM-judge kit: labelled US/IN holdout gate slices + France grey-zone slices (text pairs).
    python -m ber.experiments.llm_kit  -> work/llm_kit/{gate_holdout,france_slices}.parquet
Slices (same definition on both sides; probabilities: holdout = stage-2 p2 (m_v2ce), France = trust blend pD):
  add  : X unassigned (top p < 0.85), top p in [0.3, 0.85), top share p1st/(p1st+p2nd) >= 0.6, no country word
  veto : X accepted, top p in [0.85, 0.97), share >= 0.6, no country word
"""
import polars as pl

from .. import config, io, split
from ..features import COUNTRY_WORDS

OUT = config.WORK / "llm_kit"
CAP_GATE, CAP_FR = 3000, 40000


def top2(d, pc):
    d = d.sort(["x", pc], descending=[False, True]).with_columns(r=pl.col(pc).cum_count().over("x"))
    a = d.filter(pl.col("r") == 1).select("x", "s1", p=pl.col(pc))
    b = d.filter(pl.col("r") == 2).select("x", p2nd=pl.col(pc))
    return a.join(b, on="x", how="left").with_columns(pl.col("p2nd").fill_null(0.0)).with_columns(
        share=pl.col("p") / (pl.col("p") + pl.col("p2nd")))


def slices(d):
    ok = (pl.col("share") >= 0.6) & ~pl.col("cw")
    return pl.concat([d.filter(ok & (pl.col("p") >= 0.3) & (pl.col("p") < 0.85)).with_columns(slice=pl.lit("add")),
                      d.filter(ok & (pl.col("p") >= 0.85) & (pl.col("p") < 0.97)).with_columns(slice=pl.lit("veto"))])


def texts(d, split_name):
    x = io.load(split_name, "x", ["idx", "name", "addr", "country"]).rename({"idx": "x", "name": "xn", "addr": "xa"})
    s = io.load(split_name, "s1", ["idx", "name", "addr"]).rename({"idx": "s1", "name": "sn", "addr": "sa"})
    d = d.join(x, on="x").join(s, on="s1")
    words = pl.col("country").replace_strict({k: "|".join(v) for k, v in COUNTRY_WORDS.items()}, default="^$")
    return d.with_columns(pl.col("xa").fill_null(""), pl.col("sa").fill_null(""), words=words)


def country_word(d):
    # X name contains the country word while the S1 name does not (the shipped decoy veto)
    xl, sl = pl.col("xn").str.to_lowercase(), pl.col("sn").str.to_lowercase()
    return d.with_columns(cw=xl.str.contains(pl.col("words")) & ~sl.str.contains(pl.col("words")))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    hold = pl.DataFrame({"s1": split.holdout_s1()})
    lp = pl.scan_parquet(config.path("preds", "train_m_v2ce.parquet")).select("x", "s1", "p2")
    ev = lp.join(hold.lazy(), on="s1", how="semi").select("x").unique().collect()
    h = top2(lp.join(ev.lazy(), on="x", how="semi").collect(), "p2")
    h = country_word(texts(h, "train"))
    h = slices(h).join(io.load("train", "gt").with_columns(y=pl.lit(1, pl.UInt8)), on=["s1", "x"], how="left").with_columns(
        pl.col("y").fill_null(0))
    for sl in ("add", "veto"):
        g = h.filter(pl.col("slice") == sl)
        print(f"holdout {sl}: {g.height} pairs, precision {g['y'].mean():.3f}", flush=True)
    gate = pl.concat([h.filter(pl.col("slice") == sl).sample(n=min(CAP_GATE, h.filter(pl.col("slice") == sl).height), seed=5)
                      for sl in ("add", "veto")])
    gate = gate.with_row_index("id").select("id", "slice", "x", "s1", "p", "share", "xn", "xa", "sn", "sa", "country", "y")
    gate.write_parquet(OUT / "gate_holdout.parquet")
    t = pl.read_parquet(config.path("preds", "test_fr_trust_W1_fr.parquet"), columns=["x", "s1", "pD", "ins_country_word"])
    t = t.filter(pl.col("ins_country_word").fill_null(0) < 0.5)
    f = country_word(texts(top2(t, "pD"), "test"))
    f = slices(f)
    parts = []
    for sl in ("add", "veto"):
        g = f.filter(pl.col("slice") == sl)
        print(f"France {sl}: {g.height} pairs", flush=True)
        parts.append(g.sort("p", descending=(sl == "add")).head(CAP_FR))
    fr = pl.concat(parts).with_row_index("id").select("id", "slice", "x", "s1", "p", "share", "xn", "xa", "sn", "sa", "country")
    fr.write_parquet(OUT / "france_slices.parquet")
    print("gate", gate.height, "france", fr.height, "->", OUT, flush=True)
