"""Raw TSV loading -> compact parquet with integer row indices.

Layout written under WORK/raw/:
  {split}_s1.parquet  idx:u32, eid:str, name, addr, country
  {split}_x.parquet   idx:u32, eid:str, src:u8 (2|3), name, addr, country   (S2 and S3 stacked)
  train_gt.parquet    s1:u32, x:u32   one row per true (S1, X) link
"""
import polars as pl

from . import config

COLS = {"entity_id": "eid", "business_name": "name", "business_address": "addr", "country": "country"}


def _scan(p):
    # quote_char=None: fields contain bare quotes; values are never quoted in these files.
    return pl.scan_csv(p, separator="\t", quote_char=None, infer_schema=False).rename(COLS)


def _clean(lf):
    return lf.with_columns(
        pl.col("name").fill_null(""), pl.col("addr").fill_null(""), pl.col("country").fill_null("")
    )


def prepare_raw(split: str) -> None:
    d = config.DATA / split
    s1 = _clean(_scan(d / f"{split}_source1.tsv")).with_row_index("idx").collect()
    s1.write_parquet(config.path("raw", f"{split}_s1.parquet"))
    x = pl.concat([
        _clean(_scan(d / f"{split}_source{k}.tsv")).with_columns(src=pl.lit(k, pl.UInt8))
        for k in (2, 3)
    ]).with_row_index("idx").collect()
    x.select("idx", "eid", "src", "name", "addr", "country").write_parquet(
        config.path("raw", f"{split}_x.parquet"))
    if split == "train":
        gt = (pl.scan_csv(d / "train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema=False)
              .filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
              .select(pl.col("source1_entity_id").alias("s1eid"),
                      pl.col("matched_entity_ids").str.split(",").alias("xeid"))
              .explode("xeid").collect())
        gt = (gt.join(s1.select(pl.col("eid").alias("s1eid"), pl.col("idx").alias("s1")), on="s1eid")
                .join(x.select(pl.col("eid").alias("xeid"), pl.col("idx").alias("x")), on="xeid")
                .select("s1", "x"))
        gt.write_parquet(config.path("raw", "train_gt.parquet"))


def load(split: str, table: str, columns=None) -> pl.DataFrame:
    """table in {'s1','x','gt'} (raw) ."""
    name = "train_gt.parquet" if table == "gt" else f"{split}_{table}.parquet"
    return pl.read_parquet(config.path("raw", name), columns=columns)


if __name__ == "__main__":
    import time
    for s in config.SPLITS:
        t = time.time()
        prepare_raw(s)
        print(s, "done", round(time.time() - t, 1), "s")
