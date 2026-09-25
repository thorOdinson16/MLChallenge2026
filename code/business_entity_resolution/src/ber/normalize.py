"""Record normalisation: raw name/address -> canonical, comparable fields.

Output (WORK/norm/{split}_{s1|x}.parquet), one row per record:
  idx, country, [src],
  nm        canonical name tokens joined by space (legal forms canonicalised)
  nm_core   nm without legal forms / stop words (identity-bearing part)
  nm_alt    canonical core of the DBA / trade-name part, if present ("" otherwise)
  nm_legal  sorted unique legal-form tokens
  nm_url    URL stem when the name is a web address ("" otherwise)
  nm_indic  name contained Indic script (u8)
  ad        canonical address string (words + numbers, original order, labels removed)
  ad_words  list of address word tokens
  ad_nums   list of address number tokens (leading zeros stripped)
  ad_state  state/region code when recognisable ("" otherwise)
  ad_box    PO box / PMB number ("" otherwise)
"""
import re
import time

import polars as pl

from . import config, io, lexicon, translit

LEET = str.maketrans("0134578@$", "oleastbas")
_ORD = re.compile(r"^\d+(st|nd|rd|th)$")
_MIXED = r"[a-z][0-9]|[0-9][a-z]"


def _fix_leet(s: str) -> str:
    out = []
    for t in s.split(" "):
        if re.search(r"[a-z]", t) and re.search(r"\d", t) and not _ORD.match(t) and len(t) > 2:
            t = t.translate(LEET)
        out.append(t)
    return " ".join(out)


def _apply_py(df: pl.DataFrame, col: str, fn, mask_regex: str) -> pl.DataFrame:
    """Apply a Python function only to rows matching mask_regex (keeps things fast)."""
    m = df[col].str.contains(mask_regex)
    if not m.any():
        return df
    vals = df[col].to_list()
    for i in m.arg_true().to_list():
        vals[i] = fn(vals[i])
    return df.with_columns(pl.Series(col, vals, dtype=pl.String))


def _latin_lower(e: pl.Expr) -> pl.Expr:
    return e.str.to_lowercase().str.normalize("NFKD").str.replace_all(r"[̀-ͯ]", "")


def _tokens_canon(e: pl.Expr, mapping: dict) -> pl.Expr:
    return e.str.split(" ").list.eval(pl.element().replace(mapping).filter(pl.element() != ""))


# ------------------------------------------------------------------------------------ names
def normalize_names(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(nm_indic=pl.col("name").str.contains(translit.POLARS_INDIC).cast(pl.UInt8))
    df = _apply_py(df, "name", translit.to_latin, translit.POLARS_INDIC)
    s = _latin_lower(pl.col("name"))
    df = df.with_columns(_s=s)
    df = df.with_columns(
        nm_url=pl.col("_s").str.extract(lexicon.URL_PATTERN, 1).fill_null("").str.replace_all("-", ""),
        _s=pl.col("_s").str.replace_all(r"\.", "").str.replace_all(r"[&+]", " and "),
    )
    df = df.with_columns(
        _main=pl.col("_s").str.extract(r"^(.*?)" + lexicon.DBA_PATTERN, 1),
        _alt=pl.col("_s").str.extract(lexicon.DBA_PATTERN + r"(.*)$", 1).fill_null(""),
    )
    df = _apply_py(df, "_s", _fix_leet, _MIXED)
    clean = lambda e: e.str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars()  # noqa: E731
    df = df.with_columns(_tok=_tokens_canon(clean(pl.col("_s")), lexicon.NAME_CANON),
                         _alt_tok=_tokens_canon(clean(pl.col("_alt")), lexicon.NAME_CANON))
    legal = list(lexicon.LEGAL)
    drop = list(lexicon.LEGAL | lexicon.NAME_STOP)
    df = df.with_columns(
        nm=pl.col("_tok").list.join(" "),
        nm_core=pl.col("_tok").list.eval(pl.element().filter(~pl.element().is_in(drop))).list.join(" "),
        nm_legal=pl.col("_tok").list.eval(pl.element().filter(pl.element().is_in(legal))).list.unique()
                   .list.sort().list.join(" "),
        nm_alt=pl.col("_alt_tok").list.eval(pl.element().filter(~pl.element().is_in(drop))).list.join(" "),
    )
    # a URL name is a single glued token; use its stem as the name, and never leave the core empty
    is_url = pl.col("nm_url") != ""
    df = df.with_columns(nm=pl.when(is_url).then(pl.col("nm_url")).otherwise(pl.col("nm")),
                         nm_core=pl.when(is_url).then(pl.col("nm_url")).otherwise(pl.col("nm_core")))
    df = df.with_columns(nm_core=pl.when(pl.col("nm_core") == "").then(pl.col("nm")).otherwise(pl.col("nm_core")))
    return df.drop("_s", "_main", "_alt", "_tok", "_alt_tok")


# ------------------------------------------------------------------------------------ addresses
def normalize_addresses(df: pl.DataFrame, country: str) -> pl.DataFrame:
    amap = lexicon.ADDR_MAPS.get(country, lexicon.ADDR_GENERIC)
    states = lexicon.STATE_MAPS.get(country, {})
    df = _apply_py(df, "addr", translit.to_latin, translit.POLARS_INDIC)
    a = _latin_lower(pl.col("addr"))
    df = df.with_columns(_a=a)
    df = df.with_columns(
        ad_box=pl.col("_a").str.extract(r"\b(?:p\.?\s?o\.?\s?box|pmb|post box|box)\s*#?\s*(\d+)", 1).fill_null(""),
        _a=pl.col("_a")
            .str.replace_all(r"\b(?:p\.?\s?o\.?\s?box|pmb|post box)\s*#?\s*\d+", " ")
            .str.replace_all(r"<null>|\bn/a\b|\bnull\b|\bnone\b|\bnan\b", " ")
            .str.replace_all(r"(\d+)\s?(?:st|nd|rd|th)\b", "${1}")
            .str.replace_all(r"[º°]", " "),
    )
    if states:
        # a comma-separated component that is exactly a state name/code is the state; it is moved
        # out of the address (state names inside other components, e.g. "Iowa City", are kept)
        codes = sorted(set(states.values()) | set(states.keys()))
        comp = (pl.col("_a").str.replace_all(r"\.", " ").str.split(",")
                .list.eval(pl.element().str.replace_all(r"\s+", " ").str.strip_chars()))
        df = df.with_columns(
            ad_state=comp.list.eval(pl.element().filter(pl.element().is_in(codes))).list.first()
                         .replace(states).fill_null(""),
            _a=comp.list.eval(pl.element().filter(~pl.element().is_in(codes))).list.join(" , "),
        )
    else:
        df = df.with_columns(ad_state=pl.lit(""))
    df = df.with_columns(_t=pl.col("_a").str.extract_all(r"[a-z]+|\d+"))
    stop = list(lexicon.ADDR_STOP)
    df = df.with_columns(
        _t=pl.col("_t").list.eval(
            pl.when(pl.element().str.contains(r"^\d+$"))
              .then(pl.element().str.strip_chars_start("0").replace("", "0"))
              .otherwise(pl.element().replace(amap))
        ).list.eval(pl.element().filter(~pl.element().is_in(stop) & (pl.element().str.len_chars() > 1)
                                        | pl.element().str.contains(r"^\d+$"))))
    df = df.with_columns(
        ad=pl.col("_t").list.join(" "),
        ad_words=pl.col("_t").list.eval(pl.element().filter(~pl.element().str.contains(r"^\d+$"))),
        ad_nums=pl.col("_t").list.eval(pl.element().filter(pl.element().str.contains(r"^\d+$"))),
    )
    return df.drop("_a", "_t")


def normalize_frame(df: pl.DataFrame) -> pl.DataFrame:
    df = normalize_names(df)
    parts = []
    for (c,), g in df.group_by("country", maintain_order=True):
        parts.append(normalize_addresses(g, c))
    df = pl.concat(parts).sort("idx")
    keep = ["idx", "country"] + (["src"] if "src" in df.columns else []) + [
        "nm", "nm_core", "nm_alt", "nm_legal", "nm_url", "nm_indic",
        "ad", "ad_words", "ad_nums", "ad_state", "ad_box"]
    return df.select(keep)


def run(split: str, table: str, chunk: int = 2_000_000) -> None:
    df = io.load(split, table)
    out = []
    for i in range(0, df.height, chunk):
        out.append(normalize_frame(df.slice(i, chunk)))
    res = pl.concat(out)
    res.write_parquet(config.path("norm", f"{split}_{table}.parquet"), compression="zstd")


def load(split: str, table: str, columns=None) -> pl.DataFrame:
    return pl.read_parquet(config.path("norm", f"{split}_{table}.parquet"), columns=columns)


if __name__ == "__main__":
    import sys
    targets = [(s, t) for s in config.SPLITS for t in ("s1", "x")]
    if len(sys.argv) > 1 and sys.argv[1] == "--sample":
        df = io.load("train", "x").sample(40, seed=int(sys.argv[2]) if len(sys.argv) > 2 else 0)
        r = normalize_frame(df).join(df.select("idx", "name", "addr"), on="idx")
        pl.Config.set_tbl_rows(100); pl.Config.set_tbl_width_chars(300); pl.Config.set_fmt_str_lengths(70)
        print(r.select("name", "nm", "nm_core", "nm_alt", "nm_legal", "nm_url"))
        print(r.select("addr", "ad", "ad_nums", "ad_state", "ad_box"))
        sys.exit()
    for s, t in targets:
        t0 = time.time()
        run(s, t)
        print(s, t, "normalised in", round(time.time() - t0, 1), "s")
