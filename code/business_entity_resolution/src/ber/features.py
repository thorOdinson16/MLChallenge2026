"""Pairwise features for (x, s1) candidate pairs.

All features are language-agnostic similarity/frequency signals computed on normalised fields,
so a model trained on US/India transfers to unseen countries (France). `country` itself is never
a feature.
"""
import numpy as np
import polars as pl
import scipy.sparse as sp
from rapidfuzz import fuzz, distance
from rapidfuzz.process import cpdist
from sklearn.feature_extraction.text import CountVectorizer

S1_COLS = ["idx", "nm", "nm_core", "nm_alt", "nm_legal", "nm_url", "ad", "ad_words", "ad_nums", "ad_state", "ad_box"]
X_COLS = S1_COLS + ["nm_indic", "src"]


# ---------------------------------------------------------------------------- token IDF spaces
class TokenSpace:
    """Binary token matrices with S1-based IDF for one field. Unknown X tokens are counted separately."""

    def __init__(self, s1_texts, x_texts):
        self.cv = CountVectorizer(token_pattern=r"\S+", binary=True, lowercase=False, dtype=np.float32)
        self.S = self.cv.fit_transform(s1_texts).tocsr()
        df = np.asarray(self.S.sum(0)).ravel()
        self.idf = (np.log((self.S.shape[0] + 1) / (df + 1)) + 1).astype(np.float32)
        self.X = self.cv.transform(x_texts).tocsr()
        self.s_sum = self.S @ self.idf
        self.x_sum = self.X @ self.idf
        n_tok = np.array([len(t.split()) if t else 0 for t in x_texts], dtype=np.float32)
        self.x_unknown = n_tok - np.asarray(self.X.sum(1)).ravel()
        self.idf_max = float(self.idf.max())

    def pair(self, xi, si, prefix):
        A = self.X[xi]
        B = self.S[si]
        inter = A.multiply(B).tocsr()
        shared = inter @ self.idf
        shared_max = np.asarray(inter.multiply(self.idf[None, :]).max(1).todense()).ravel()
        sx = self.x_sum[xi] + self.x_unknown[xi] * self.idf_max
        ss = self.s_sum[si]
        return {
            f"{prefix}_wjacc": shared / np.maximum(sx + ss - shared, 1e-6),
            f"{prefix}_wcov_x": shared / np.maximum(sx, 1e-6),
            f"{prefix}_wcov_s": shared / np.maximum(ss, 1e-6),
            f"{prefix}_shared_max_idf": shared_max,
            f"{prefix}_n_shared": np.asarray(inter.sum(1)).ravel(),
            f"{prefix}_x_unknown": self.x_unknown[xi],
        }


class FeatureContext:
    """Everything needed to featurise pairs of one split+country, built once."""

    def __init__(self, s1: pl.DataFrame, x: pl.DataFrame):
        self.s1 = s1.select(S1_COLS).sort("idx")
        self.x = x.select(X_COLS).sort("idx")
        # positional lookup: global idx -> row
        self.s_pos = np.full(int(self.s1["idx"].max()) + 1, -1, dtype=np.int64)
        self.s_pos[self.s1["idx"].to_numpy()] = np.arange(self.s1.height)
        self.x_pos = np.full(int(self.x["idx"].max()) + 1, -1, dtype=np.int64)
        self.x_pos[self.x["idx"].to_numpy()] = np.arange(self.x.height)
        self.name_sp = TokenSpace(self.s1["nm_core"].to_list(), self.x["nm_core"].to_list())
        aw = lambda d: d["ad_words"].list.join(" ").to_list()  # noqa: E731
        an = lambda d: d["ad_nums"].list.join(" ").to_list()  # noqa: E731
        self.addr_sp = TokenSpace(aw(self.s1), aw(self.x))
        self.num_sp = TokenSpace(an(self.s1), an(self.x))
        # frequency of exact values among S1 (generic names / shared buildings)
        self.s1_core_freq = self._freq(self.s1["nm_core"], self.s1["nm_core"])
        self.s1_ad_freq = self._freq(self.s1["ad"], self.s1["ad"])
        self.x_core_freq_in_s1 = self._freq(self.s1["nm_core"], self.x["nm_core"])

    @staticmethod
    def _freq(ref: pl.Series, target: pl.Series) -> np.ndarray:
        c = ref.value_counts(name="c")
        return (pl.DataFrame({ref.name: target}).join(c, on=ref.name, how="left")["c"]
                .fill_null(0).to_numpy().astype(np.float32))


def _cp(a, b, scorer):
    return cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32) / (100.0 if scorer not in (
        distance.JaroWinkler.normalized_similarity, distance.Levenshtein.distance) else 1.0)


def _num_feats(xn: pl.Series, sn: pl.Series) -> dict:
    d = pl.DataFrame({"xn": xn, "sn": sn}).with_columns(
        x1=pl.col("xn").list.first().fill_null(""), s1=pl.col("sn").list.first().fill_null(""))
    d = d.with_columns(
        both=(pl.col("x1") != "") & (pl.col("s1") != ""),
        eq=(pl.col("x1") == pl.col("s1")) & (pl.col("x1") != ""),
        pref=(pl.col("s1").str.starts_with(pl.col("x1")) | pl.col("x1").str.starts_with(pl.col("s1")))
             & (pl.col("x1") != "") & (pl.col("s1") != ""),
        suf=(pl.col("s1").str.ends_with(pl.col("x1")) | pl.col("x1").str.ends_with(pl.col("s1")))
            & (pl.col("x1") != "") & (pl.col("s1") != ""),
        x_in_s=pl.col("sn").list.contains(pl.col("x1")),
        s_in_x=pl.col("xn").list.contains(pl.col("s1")),
        nx=pl.col("xn").list.len(), ns=pl.col("sn").list.len(),
        n_common=pl.col("xn").list.set_intersection(pl.col("sn")).list.len(),
    )
    lev = cpdist(d["x1"].to_list(), d["s1"].to_list(), scorer=distance.Levenshtein.distance, workers=-1)
    return {
        "num_first_both": d["both"].to_numpy().astype(np.float32),
        "num_first_eq": d["eq"].to_numpy().astype(np.float32),
        "num_first_prefix": d["pref"].to_numpy().astype(np.float32),
        "num_first_suffix": d["suf"].to_numpy().astype(np.float32),
        "num_first_lev": np.where(d["both"].to_numpy(), lev, -1).astype(np.float32),
        "num_x_first_in_s": d["x_in_s"].fill_null(False).to_numpy().astype(np.float32),
        "num_s_first_in_x": d["s_in_x"].fill_null(False).to_numpy().astype(np.float32),
        "num_nx": d["nx"].to_numpy().astype(np.float32),
        "num_ns": d["ns"].to_numpy().astype(np.float32),
        "num_common": d["n_common"].to_numpy().astype(np.float32),
    }


def _initials_forms(core: pl.Series) -> tuple:
    """Glued core and 'initials + last word' forms, used to compare against URL-style names."""
    t = pl.col("t")
    d = pl.DataFrame({"t": core.str.split(" ")}).select(
        glued=t.list.join(""),
        init_last=t.list.slice(0, t.list.len() - 1).list.eval(pl.element().str.slice(0, 1)).list.join("")
                  + t.list.last().fill_null(""),
        inits=t.list.eval(pl.element().str.slice(0, 1)).list.join(""))
    return d["glued"], d["init_last"], d["inits"]


def featurize(ctx: FeatureContext, pairs: pl.DataFrame) -> pl.DataFrame:
    """pairs: (x, s1, ...retrieval columns...). Returns pairs + feature columns (float32)."""
    xi = ctx.x_pos[pairs["x"].to_numpy()]
    si = ctx.s_pos[pairs["s1"].to_numpy()]
    X = ctx.x[xi]
    S = ctx.s1[si]
    f = {}
    # ---- names
    xc, sc = X["nm_core"].to_list(), S["nm_core"].to_list()
    f["nm_ratio"] = _cp(X["nm"].to_list(), S["nm"].to_list(), fuzz.ratio)
    f["core_ratio"] = _cp(xc, sc, fuzz.ratio)
    f["core_tsort"] = _cp(xc, sc, fuzz.token_sort_ratio)
    f["core_tset"] = _cp(xc, sc, fuzz.token_set_ratio)
    f["core_partial"] = _cp(xc, sc, fuzz.partial_ratio)
    f["core_jw"] = _cp(xc, sc, distance.JaroWinkler.normalized_similarity)
    f["core_eq"] = (X["nm_core"] == S["nm_core"]).to_numpy().astype(np.float32)
    has_alt = (X["nm_alt"] != "").to_numpy()
    f["alt_ratio"] = np.where(has_alt, _cp(X["nm_alt"].to_list(), sc, fuzz.token_sort_ratio), -1).astype(np.float32)
    # URL / glued names vs glued, initials+last, initials forms of the S1 core
    g, il, ini = _initials_forms(S["nm_core"])
    xg = X["nm_core"].str.replace_all(" ", "").to_list()
    f["glued_ratio"] = np.maximum(_cp(xg, g.to_list(), fuzz.ratio), _cp(xg, il.to_list(), fuzz.ratio))
    f["glued_partial"] = _cp(xg, g.to_list(), fuzz.partial_ratio)
    f["x_is_url"] = (X["nm_url"] != "").to_numpy().astype(np.float32)
    f["x_initials_eq"] = (X["nm_core"].str.replace_all(" ", "") == ini).to_numpy().astype(np.float32)
    f.update(ctx.name_sp.pair(xi, si, "ntok"))
    # legal form agreement
    xl, sl = X["nm_legal"], S["nm_legal"]
    f["legal_x_has"] = (xl != "").to_numpy().astype(np.float32)
    f["legal_s_has"] = (sl != "").to_numpy().astype(np.float32)
    f["legal_eq"] = ((xl == sl) & (xl != "")).to_numpy().astype(np.float32)
    f["legal_ratio"] = _cp(xl.to_list(), sl.to_list(), fuzz.token_set_ratio)
    f["x_core_len"] = X["nm_core"].str.len_chars().to_numpy().astype(np.float32)
    f["s_core_len"] = S["nm_core"].str.len_chars().to_numpy().astype(np.float32)
    f["x_indic"] = X["nm_indic"].to_numpy().astype(np.float32)
    f["s1_core_freq"] = ctx.s1_core_freq[si]
    f["x_core_freq_in_s1"] = ctx.x_core_freq_in_s1[xi]
    # ---- addresses
    xa, sa = X["ad"].to_list(), S["ad"].to_list()
    x_empty = (X["ad"] == "").to_numpy()
    s_empty = (S["ad"] == "").to_numpy()
    f["ad_x_empty"] = x_empty.astype(np.float32)
    f["ad_s_empty"] = s_empty.astype(np.float32)
    for nm_, sc_ in (("ad_ratio", fuzz.ratio), ("ad_tsort", fuzz.token_sort_ratio),
                     ("ad_tset", fuzz.token_set_ratio), ("ad_partial", fuzz.partial_ratio)):
        f[nm_] = np.where(x_empty | s_empty, -1, _cp(xa, sa, sc_)).astype(np.float32)
    f.update(ctx.addr_sp.pair(xi, si, "atok"))
    f.update(ctx.num_sp.pair(xi, si, "ntk"))
    f.update(_num_feats(X["ad_nums"], S["ad_nums"]))
    xs, ss = X["ad_state"], S["ad_state"]
    f["state_eq"] = np.where((xs == "").to_numpy() | (ss == "").to_numpy(), -1,
                             (xs == ss).to_numpy()).astype(np.float32)
    f["x_box"] = (X["ad_box"] != "").to_numpy().astype(np.float32)
    f["s1_ad_freq"] = ctx.s1_ad_freq[si]
    f["x_ad_len"] = X["ad"].str.len_chars().to_numpy().astype(np.float32)
    f["s_ad_len"] = S["ad"].str.len_chars().to_numpy().astype(np.float32)
    f["src"] = X["src"].to_numpy().astype(np.float32)
    return pairs.with_columns([pl.Series(k, np.asarray(v, dtype=np.float32)) for k, v in f.items()])
