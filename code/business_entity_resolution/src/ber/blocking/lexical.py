"""Sparse TF-IDF top-k retrieval (query records -> top-k S1 records), chunked and multi-threaded."""
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

VIEWS = {
    # name: (column builder, analyzer, ngram_range, max_df)
    "name_c3": (lambda d: d["nm_core"], "char_wb", (3, 3), 0.5),
    "addr_w": (lambda d: d["ad"], "word", (1, 1), 0.3),
    "addr_c3": (lambda d: d["ad"], "char_wb", (3, 3), 0.3),
    "full_c3": (lambda d: d["nm_core"] + " | " + d["ad"], "char_wb", (3, 3), 0.3),
}


def make_vectorizer(analyzer, ngram, max_df):
    return TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=2, max_df=max_df, sublinear_tf=True,
                           dtype=np.float32, token_pattern=r"\S+" if analyzer == "word" else None,
                           lowercase=False)


def topk(q_texts, db_texts, analyzer, ngram, max_df, k, chunk=250_000, threshold=0.05):
    """Return polars frame (q, db, score, rank) with q/db positional indices."""
    vec = make_vectorizer(analyzer, ngram, max_df)
    D = vec.fit_transform(db_texts)
    Dt = D.T.tocsr()
    out = []
    for i in range(0, len(q_texts), chunk):
        Q = vec.transform(q_texts[i:i + chunk])
        C = sp_matmul_topn(Q, Dt, top_n=k, threshold=threshold, sort=True, n_threads=-1).tocsr()
        n_per = np.diff(C.indptr)
        rows = np.repeat(np.arange(C.shape[0], dtype=np.uint32), n_per) + i
        rank = (np.arange(C.nnz) - np.repeat(C.indptr[:-1], n_per)).astype(np.uint8)
        out.append(pl.DataFrame({"q": rows, "db": C.indices.astype(np.uint32),
                                 "score": C.data.astype(np.float32), "rank": rank}))
    return pl.concat(out) if out else pl.DataFrame(schema={"q": pl.UInt32, "db": pl.UInt32,
                                                          "score": pl.Float32, "rank": pl.UInt8})


def retrieve(view: str, q: pl.DataFrame, db: pl.DataFrame, k: int, **kw) -> pl.DataFrame:
    """q, db: normalised frames of one country. Returns (x, s1, score, rank) with global idx."""
    col, analyzer, ngram, max_df = VIEWS[view]
    r = topk(col(q).to_list(), col(db).to_list(), analyzer, ngram, max_df, k, **kw)
    qi = q["idx"].to_numpy()
    di = db["idx"].to_numpy()
    return r.with_columns(x=pl.Series(qi[r["q"].to_numpy()]), s1=pl.Series(di[r["db"].to_numpy()])).select(
        "x", "s1", "score", "rank")
