"""Dense retrieval: embeddings (LSA or bi-encoder) + exact brute-force top-k on GPU."""
import numpy as np
import polars as pl
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def gpu_topk(Q: np.ndarray, D: np.ndarray, k: int, q_batch: int = 4096, d_chunk: int = 200_000):
    """Exact inner-product top-k. Q (nq, d), D (nd, d) float16/32, rows L2-normalised.

    Returns (idx uint32 [nq, k], score float32 [nq, k]); fewer than k DB rows -> padded with -inf."""
    k_eff = min(k, D.shape[0])
    Dg = [torch.from_numpy(D[i:i + d_chunk]).to(DEVICE, torch.float16) for i in range(0, D.shape[0], d_chunk)]
    offs = np.cumsum([0] + [c.shape[0] for c in Dg])
    out_i = np.empty((Q.shape[0], k_eff), dtype=np.uint32)
    out_s = np.empty((Q.shape[0], k_eff), dtype=np.float32)
    with torch.inference_mode():
        for i in range(0, Q.shape[0], q_batch):
            qb = torch.from_numpy(Q[i:i + q_batch]).to(DEVICE, torch.float16)
            best_s, best_i = None, None
            for c, off in zip(Dg, offs[:-1]):
                s = qb @ c.T
                kk = min(k_eff, c.shape[0])
                v, ix = torch.topk(s, kk, dim=1)
                ix = ix + int(off)
                if best_s is None:
                    best_s, best_i = v, ix
                else:
                    cs = torch.cat([best_s, v], 1)
                    ci = torch.cat([best_i, ix], 1)
                    v2, j = torch.topk(cs, k_eff, dim=1)
                    best_s, best_i = v2, torch.gather(ci, 1, j)
            out_i[i:i + qb.shape[0]] = best_i.cpu().numpy().astype(np.uint32)
            out_s[i:i + qb.shape[0]] = best_s.float().cpu().numpy()
    del Dg
    torch.cuda.empty_cache()
    return out_i, out_s


def topk_frame(q_idx: np.ndarray, db_idx: np.ndarray, I: np.ndarray, S: np.ndarray) -> pl.DataFrame:
    """Turn top-k matrices into a long frame (x, s1, score, rank) with global ids."""
    nq, k = I.shape
    return pl.DataFrame({
        "x": np.repeat(q_idx.astype(np.uint32), k),
        "s1": db_idx[I.ravel()].astype(np.uint32),
        "score": S.ravel(),
        "rank": np.tile(np.arange(k, dtype=np.uint8), nq),
    })


class LSA:
    """Char n-gram TF-IDF -> TruncatedSVD. Unsupervised: fit on the data being matched (any country)."""

    def __init__(self, dim=256, ngram=(2, 4), max_features=200_000, fit_rows=400_000, seed=0):
        self.dim, self.ngram, self.max_features, self.fit_rows, self.seed = dim, ngram, max_features, fit_rows, seed

    def fit(self, texts: list):
        rng = np.random.default_rng(self.seed)
        sample = [texts[i] for i in rng.choice(len(texts), min(self.fit_rows, len(texts)), replace=False)]
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=self.ngram, min_df=3, max_df=0.5,
                                   sublinear_tf=True, dtype=np.float32, lowercase=False,
                                   max_features=self.max_features)
        X = self.vec.fit_transform(sample)
        self.svd = TruncatedSVD(self.dim, algorithm="randomized", n_iter=4, random_state=self.seed).fit(X)
        self.comp = self.svd.components_.T.astype(np.float32)        # (V, dim)
        return self

    def transform(self, texts: list, chunk=500_000) -> np.ndarray:
        out = np.empty((len(texts), self.dim), dtype=np.float16)
        for i in range(0, len(texts), chunk):
            E = self.vec.transform(texts[i:i + chunk]) @ self.comp
            n = np.linalg.norm(E, axis=1, keepdims=True)
            out[i:i + chunk] = (E / np.maximum(n, 1e-6)).astype(np.float16)
        return out
