"""Turn pair probabilities into final matches.

1. Uniqueness: every X keeps at most its single best S1 (each S2/S3 record belongs to <= 1 entity).
2. Per S1, choose the prefix of its candidates (sorted by probability) that maximises the *expected*
   F0.5 under independent Bernoulli match probabilities, the empty set included. With tp true positives
   among k selected and u true matches left unselected:  F0.5 = 1.25*tp / (0.25*(tp+u) + k).
"""
import numpy as np
import polars as pl


def _pb(p: np.ndarray) -> np.ndarray:
    """Poisson-binomial pmf of the number of successes."""
    d = np.zeros(len(p) + 1)
    d[0] = 1.0
    for i, q in enumerate(p):
        d[1:i + 2] = d[1:i + 2] * (1 - q) + d[0:i + 1] * q
        d[0] *= (1 - q)
    return d


def best_prefix(p: np.ndarray, miss: float = 0.0) -> tuple[int, float]:
    """p sorted descending. `miss`: prob. that an extra true match exists outside the candidates.
    Returns (k, expected F0.5)."""
    n = len(p)
    ext = np.append(p, miss) if miss > 0 else p
    best_k, best_e = 0, float(np.prod(1 - ext))
    if n == 0:
        return best_k, best_e
    # suffix distributions of unselected positives
    suf = [None] * (n + 1)
    suf[n] = _pb(ext[n:])
    for k in range(n - 1, 0, -1):
        suf[k] = _pb(ext[k:])
    pre = np.array([1.0])
    for k in range(1, n + 1):
        q = p[k - 1]
        new = np.zeros(k + 1)
        new[:k] += pre * (1 - q)
        new[1:] += pre * q
        pre = new
        a = np.arange(k + 1)[:, None]
        b = np.arange(len(suf[k]))[None, :]
        e = float((pre[:, None] * suf[k][None, :] * (1.25 * a / (0.25 * (a + b) + k))).sum())
        if e > best_e:
            best_k, best_e = k, e
    return best_k, best_e


def assign_unique(scored: pl.DataFrame, prob_col: str = "p") -> pl.DataFrame:
    """Keep each x's single highest-probability s1."""
    return scored.sort(prob_col, descending=True).unique("x", keep="first")


def decode(scored: pl.DataFrame, prob_col: str = "p", min_p: float = 0.02, miss: float = 0.0,
           max_cands: int = 30) -> pl.DataFrame:
    """scored: (x, s1, p). Returns selected (s1, x) pairs."""
    a = assign_unique(scored, prob_col).filter(pl.col(prob_col) >= min_p)
    a = a.sort(["s1", prob_col], descending=[False, True])
    s1 = a["s1"].to_numpy()
    xs = a["x"].to_numpy()
    ps = a[prob_col].to_numpy().astype(np.float64)
    bounds = np.flatnonzero(np.diff(s1)) + 1
    starts = np.r_[0, bounds]
    ends = np.r_[bounds, len(s1)]
    keep = np.zeros(len(s1), dtype=bool)
    for st, en in zip(starts, ends):
        p = ps[st:min(en, st + max_cands)]
        k, _ = best_prefix(p, miss)
        keep[st:st + k] = True
    return pl.DataFrame({"s1": s1[keep], "x": xs[keep]}).with_columns(pl.col("s1").cast(pl.UInt32),
                                                                         pl.col("x").cast(pl.UInt32))


def decode_threshold(scored: pl.DataFrame, thr: float, prob_col: str = "p") -> pl.DataFrame:
    """Baseline decoder: unique assignment + global threshold."""
    return assign_unique(scored, prob_col).filter(pl.col(prob_col) >= thr).select("s1", "x")
