"""Evaluation utilities: macro F0.5 (with singleton rule) and blocking diagnostics."""
from typing import Dict, Set, Iterable


def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision == 0 and recall == 0:
        return 0.0
    b2 = beta * beta
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom > 0 else 0.0


def entity_f05(pred: Set[str], truth: Set[str]) -> float:
    if not pred and not truth:
        return 1.0          # correctly predicted singleton
    if not pred or not truth:
        return 0.0          # false merge on singleton, or missed everything
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    return f_beta(tp / len(pred), tp / len(truth))


def macro_f05(pred: Dict[str, Set[str]], truth: Dict[str, Set[str]], s1_ids: Iterable[str]) -> float:
    total, n = 0.0, 0
    for s in s1_ids:
        total += entity_f05(pred.get(s, set()), truth.get(s, set()))
        n += 1
    return total / max(n, 1)


def blocking_recall(cands: Dict[str, Set[str]], truth: Dict[str, Set[str]], s1_ids: Iterable[str]):
    """Returns (pair_recall, entity_full_recall, avg_candidates_per_s1)."""
    hit = tot = full = ents = ncand = 0
    for s in s1_ids:
        t = truth.get(s, set())
        c = cands.get(s, set())
        ncand += len(c)
        ents += 1
        if not t:
            continue
        h = len(t & c)
        hit += h
        tot += len(t)
        full += int(h == len(t))
    n_nonsingle = sum(1 for s in s1_ids if truth.get(s))
    return (hit / max(tot, 1), full / max(n_nonsingle, 1), ncand / max(ents, 1))


def load_truth(gt_df) -> Dict[str, Set[str]]:
    out = {}
    for r in gt_df.itertuples(index=False):
        ids = [t for t in str(r.matched_entity_ids).split(",") if t.strip()]
        out[r.source1_entity_id] = set(ids)
    return out


if __name__ == "__main__":
    # self-test: reproduces the 0.714 example from the problem statement
    pred = {"S2-00047", "S2-00193", "S3-00812"}
    truth = {"S2-00047", "S3-00812"}
    print(round(entity_f05(pred, truth), 3), "(expected 0.714)")
    assert abs(entity_f05(pred, truth) - 0.714) < 0.001
    assert entity_f05(set(), set()) == 1.0
    assert entity_f05({"S2-1"}, set()) == 0.0
    print("eval self-test OK")