"""Indic-script -> Latin transliteration.

Primary: a token dictionary *learned from the training ground truth* (Indic token in an S2/S3
record aligned with the Latin token of its matched S1 record). Fallback for unseen tokens:
rule-based transliteration (indic_transliteration, MIT) followed by a light romanisation cleanup.
"""
import json
import re
from collections import Counter, defaultdict
from functools import lru_cache

import polars as pl

from . import config

INDIC_RE = re.compile(r"[ऀ-෿]")
INDIC_TOKEN_RE = re.compile(r"[ऀ-෿]+")
POLARS_INDIC = r"[ऀ-෿]"

_SCRIPTS = [  # (start, end, sanscript scheme)
    (0x0900, 0x097F, "devanagari"), (0x0980, 0x09FF, "bengali"), (0x0A00, 0x0A7F, "gurmukhi"),
    (0x0A80, 0x0AFF, "gujarati"), (0x0B00, 0x0B7F, "oriya"), (0x0B80, 0x0BFF, "tamil"),
    (0x0C00, 0x0C7F, "telugu"), (0x0C80, 0x0CFF, "kannada"), (0x0D00, 0x0D7F, "malayalam"),
]
_DICT_PATH = "translit_dict.json"
_TOKEN_SPLIT = re.compile(r"[\s,;:()\[\]{}\"'|/\\]+")


def _latin_tokens(s: str):
    return [t for t in re.split(r"[^a-z0-9]+", s.lower()) if t]


def _raw_tokens(s: str):
    return [t.strip(".-") for t in _TOKEN_SPLIT.split(s) if t.strip(".-")]


def learn(max_pairs: int = 3_000_000) -> dict:
    """Learn Indic token -> Latin token from train ground-truth pairs (India only)."""
    from . import io
    gt = io.load("train", "gt").sample(n=min(max_pairs, 7_638_365), seed=config.SEED)
    s1 = io.load("train", "s1", ["idx", "name", "addr", "country"]).filter(pl.col("country") == "India")
    x = io.load("train", "x", ["idx", "name", "addr"]).filter(
        pl.col("name").str.contains(POLARS_INDIC) | pl.col("addr").str.contains(POLARS_INDIC))
    p = (gt.join(s1.rename({"idx": "s1", "name": "n1", "addr": "a1"}), on="s1")
           .join(x.rename({"idx": "x", "name": "nx", "addr": "ax"}), on="x"))
    print("pairs with indic text:", p.height)

    pos = defaultdict(Counter)        # positional alignment counts (names)
    co = defaultdict(Counter)         # co-occurrence counts (names + addresses)
    tok_n = Counter()
    lat_df = Counter()
    for n1, a1, nx, ax in p.select("n1", "a1", "nx", "ax").iter_rows():
        for is_name, lat_src, ind_src in ((True, n1, nx), (False, a1, ax)):
            if not INDIC_RE.search(ind_src):
                continue
            lt = _latin_tokens(lat_src)
            it = _raw_tokens(ind_src)
            if is_name and len(lt) == len(it):
                for a, b in zip(it, lt):
                    if INDIC_RE.search(a) and not INDIC_RE.search(b):
                        pos[a][b] += 1
            lset = set(lt)
            lat_df.update(lset)
            for a in set(it):
                if INDIC_RE.search(a):
                    tok_n[a] += 1
                    co[a].update(lset)
    out = {}
    for a, n in tok_n.items():
        if pos[a]:
            b, c = pos[a].most_common(1)[0]
            if c >= 2 and c / sum(pos[a].values()) >= 0.5:
                out[a] = b
                continue
        # co-occurrence with lift: the Latin token that almost always accompanies `a`
        best, best_score = None, 0.0
        for b, c in co[a].most_common(8):
            pb = c / n
            lift = pb / (lat_df[b] / max(len(p), 1))
            if pb >= 0.6 and lift > 3 and pb * min(lift, 50) > best_score:
                best, best_score = b, pb * min(lift, 50)
        if best is not None and n >= 2:
            out[a] = best
    print("learned tokens:", len(out))
    with open(config.path(_DICT_PATH), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    return out


def _skeleton(s: str) -> str:
    """Crude phonetic key so romanisations like 'bekari' meet vocabulary words like 'bakery'."""
    s = s.lower()
    for a, b in (("ph", "f"), ("bh", "b"), ("dh", "d"), ("th", "t"), ("kh", "k"), ("gh", "g"), ("sh", "s"),
                 ("ch", "c"), ("ck", "k"), ("q", "k"), ("x", "ks"), ("z", "j"), ("w", "v"), ("y", "i")):
        s = s.replace(a, b)
    s = s[:1] + re.sub(r"[aeiou]", "", s[1:])
    return re.sub(r"(.)\1+", r"\1", s)


def resolve_unseen(learned: dict) -> dict:
    """Map Indic tokens missing from the learned dictionary to the closest Latin vocabulary word.

    Vocabulary = Latin tokens of India S1 names (train + test; S1 is always Latin). Unsupervised,
    uses no labels, so it also applies to test-only words."""
    from rapidfuzz import process, fuzz
    from . import io
    vocab = Counter()
    for split in config.SPLITS:
        s1 = io.load(split, "s1", ["name", "country"]).filter(pl.col("country") == "India")
        for n in s1["name"]:
            vocab.update(_latin_tokens(n))
    words = [w for w, c in vocab.items() if c >= 3 and len(w) >= 3 and w.isalpha()]
    keys = [_skeleton(w) for w in words]
    types = Counter()
    for split in config.SPLITS:
        x = io.load(split, "x", ["name"]).filter(pl.col("name").str.contains(POLARS_INDIC))
        for n in x["name"]:
            types.update(t for t in _raw_tokens(n) if INDIC_RE.search(t))
    out = {}
    for t, _ in types.items():
        if t in learned or t.rstrip(".") in learned:
            continue
        rom = fallback(t)
        if len(rom) < 3:
            continue
        k = _skeleton(rom)
        best = process.extractOne(k, keys, scorer=fuzz.ratio, score_cutoff=80)
        if best is not None:
            cand = words[best[2]]
            if fuzz.ratio(rom, cand) >= 65 and len(cand) >= 4:
                out[t] = cand
    return out


def build() -> dict:
    learned = learn(max_pairs=10_000_000)
    resolved = resolve_unseen(learned)
    print("resolved unseen tokens:", len(resolved), list(resolved.items())[:25])
    full = {**resolved, **learned}
    with open(config.path(_DICT_PATH), "w", encoding="utf-8") as f:
        json.dump(full, f, ensure_ascii=False)
    _dict.cache_clear()
    return full


@lru_cache(maxsize=1)
def _dict() -> dict:
    p = config.path(_DICT_PATH)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _scheme(tok: str):
    o = ord(INDIC_RE.search(tok).group(0))
    for lo, hi, s in _SCRIPTS:
        if lo <= o <= hi:
            return s
    return None


_CLEAN = [(re.compile(p), r) for p, r in [
    (r"[^a-z ]", ""), (r"aa+", "a"), (r"ii+", "i"), (r"uu+", "u"), (r"ee+", "e"), (r"oo+", "o"),
    (r"([bcdfghjklmnpqrstvwxyz])a\b", r"\1"),  # drop inherent schwa at word end
]]


@lru_cache(maxsize=500_000)
def fallback(tok: str) -> str:
    from indic_transliteration import sanscript
    sch = _scheme(tok)
    if sch is None:
        return ""
    try:
        s = sanscript.transliterate(tok, sch, sanscript.HK).replace("M", "n").lower()
    except Exception:
        return ""
    for rx, r in _CLEAN:
        s = rx.sub(r, s)
    return s


def to_latin(text: str) -> str:
    """Replace every Indic token in `text` with its Latin form."""
    if not text or not INDIC_RE.search(text):
        return text
    d = _dict()

    def rep(m):
        t = m.group(0)
        return d.get(t) or d.get(t.rstrip(".")) or fallback(t)
    return INDIC_TOKEN_RE.sub(rep, text)


def coverage(split: str = "test") -> None:
    """Report share of Indic tokens (by occurrence) covered by the learned dictionary."""
    from . import io
    x = io.load(split, "x", ["name", "addr"]).filter(
        pl.col("name").str.contains(POLARS_INDIC) | pl.col("addr").str.contains(POLARS_INDIC)).sample(200_000, seed=1)
    d = _dict()
    for col in ("name", "addr"):
        c = Counter(t for s in x[col] for t in INDIC_TOKEN_RE.findall(s))
        hit = sum(v for k, v in c.items() if k in d)
        print(f"{split} {col}: indic tokens {sum(c.values())}, types {len(c)}, covered {hit / max(sum(c.values()), 1):.4f}")
        miss = [k for k, _ in c.most_common() if k not in d][:15]
        print("  top misses:", [(m, fallback(m)) for m in miss])


if __name__ == "__main__":
    import sys
    if "--coverage" not in sys.argv:
        build()
    coverage("train")
    coverage("test")
