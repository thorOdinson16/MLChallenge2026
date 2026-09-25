import re
import unicodedata
import numpy as np
import pandas as pd
from collections import Counter

pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 80)

D = "dataset"
SEED = 7
N_SUB = 100_000


def rd(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def section(t):
    print("\n" + "=" * 20 + " " + t + " " + "=" * 20)


def strip_acc(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def norm(s):
    s = strip_acc(s.lower())
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(s.split())


def script_of(ch):
    o = ord(ch)
    if o < 128:
        return "latin"
    if 0x0900 <= o <= 0x097F: return "devanagari"
    if 0x0980 <= o <= 0x09FF: return "bengali"
    if 0x0A00 <= o <= 0x0A7F: return "gurmukhi"
    if 0x0A80 <= o <= 0x0AFF: return "gujarati"
    if 0x0B00 <= o <= 0x0B7F: return "odia"
    if 0x0B80 <= o <= 0x0BFF: return "tamil"
    if 0x0C00 <= o <= 0x0C7F: return "telugu"
    if 0x0C80 <= o <= 0x0CFF: return "kannada"
    if 0x0D00 <= o <= 0x0D7F: return "malayalam"
    if 0x00C0 <= o <= 0x024F: return "latin-ext(accents)"
    if 0x0400 <= o <= 0x04FF: return "cyrillic"
    if 0x0600 <= o <= 0x06FF: return "arabic"
    if 0x4E00 <= o <= 0x9FFF: return "cjk"
    return "other"


def scripts(s):
    c = Counter(script_of(ch) for ch in s if not ch.isspace() and ch.isalpha())
    return c


# ---------- load ----------
tr1 = rd(f"{D}/train/train_source1.tsv")
tr2 = rd(f"{D}/train/train_source2.tsv")
tr3 = rd(f"{D}/train/train_source3.tsv")
gt = rd(f"{D}/train/train_ground_truth.tsv")
te1 = rd(f"{D}/test/test_source1.tsv")
te2 = rd(f"{D}/test/test_source2.tsv")
te3 = rd(f"{D}/test/test_source3.tsv")

lut = {}
for df in (tr1, tr2, tr3):
    for r in df.itertuples(index=False):
        lut[r.entity_id] = (r.business_name, r.business_address, r.country)

# ============================================================
section("A. FRANCE: raw samples (test)")
fr1 = te1[te1.country == "France"].sample(15, random_state=SEED)
for r in fr1.itertuples(index=False):
    print(f"S1 | {r.business_name} | {r.business_address}")
for name, df in (("S2", te2), ("S3", te3)):
    print(f"\n-- {name} France sample --")
    f = df[df.country == "France"].sample(15, random_state=SEED)
    for r in f.itertuples(index=False):
        print(f"{name} | {r.business_name} | {r.business_address}")

# ============================================================
section("B. SCRIPTS IN NAMES: share of rows containing each script")
def script_table(df, label):
    cnt = Counter()
    n = len(df)
    for s in df.business_name.values:
        for sc in set(scripts(s)):
            cnt[sc] += 1
    print(f"{label} (n={n:,}):", {k: f"{100*v/n:.2f}%" for k, v in cnt.most_common(8)})

sample = lambda df, k=200_000: df.sample(min(k, len(df)), random_state=SEED)
for cty in ("US", "India"):
    for nm, df in (("train S1", tr1), ("train S2", tr2), ("train S3", tr3)):
        script_table(sample(df[df.country == cty]), f"{nm} {cty}")
for nm, df in (("test S1", te1), ("test S2", te2), ("test S3", te3)):
    script_table(sample(df[df.country == "France"]), f"{nm} France")

print("\nFrance accent chars in S1 names (top 12):")
c = Counter()
for s in sample(te1[te1.country == "France"], 100_000).business_name.values:
    for ch in s:
        if not ch.isascii():
            c[ch] += 1
print(c.most_common(12))

print("\nFrance: names that are fully non-latin in S2 (sample):")
fr2 = sample(te2[te2.country == "France"], 100_000)
nonlat = fr2[fr2.business_name.map(lambda s: sum(v for k, v in scripts(s).items() if k not in ("latin", "latin-ext(accents)")) > 0)]
print(f"rows with any non-latin script: {len(nonlat):,} / {len(fr2):,}")
print(nonlat.head(8)[["business_name", "business_address"]].to_string(index=False))

# ============================================================
section("C. BLOCKING PROBE on 100k train S1 (recall of simple keys)")
gt["ids"] = gt.matched_entity_ids.map(lambda x: [t for t in x.split(",") if t.strip()])
sub = gt.sample(N_SUB, random_state=SEED)
sub = sub[sub.ids.map(len) > 0]
pairs = []
for r in sub.itertuples():
    s1n, s1a, cty = lut[r.source1_entity_id]
    for m in r.ids:
        mn, ma, _ = lut[m]
        pairs.append((r.source1_entity_id, m, s1n, s1a, mn, ma, cty))
P = pd.DataFrame(pairs, columns=["s1", "m", "s1n", "s1a", "mn", "ma", "country"])
P["src"] = P.m.str[:2]
print(f"true pairs in probe: {len(P):,}")

def toks(s):
    return set(norm(s).split())

def nums(s):
    return set(re.findall(r"\d+", s))

STOP = {"road", "street", "st", "rd", "ave", "avenue", "drive", "dr", "lane", "ln", "boulevard", "blvd",
        "no", "floor", "unit", "the", "of", "and", "near", "opp", "opposite", "saint", "nagar", "new"}

def addr_tok(s):
    return {t for t in norm(s).split() if t not in STOP and len(t) > 2 and not t.isdigit()}

def first_tok(s):
    t = norm(s).split()
    return t[0] if t else ""

rows = []
for r in P.itertuples(index=False):
    a1, a2 = addr_tok(r.s1a), addr_tok(r.ma)
    n1, n2 = nums(r.s1a), nums(r.ma)
    nm1, nm2 = toks(r.s1n), toks(r.mn)
    rows.append(dict(
        country=r.country, src=r.src,
        addr_empty=(r.ma.strip() == ""),
        addr_tok_shared=len(a1 & a2) > 0,
        addr_tok_shared2=len(a1 & a2) > 1,
        num_shared=len(n1 & n2) > 0,
        name_tok_shared=len(nm1 & nm2) > 0,
        name_latin=r.mn.isascii(),
        any_key=(len(a1 & a2) > 0) or (len(n1 & n2) > 0) or (len(nm1 & nm2) > 0),
        addr_or_num=(len(a1 & a2) > 0) or (len(n1 & n2) > 0),
        name_first_same=first_tok(r.s1n) == first_tok(r.mn),
    ))
R = pd.DataFrame(rows)
print("\n% of true pairs sharing a key, by country x source:")
print((R.groupby(["country", "src"]).mean(numeric_only=True) * 100).round(1).T.to_string())
print("\nOverall:")
print((R.mean(numeric_only=True) * 100).round(1).to_string())
print("\nWhen name is NON-latin (script mismatch), % of pairs with shared addr token / number:")
nl = R[~R.name_latin]
print(f"rows: {len(nl):,}  addr_tok_shared {100*nl.addr_tok_shared.mean():.1f}%  num_shared {100*nl.num_shared.mean():.1f}%  addr_or_num {100*nl.addr_or_num.mean():.1f}%  addr_empty {100*nl.addr_empty.mean():.1f}%")

# ============================================================
section("D. SIMILARITY DISTRIBUTIONS: true pairs vs random same-country pairs")
from rapidfuzz import fuzz

rng = np.random.default_rng(SEED)
K = min(20000, len(P))
T = P.sample(K, random_state=SEED)
same = T.name_latin = T.mn.map(lambda s: s.isascii())
Tl = T[same]

def sims(a, b):
    return fuzz.token_set_ratio(norm(a), norm(b))

t_name = [sims(a, b) for a, b in zip(Tl.s1n, Tl.mn)]
t_addr = [sims(a, b) for a, b in zip(Tl.s1a, Tl.ma) if b.strip() != ""]
# random negatives: same country, random other record from same source
neg_name, neg_addr = [], []
for r in Tl.itertuples():
    pool = tr2 if r.src == "S2" else tr3
    j = rng.integers(0, len(pool))
    row = pool.iloc[j]
    neg_name.append(sims(r.s1n, row.business_name))
    if row.business_address.strip():
        neg_addr.append(sims(r.s1a, row.business_address))

q = [1, 5, 25, 50, 75]
print(f"(latin-name true pairs only, n={len(Tl):,})")
print("name token_set  TRUE :", np.percentile(t_name, q).round(0))
print("name token_set  RAND :", np.percentile(neg_name, q).round(0))
print("addr token_set  TRUE :", np.percentile(t_addr, q).round(0))
print("addr token_set  RAND :", np.percentile(neg_addr, q).round(0))

# hard negative: same-name different S1 (from EDA 1): how similar are their addresses?
s1 = tr1.copy()
s1["nn"] = s1.business_name.map(norm)
g = s1[s1.duplicated("nn", keep=False)].groupby("nn")
hn = []
for k, grp in list(g)[:3000]:
    a = grp.business_address.values
    if len(a) >= 2 and a[0] and a[1]:
        hn.append(sims(a[0], a[1]))
print("addr token_set between DIFFERENT S1 entities with the SAME name:", np.percentile(hn, q).round(0), f"(n={len(hn)})")

print("\nDone.")