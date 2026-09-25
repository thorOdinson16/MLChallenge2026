import pandas as pd
import numpy as np
import re

pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 70)
pd.set_option("display.max_columns", 20)

D = "dataset"


def rd(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def section(t):
    print("\n" + "=" * 20 + " " + t + " " + "=" * 20)


# ---------- load ----------
tr = {i: rd(f"{D}/train/train_source{i}.tsv") for i in (1, 2, 3)}
te = {i: rd(f"{D}/test/test_source{i}.tsv") for i in (1, 2, 3)}
gt = rd(f"{D}/train/train_ground_truth.tsv")

# ---------- 1. sizes ----------
section("1. ROW COUNTS")
for i in (1, 2, 3):
    print(f"train S{i}: {len(tr[i]):>10,}    test S{i}: {len(te[i]):>10,}")
print(f"ground truth rows: {len(gt):,}")

# ---------- 2. countries ----------
section("2. COUNTRY COUNTS")
for name, dd in (("train", tr), ("test", te)):
    for i in (1, 2, 3):
        print(f"{name} S{i}:", dd[i].country.value_counts().to_dict())

# ---------- 3. missing / empty ----------
section("3. EMPTY RATE PER COLUMN (%)")
for name, dd in (("train", tr), ("test", te)):
    for i in (1, 2, 3):
        df = dd[i]
        rates = {c: round(100 * (df[c].str.strip() == "").mean(), 2) for c in df.columns}
        print(f"{name} S{i}:", rates)

# ---------- 4. ground truth stats ----------
section("4. GROUND TRUTH STATS")
gt["ids"] = gt.matched_entity_ids.map(lambda x: [t for t in x.split(",") if t.strip()])
gt["n"] = gt.ids.map(len)
gt["n2"] = gt.ids.map(lambda l: sum(t.startswith("S2-") for t in l))
gt["n3"] = gt.ids.map(lambda l: sum(t.startswith("S3-") for t in l))
print(f"S1 entities in GT: {len(gt):,}  (train S1 rows: {len(tr[1]):,})")
print(f"singleton rate: {100 * (gt.n == 0).mean():.2f}%")
print("matches per S1 (total):")
print(gt.n.value_counts().sort_index().head(15).to_string())
print("matches per S1 from S2:", gt.n2.value_counts().sort_index().head(8).to_dict())
print("matches per S1 from S3:", gt.n3.value_counts().sort_index().head(8).to_dict())
print(f"total matched S2 ids: {gt.n2.sum():,} / {len(tr[2]):,} S2 rows  ({100 * gt.n2.sum() / len(tr[2]):.1f}% of S2 are matched)")
print(f"total matched S3 ids: {gt.n3.sum():,} / {len(tr[3]):,} S3 rows  ({100 * gt.n3.sum() / len(tr[3]):.1f}% of S3 are matched)")

# does each S2/S3 id map to at most one S1?
allids = pd.Series([t for l in gt.ids for t in l])
print(f"duplicate S2/S3 ids across S1 groups: {allids.duplicated().sum():,}")
print(f"GT ids missing from train S2/S3 files: "
      f"{(~allids.isin(set(tr[2].entity_id) | set(tr[3].entity_id))).sum():,}")

# singleton rate and match counts by country
c1 = tr[1].set_index("entity_id").country
gt["country"] = gt.source1_entity_id.map(c1)
print("\nby country (singleton %, mean matches):")
print(gt.groupby("country").agg(rows=("n", "size"),
                                singleton_pct=("n", lambda x: round(100 * (x == 0).mean(), 2)),
                                mean_matches=("n", "mean")).to_string())

# ---------- 5. duplicates / structure ----------
section("5. DUPLICATE NAMES")
for i in (1, 2, 3):
    df = tr[i]
    print(f"S{i}: unique names {df.business_name.nunique():,}/{len(df):,}  "
          f"exact (name,addr) dups: {df.duplicated(['business_name', 'business_address']).sum():,}")

# ---------- 6. length stats ----------
section("6. NAME / ADDRESS LENGTH (chars)")
for i in (1, 2, 3):
    df = tr[i]
    print(f"S{i} name len:", df.business_name.str.len().describe()[["mean", "50%", "max"]].round(1).to_dict(),
          " addr len:", df.business_address.str.len().describe()[["mean", "50%", "max"]].round(1).to_dict())

# ---------- 7. non-ascii / postal patterns ----------
section("7. NON-ASCII AND POSTAL CODE PRESENCE (train, % of rows)")
for i in (1, 2, 3):
    df = tr[i]
    for c in ("India", "US"):
        s = df[df.country == c]
        if len(s) == 0:
            continue
        na = 100 * s.business_name.map(lambda x: not x.isascii()).mean()
        pin = 100 * s.business_address.str.contains(r"\b\d{6}\b").mean() if c == "India" else \
            100 * s.business_address.str.contains(r"\b\d{5}\b").mean()
        print(f"S{i} {c}: non-ascii name {na:.2f}%   postal code in address {pin:.1f}%")
for i in (1, 2, 3):
    s = te[i][te[i].country == "France"]
    if len(s):
        na = 100 * s.business_name.map(lambda x: not x.isascii()).mean()
        pc = 100 * s.business_address.str.contains(r"\b\d{5}\b").mean()
        print(f"TEST S{i} France: rows {len(s):,}  non-ascii name {na:.2f}%  5-digit code in addr {pc:.1f}%")

# ---------- 8. sample matched groups ----------
section("8. SAMPLE MATCHED GROUPS (train)")
lut = {}
for i in (1, 2, 3):
    for r in tr[i].itertuples(index=False):
        lut[r.entity_id] = (r.business_name, r.business_address, r.country)

rng = np.random.default_rng(7)
matched = gt[gt.n > 0]
for cty in sorted(matched.country.unique()):
    sub = matched[matched.country == cty]
    picks = sub.sample(min(8, len(sub)), random_state=7)
    print(f"\n----- {cty} -----")
    for r in picks.itertuples():
        n, a, _ = lut[r.source1_entity_id]
        print(f"\n[{r.source1_entity_id}] {n} | {a}")
        for m in r.ids[:5]:
            mn, ma, _ = lut[m]
            print(f"    -> {m}: {mn} | {ma}")

# ---------- 9. sample singletons ----------
section("9. SAMPLE SINGLETONS (train)")
single = gt[gt.n == 0]
for r in single.sample(min(10, len(single)), random_state=7).itertuples():
    n, a, c = lut[r.source1_entity_id]
    print(f"[{r.source1_entity_id}] {c} | {n} | {a}")

# ---------- 10. unmatched S2/S3 (distractors) ----------
section("10. UNMATCHED S2/S3 ROWS (distractors), sample")
matched_ids = set(allids)
for i in (2, 3):
    un = tr[i][~tr[i].entity_id.isin(matched_ids)]
    print(f"\nS{i}: {len(un):,} unmatched rows")
    for r in un.sample(min(6, len(un)), random_state=7).itertuples(index=False):
        print(f"  {r.entity_id} | {r.country} | {r.business_name} | {r.business_address}")

# ---------- 11. same-name S1 collisions (hard negatives) ----------
section("11. S1 ROWS SHARING A NAME WITH ANOTHER S1 ROW (hard negatives)")
s1 = tr[1].copy()
s1["nn"] = s1.business_name.str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.strip()
dup = s1[s1.duplicated("nn", keep=False)].sort_values("nn")
print(f"S1 rows sharing a normalized name: {len(dup):,} / {len(s1):,}")
print(dup.head(12)[["entity_id", "business_name", "business_address", "country"]].to_string(index=False))

# ---------- 12. S2-S3 co-occurrence ----------
section("12. GROUPS WITH BOTH S2 AND S3 MATCHES")
both = ((gt.n2 > 0) & (gt.n3 > 0)).sum()
print(f"S1 groups with >=1 S2 and >=1 S3 match: {both:,} ({100 * both / len(gt):.1f}%)")