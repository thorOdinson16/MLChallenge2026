# AnchorMatch architecture — Business Entity Resolution (Amazon ML Challenge 2026)

Living design doc. Every stage is a separate, cached, independently re-runnable module so
any stage can be improved without touching the others. Scores for every change go in the
experiment log at the bottom.

## 1. Facts from EDA that drive the design

| Fact | Consequence |
|---|---|
| Every S2/S3 record matches **at most one** S1 entity (0 multi-assigned in 7.6M) | Solve the problem **X-centric**: for each S2/S3 record, pick its best S1 or "none". Uniqueness becomes a hard constraint. |
| ~26% of S2/S3 records match no S1 | These are the main false-merge source; "none" must be a real option. |
| Matches never cross countries | Partition everything by `country` (open set, France included). |
| 5.6% of S1 are singletons; mean 3.5 matches/S1 (S2 1.7, S3 1.8) | Macro-F0.5 is decided mostly by precision inside multi-match clusters. |
| India: 23% (S2) / 13% (S3) names in Indic scripts; 22% of addresses have Indic state names | Transliteration dictionary learned from train pairs, plus a multilingual encoder. |
| Names: typos, char swaps, digit-for-letter (`Br0wn`), accents added, token reorder, legal-suffix churn, DBA/`t/a` prefixes, random replacement names (`Kelodovadrex`), URL forms (`aslion.com` = initials) | Many complementary name signals; address must carry records whose name is destroyed. |
| Addresses: 3–5% empty, `null`/`N/A` literals, component reorder, abbreviations, **house-number mutation** (6437→643, 175/J→275/J), PO Box/PMB/unit injection | Soft house-number features (prefix/substring/edit), not only exact equality. |
| Generic names are shared by hundreds of S1 (`primary care group` ×253); addresses shared by up to 14 S1 | Name-only or address-only evidence is ambiguous → frequency/IDF features and competitor-gap features. |
| France appears only in test; same generator style (R./Rue, All., region vs department, `St.-` vs `Saint-`, SARL/SAS/EURL/SCI, URL names) | Features must be language-agnostic after normalisation; French normalisation rules written by hand; France-specific sanity checks and optional pseudo-label adaptation. |

## 2. Pipeline

```
raw TSV ──► [A] normalise ──► [B] candidate generation ──► [C] matcher (GBDT)
                                (X → top-K S1)               │
                                                             ▼
                       output ◄── [E] decode ◄── [D] collective re-scoring
```

### [A] Normalisation (`ber/normalize.py`, `ber/translit.py`)
Per record: canonical name, core name (legal form stripped), legal-form class, DBA/alt
name split, URL stem, initials, script flags; address canonical form, house numbers,
postcode, unit, street tokens, city/state tokens, missing flags. Indic→Latin transliteration
via a token dictionary **learned from train pairs**, with a rule-based fallback.

### [B] Candidate generation — X-centric (`ber/blocking/`)
For every S2/S3 record, retrieve top-K S1 in the same country from a union of:
1. **Fine-tuned multilingual bi-encoder** (Apache-2.0 MiniLM, contrastive on train pairs), GPU brute-force top-k.
2. **Sparse lexical** TF-IDF retrieval on name char-n-grams and address tokens.
3. **Exact keys**: core name, (house number + street token), URL stem / initials.

A cheap first-stage ranker trims the union to the final `K`. That trimmed set **is**
`candidate_pairs.tsv`. Tracked metrics: recall ceiling and pairs per X.

### [C] Pairwise matcher (`ber/features.py`, `ber/train.py`)
LightGBM on roughly 60 features: name similarity (several rapidfuzz scorers, token IDF overlap,
transliterated/core/alt-name variants, legal-form agreement), address similarity (house
number exact/soft, street, city, state, unit, postcode), bi-encoder cosine, missing flags,
value frequencies, and **context features** (rank and score gap of this S1 among X's
candidates, rank of X among S1's candidates). Trained with GroupKFold over S1, out-of-fold.

### [D] Collective re-scoring
Second-stage GBDT adds cluster evidence: similarity of X to the S1's other confident
matches, the S1's confident-match count, and the best competing S1's probability. This
rescues garbled-name records and suppresses chain/shared-address confusions.

### [E] Decoding (`ber/decode.py`)
1. Uniqueness: each X keeps only its argmax S1.
2. Per S1, choose the prefix of candidates, sorted by probability, that maximises
   **expected F0.5**, with the empty set as an option. This is optimal decoding for the
   macro metric, not a global threshold.

### Validation (`ber/evaluate.py`)
Exact macro-F0.5 re-implementation, blocking recall@K, and breakdowns by country, source,
and cluster size, plus error dumps. The holdout is by S1 entity; candidate retrieval always
runs against the full S1 pool, as it does at test time.

## 3. Resource budget (16 GB RAM, RTX 4050 6 GB, disk-limited)
Polars and parquet (zstd) throughout. Process per country and in X-chunks. Store embeddings
as fp16. Never hold all pairs × features in memory; stream chunks into LightGBM datasets.

## 4. Compliance
No external lookups. Only provided data plus hand-written normalisation rules. Models:
LightGBM (MIT), MiniLM (Apache-2.0, 118M params). `requirements.txt` pinned.

## 5. Experiment log
| Date | Change | Val macro-F0.5 (all / US / IN) | Blocking recall | Notes |
|---|---|---|---|---|
| 09-25 | Sparse char-3gram TF-IDF top-k (sparse_dot_topn) | – | – | Too slow: >12 min for 150k queries on 8 cores. Dropped. |
| 09-25 | LSA (char 2–4 TF-IDF → SVD-256) + GPU exact top-k | – | US R@10: name .688 / addr .893 / full .936; union@10 .979 (27 cands/x). IN: .596/.781/.870; union@10 .947 | 150k matched X vs full S1 pool; GPU search 15 s |
| 09-25 | Rare-token index (df≤1000) | – | US R@10 .911, IN R@10 .890 | Complementary exact keys |
| 09-25 | Bi-encoder v1 with random in-batch negatives | – | – | Loss collapsed to 0.005 by step 400 (too easy). Replaced by hard-negative batches (same city/state or same first name word). |
| 09-25 | Bi-encoder enc_v1: hard-negative batches, 1.6M pairs, 1 epoch, bs 192, 43 min | – | **HOLDOUT** US R@1 .972 / R@10 .989; IN R@1 .966 / R@10 .986 | Large jump over LSA. Encode speed 6.5–9k texts/s. |
| 09-25 | Pruner v0 on enc_v1 sample | – | union pre-prune .9935; pruned k=20 .9899 at 2.9 pairs/x | enc finds 99.7% of *train* pairs vs ~99% of holdout, so it memorised. Downstream models must train on entities the encoder never saw, which led to the ENC/MATCH fold split. |
| 09-26 | **v1 full pipeline**: enc_v2 + lsa_full/lsa_name/keys → pruner k12 → m_v1 (2-stage LGBM, 2M TRAIN_X) → thr 0.8 | **0.9856** (US .9865 / IN .9844), micro P .996 R .965 | union .9945 → pruned .9892 (1.64/x train, 2.21/x test) | Expected-F decoder .9848 < thr 0.8 (probabilities not calibrated). Test France: 3.50 matches/S1 and 4.8% empty, vs US 3.38 / 5.7%, so France transfer looks healthy. **Submission v1.** |
| 09-26 | enc_v2: ENC folds only, 2 epochs, bs 256, frozen word embeddings + grad checkpointing | – | – | Running the LSA pass at the same time pushed training into Windows shared-GPU-memory fallback (4 GB spilled, 3× slower). Fix: memory-fraction cap, frozen 96M-param embedding matrix, checkpointing → 0.31 s/it at 2.3 GB. Never share the GPU between jobs. |
