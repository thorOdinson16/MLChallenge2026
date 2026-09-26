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
| 09-26 | Isotonic-calibrated p2 (global and per-country) fed into the expected-F decoder, tested on a leak-free half-split of holdout (see `ber/experiments/calibration.py`) | thr baseline 0.98578 vs expF baseline 0.98399; thr calibrated 0.98576 vs expF calibrated ~0.98490 | – | Calibration does **not** close the expected-F-vs-threshold gap noted above — expF stays ~0.001-0.002 behind the best threshold in every version (uncalibrated, global-cal, per-country-cal). Root cause is more likely the independence assumption in `decode.best_prefix` (Poisson-binomial over each candidate's p2) being violated by design: stage-2's own cluster-evidence features (`c_other_p`, `c_n_strong_s`, ...) explicitly encode correlation between a S1's candidates. Calibration only fixes marginal accuracy, not that correlation, so it can't fix this. A real fix would need a decode that models the joint/correlated structure, not a calibration step. |
| 09-26 | Finer threshold sweep for `decode_threshold`, 0.005 steps, full holdout (not split) | best 0.98569 @ thr=0.83 vs current THR=0.8 @ 0.98563 | – | Confirms the "0.80-0.85 plateau" note precisely: true optimum is 0.83, but the gain over 0.8 is +0.00006 — noise, not real. Not worth changing `pipeline.py`'s THR. |
| 09-26 | Two bugs found while reading the code closely (neither invalidates existing results, both worth knowing) | – | – | (1) `matcher.eval_s1()`'s filter `s1 * 40503 % 3 == 0` is a no-op since 40503 is divisible by 3 — it always evaluates true, so `eval_s1()` silently returns the *entire* holdout set rather than the ~1/3 subset the code implies. (2) `split.fold_expr()` uses an odd multiplier (2654435761) with `N_FOLDS=10` (even); since odd × anything preserves input parity through `% 2^32`, and `% 10` (even) preserves that parity too, `HOLDOUT` (fold 0, even) can only ever contain even-indexed S1 rows. Harmless in practice since row order is arbitrary, but real. |
| 09-26 | **Stage 3** (`ber/stage3.py`): LightGBM trained on out-of-sample holdout rows (v2 p1/p2 + raw v2 feats + X-competition features, grouped 5-fold CV) | holdout 0.98885 -> **0.98919** (thr 0.75; +0.00034) | – | 68% of gain is the gap between a pair's p2 and the strongest competing p2 of the same X (another collective-inference round on final probabilities). Pool-size features (namesake counts) dropped so it transfers to the smaller test S1 pool. Extended version (proper runner-up gap, competitor name/address flags) 0.98923: no further gain. Test-graph output reproduces exactly (`sub_stage3_all.tsv`). |
| 09-26 | LB probes on v2 (public LB, 6-decimal): v2_main **0.982614**; v2+fr_swap 0.982 (worse); v2 + 29k France pairs with p2 0.3-0.8 and strong address 0.98175 (**worse**, added pairs ~59% correct); v2 with France emptied 0.848 -> US/IN ~0.988, France ~0.953; half of France pairs kept 0.94459 | – | – | France's mid-band probabilities are calibrated (precision ~ mean p2), so no threshold below 0.8 helps. France's loss looks recall-like (half-probe shortfall ratio 0.82 vs 0.89 for US/India). Model self-estimate for France (~0.98) is far above the LB (~0.953), so ~29k errors are invisible to its own calibration. |
| 09-26 | Negative results, all measured on labeled holdout unless noted | each <= +0.0001 | – | Per-country / per-slice / finer thresholds; entity-level rescue and guard rules; v1+v2 logit blend (+0.00006); slot-count prior (would flip ~2% of lost-to-other pairs); within-cluster name-typo consistency (P 0.53 vs 0.43); X-name style classifier inside the uncertain band (AUC 0.524); twin-S1 (same name+house number) hypothesis for France (holdout precision 0.997-0.999 for twin S1, same as non-twins); France looser pruning (k=30, min_p 1e-5 adds 5.8M pairs, only 132 with enc score >= 0.9); retriever-overlap capture-recapture (encoder finds ~100% of every proxy set, uninformative). |
| 09-26 | Where US/India loss lives (v2 holdout, thr 0.8): recall 0.9717, precision 0.9975 | – | – | 70% of missed pairs are records with an empty address (33.6k true pairs, recall 0.55). 87% of those misses have a same-name S1 in the pool (recall 0.04-0.39 for k>=2): namesake ties are irreducible from the record's content (F0.5 break-even probability ~0.75). Unique-name empty-address records already reach 0.89. Pruner drops 3.3k of them (union 0.908 -> pruned 0.810); k=30, min_p 1e-5 recovers ~1.6k pairs for ~2M extra candidates (est. +0.0003). US/India ceiling ~0.989-0.990. |
| 09-26 | **Cross-encoder on the uncertain band** (`ber/crossenc.py`): multilingual MiniLM (Apache-2.0, 118M) fine-tuned on (X, S1) text pairs with p1 in [0.01, 0.99], top-4 per X; trained on ALL non-holdout folds (1.2M pairs, 76 min on the GPU) because it never sees encoder features; its logit feeds the stage-3 stacker | stage-3 holdout CV (5-fold grouped, out-of-fold): 0.98885 (v2) -> 0.98919 (stacker alone) -> 0.99054 (first CE, MATCH folds only, 319k pairs) -> **0.99096** (this CE; US 0.9906, India 0.9915). Both CEs together add nothing (0.99097) | – | Band AUC on holdout: p2 0.9586 -> p2+ce 0.9724 (first CE: 0.939 -> 0.955, India 0.949 -> 0.972). Complementary to the GBDT: alone it is weaker than v2 in the band, together they are much better; it reads cross-script pairs (Devanagari vs Latin) directly. France has 3.7x more uncertain pairs per S1 than the US (427k for 259k S1). Reproduce: `python -m ber.crossenc ce_v2 0.01 0.99 4 2 all`, then `python -m ber.stage3 all` (writes sub_stage3_*.tsv). |
| 09-26 | **Leaderboard: v2 + stage-3 + ce_v2 (all countries) = 0.984809** (v2_main 0.982614, +0.0022) | – | – | Holdout predicted about +0.0018 from the US/India part alone (85% of the score), so France also gained roughly +0.0004. Change vs v2_main: US +8.1k/-8.3k pairs, India +22.6k/-13.4k, France +27.9k/-13.9k. The US/India-only variant (France = v2_main) was built but not needed. |
