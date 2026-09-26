# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** [Date]

---

## 1. Executive Summary
We treat the task from the S2/S3 side. Every S2/S3 record belongs to at most one S1 entity, so each
record retrieves its most likely S1 entities from a fine-tuned multilingual bi-encoder and three
lexical retrievers. A learned stage-0 pruner cuts that union to about 3 candidates per record. A
two-stage LightGBM matcher scores the pairs; its second stage reads cluster context such as competing
candidates and the entity's other matches. Final matches come from a decoder that keeps one S1 per
record and then picks, per S1 entity, the set that maximises the expected macro F0.5. The empty set is
one of the options.

---

## 2. Methodology

### 2.1 Problem Analysis
- **Structure:** each S2/S3 record matches at most one S1 entity (0 exceptions among 7.6M training
  links). About 26% of S2/S3 records match nothing, and these are the main source of false merges.
  Matches never cross countries. 5.6% of S1 entities are singletons. The mean is 3.5 matches per
  entity (1.7 from S2, 1.8 from S3).
- **Name noise:** typos, character swaps, digit-for-letter substitutions (`Br0wn`, `5iddipet`),
  added accents, token reordering, legal-suffix changes (Pvt/Private, Ltd/Limited, `[LLC]`),
  honorific prefixes (Shri, Smt, M/s), DBA/`t/a` trade names, website-form names (`aslion.com` =
  initials of *Arredondo Strategic Lion*), and fully replaced names.
- **Scripts:** in India, 23% (S2) and 13% (S3) of names, and about 22% of state names, are written
  in Devanagari, Tamil, Telugu, Kannada, Bengali, Gujarati or Malayalam.
- **Address noise:** components reordered, abbreviations, 3–5% empty addresses, literal `null` and
  `N/A`, PO Box/PMB injection, and **house numbers altered** (6437→643, 175/J→275/J).
- **Ambiguity:** generic names are shared by up to 253 S1 entities, and one address can host up to 14
  S1 entities. Neither the name nor the address is sufficient on its own.
- **France** appears only in test. It follows the same generator style (R./Rue, All., St.-/Saint-,
  SARL/SAS/EURL, and region/department swaps).

### 2.2 Solution Strategy
**Approach Type:** retrieval with a fine-tuned bi-encoder and lexical retrievers, then a learned
pruner, then a two-stage GBDT matcher, then expected-F0.5 decoding.  
**Core Innovation:**
1. The S2/S3-side formulation, with uniqueness enforced as a hard constraint.
2. A contrastive bi-encoder trained with *hard-negative batching* (batches of businesses from the same
   city or with the same name) and a **fold split** that keeps the encoder's training entities away
   from the entities used to train the downstream models. The downstream models therefore see
   test-like encoder scores.
3. Per-entity **expected-F0.5-optimal decoding**, which is the exact optimum of the macro metric under
   the model's probabilities, instead of one global threshold.

---

## 3. Candidate Generation (Blocking)
- **Normalisation first.** An Indic→Latin token dictionary is *learned from the training pairs* by
  positional and co-occurrence alignment. Tokens seen only in test map to the nearest Latin word in
  the S1 vocabulary. Other steps: legal-form canonicalisation, DBA splitting, URL stems, leetspeak
  repair, abbreviation expansion per country, state/region extraction (French regions and departments
  map to one code), and number extraction.
- **Retrievers.** All run within the same country and are exact GPU top-k where dense.
  - Fine-tuned bi-encoder (multilingual MiniLM-L12, Apache-2.0, 256-dim projection), top-30.
  - LSA on character 2–4-grams (TF-IDF followed by 256-dim SVD, fitted without labels on each
    country's own data, so it adapts to France), over name+address and over name only.
  - An inverted index on rare tokens (name words, address words and numbers with S1 document
    frequency ≤ 1000, weighted by IDF).
- **Stage-0 pruner.** A LightGBM on retrieval signals (score, rank and gap to the best score per
  retriever, number of retrievers agreeing, number of candidates) keeps the top candidates per record.
- **Candidate pairs generated:** [fill]
- **How true matches were kept:** each retriever's contribution was measured on held-out
  entities. The union reaches 99.35% recall before pruning.

---

## 4. Matching Model
**Features used (about 75, all language-agnostic; `country` is never a feature):**
- **Name:** ratio, token-sort, token-set, partial and Jaro-Winkler scores on the canonical and core
  name; IDF-weighted token Jaccard and coverage; rarest shared token; DBA-alternative similarity;
  glued/initials/URL similarity; legal-form agreement; how often the name occurs among S1 records.
- **Address:** fuzzy scores; IDF-weighted word overlap; number-set overlap; first house number
  exact, prefix, suffix and Levenshtein distance; state agreement; PO box flag; address frequency;
  missing-address flags.
- **Retrieval:** each retriever's score and rank, and the pruner probability.
- **Context (stage 2):** rank and gap of the pair among the record's candidates and among the entity's
  candidates; the entity's count of strong matches, overall and per source; similarity of the record to
  the entity's strongest *other* match (cluster evidence, which rescues records with garbled names).

**Model type:** LightGBM, two stages. Stage 1 runs out-of-fold; stage 2 adds context features built
from the stage-1 probabilities.  
**Threshold selection method:** expected-F0.5-optimal subset per S1 entity, compared on the holdout
against global thresholds.

---

## 5. Results & Error Analysis
- **F_0.5 Score (macro, holdout entities never used in training):** [fill]
- **Common false positives (wrong merges):** [fill]
- **Common false negatives (missed matches):** [fill]

---

## 6. Conclusion
[fill]

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/` contains `src/ber/` (package), `README.md` (exact commands),
`requirements.txt` (pinned) and `ARCHITECTURE.md` (design and experiment log). The entry point is
`python -m ber.pipeline all`. Its steps are prepare, encoder, retrieve, prune, cands, feats, train
and submit, and each caches its result under `work/`.

### B. Additional Results
See the experiment log in `ARCHITECTURE.md`.
