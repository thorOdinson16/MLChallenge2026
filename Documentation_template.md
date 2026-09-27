# ML Challenge 2026: Business Entity Resolution — AnchorMatch

**Team Name:** Oura Oura  
**Team Members:** Abhijna D S, Abhijnan B C, A K Pranav, Abyud Somashekara Shetty  
**Submission Date:** 2026-09-27

---

## 1. Executive Summary
We treat the task from the S2/S3 side. Every S2/S3 record belongs to at most one S1 entity, so each record retrieves
its most likely S1 entities from a fine-tuned multilingual bi-encoder and three lexical retrievers. A learned stage-0
pruner cuts that union to 2.2 candidates per record. A two-stage LightGBM matcher scores every pair over the complete
candidate graph (its second stage reads cluster context: competing candidates and the entity's other matches).
Cross-encoders then re-read the pairs the matcher is unsure about, and a stage-3 stacker trained on out-of-sample
holdout rows combines everything. France has no labels, so the cross-encoders are adapted to it with confident
pseudo-labels from the test inputs themselves. Final matches come from a decoder that keeps one S1 per record,
respects per-source caps, and picks per S1 the set with the highest expected macro F0.5 (the empty set included).
France is decoded with a "trust" blend that leans on the adapted cross-encoders more than on the US/India-trained
matcher, followed by four label-free rules that were each screened on the labelled holdout.
Holdout macro-F0.5 (US/India) 0.9914; public leaderboard **0.98832** (v1 0.975 -> v2 0.98261 -> 0.98832).

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
**Approach Type:** retrieval (fine-tuned bi-encoder + lexical retrievers) -> learned pruner -> two-stage GBDT matcher
over the full candidate graph -> cross-encoders on the uncertain band -> stage-3 stacker -> expected-F0.5 decoding,
with unsupervised domain adaptation to France.  
**Core Innovation:**
1. The S2/S3-side formulation, with uniqueness enforced as a hard constraint and collective inference: every stage
   after the first reads the competition between a record's candidates and the entity's other matches.
2. A **fold contract** that keeps every model's training entities away from the entities its outputs are evaluated or
   stacked on: bi-encoder on ENC folds, matcher on MATCH folds, holdout fold 0 never trained on until stage 3, which
   uses it through grouped cross-validation. Each downstream model therefore sees test-like inputs.
3. **Label-free adaptation to an unseen country.** France appears only in test. Confident stage-3 decisions on France
   test pairs become pseudo-labels for the cross-encoders (several rounds). Rules for France are accepted only when a
   label-free diagnostic (the same address/name-edit cell is highly precise in US/India) and a France-only A/B agree.
4. A **thinned-pool** simulation: the test S1 pool is sparser than train, so stage 3 is also trained on holdout graphs
   with 19% of S1 removed and the matcher context recomputed.
5. Per-entity **expected-F0.5-optimal decoding**, the exact optimum of the macro metric under calibrated probabilities.

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
- **Candidate pairs generated (test):** before pruning, the lexical pass (LSA full + LSA name +
  keys) produced 366.6M pairs (~37 per record) and the encoder pass 299.1M (30 per record). After the
  merge and stage-0 pruning, the final set is **21.85M pairs for 9.97M S2/S3 records (2.2 per record)**,
  a reduction ratio above 99.999% of the within-country cross product.
- **How true matches were kept:** each retriever's contribution was measured on held-out entities.
  Encoder alone, holdout R@10: US 98.9%, India 98.6% (LSA name+address: 93.6% / 87.0%).
  Recall of the retriever union: **99.45%**. Recall after pruning (12 kept per record plus a probability
  floor): **98.92%** on holdout entities.

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

**v2 additions:** edit fingerprints of the name change (inserted/deleted/swapped words, whether the inserted word is a
real dictionary word, a country word, a legal form, or a noise suffix; house-number shifts) and context computed over the
complete candidate graph of each split rather than a sample.

**Model types:**
- **Stage 1 / stage 2:** LightGBM. Stage 1 runs out-of-fold; stage 2 adds context features built from stage-1
  probabilities.
- **Cross-encoders:** (X text, S1 text) read jointly, raw `name ; address`, sequence-classification head, BCE, fp16.
  Bases: multilingual MiniLM-L12, XLM-R base, XLM-R large, multilingual E5 base, DistilBERT multilingual (licences and
  sizes in section 7). Trained on non-holdout folds of the uncertain band (stage-1 p in [0.01, 0.99], top-4 per record)
  plus France pseudo-labelled rows. One full-coverage variant also re-reads confident pairs. Different bases and seeds
  are used on purpose: their errors are less correlated, and stage 3 combines them.
- **France pseudo-label rounds.** Round r trains cross-encoders on France test pairs labelled by round r-1's final
  decisions (p >= 0.9 -> 1, p <= 0.05 -> 0, ~580k rows). Pairs covered by an LB-verified France rule get the rule's
  label, which removes the confirmation bias of self-training. Five rounds were run; the gain per round shrinks
  (round 4 -> 5 labels are almost identical).
- **Stage 3:** LightGBM trained on the holdout rows, where the matcher and cross-encoder outputs are out-of-sample
  (grouped 5-fold). Inputs: stage-1/2 probabilities, the gap to the strongest competing pair of the same record, the
  cross-encoder logits and raw pair features. Pool-size features are excluded so it transfers to the smaller test pool.

**Threshold selection method:** chosen on the labeled holdout, averaged over the full and the thinned pool:
- decoder options: global thresholds 0.7–0.85 or expected-F0.5 per S1 (chosen);
- source caps: at most 5 S2 and 6 S3 matches per S1;
- country-word decoy rule: drop a pair whose X name inserts the country word;
- exact-duplicate consistency: records with identical normalised name and address share one S1 (99.7% in train).

**Per-country final.** Countries never interact in the metric, so each is decoded separately:
- **US/India:** stage 3 (expected-F0.5 decoder, duplicate consistency). A new cross-encoder enters only if stage 3 is
  at least as good on both the full and the thinned holdout and strictly better on one.
- **France trust rule** (`experiments/fr_trust.py`). The matcher never saw France, so its probability features are the
  least transferable. France pairs are scored by a logit blend (w = 0.5) of stage 3 with all features and a stage 3
  without the matcher-probability features, averaged over 3 seeds, and accepted at p >= 0.85. Validated on an
  unseen-country proxy: matcher and stage 3 trained on US only and evaluated on India, where India F0.5 goes from
  0.9600 to 0.9630. The grid over w and threshold was flat around this point.

**France post-rules** (label-free, then verified by France-only leaderboard A/Bs). They apply in this order: noise words,
acronyms, tie trim.
- French noise-suffix words (*fils*, *service*, *associés*) at the same address and house number are matches:
  accepted, +0.00054 LB.
- Category-word swaps and dropped words look like the same edit but turned out to be sibling-business decoys:
  rejected, −0.00088 LB.
- **Acronym rescue** (`ber/acronym.py`). France has 6× more acronym-named records than the US (23,976 vs 3,912) but
  matched only 72% of them (US 93%, India 97%). The France final is a blend of text cross-encoders, which cannot see
  that "RC" = "Rassemblement Club"; retrieval also misses many true S1 because an acronym has no name similarity. Two
  rules, each with a strict address check (same house number or none; every street/city word fuzzy-matches) and unique
  per record:
  - (A) an existing candidate whose initials equal the acronym;
  - (B) a new blocking key (initials, house number) over the whole S1 pool.

  Labelled US/India holdout precision 13/13 and 41/41 (broad cell 2,759/2,759); 4,737 France pairs.
- **Namesake-tie trim** (`ber/tie_trim.py`, France only). An accepted pair is dropped when its record's top-probability
  share p1/(p1+p2) is below 0.6: an empty-address record with two same-name S1 in different cities is a coin flip.
  US/India holdout precision in that bucket is 0.67, below the F0.5 keep threshold of ~0.75.
- **Near-duplicate twins** (`ber/twin.py`, US/India only). An unassigned record with the same normalised address and a
  near-identical name as an accepted record inherits its S1. Stage-3 holdout: +0.00002 to +0.00004 on both halves,
  full and thinned pools.
  France is excluded because there the near-identical names are word-change siblings.

**Rejected by the labelled holdout** (never shipped), which is how every rule was screened:
- a co-training veto/add by the full-coverage cross-encoder: −0.0002 on the real stage 3;
- "S1 name + one business word" insertions (Groupe/Holding/Participations/International…, 81k France records):
  decoys, US/India precision 0.0–2.5%, with an upward house-number shift;
- address-only matching of invented one-word names: 0.58;
- no-house-number whole-pool rescue: 0.10;
- accent-dropped glued names: 0.66 overall;
- relaxing the country-word veto: all 6,177 candidates are "<name> France" decoys;
- lower France thresholds (clear-winner adds at p >= 0.3; an entity-aware variant using a 0.5 bar for empty S1 and
  0.75 otherwise): −0.003 holdout for the first; the second is +0.0004 on holdout but −0.0005 to −0.0012 on the
  India unseen-country proxy, where mid-range probabilities are over-confident. Not shipped;
- zero-shot LLM judge (Qwen3-4B-Instruct) on grey-zone pairs: labelled-gate AUC 0.53 / 0.51, i.e. no signal. Not used;
- France exact-duplicate propagation: only 64 extra pairs, because identical records already get identical scores;
- "under-filled" S1 entities (fewer matches than the generator's rate): borderline pairs are 40–72% precise, below the
  ~0.75 break-even.

---

## 5. Results & Error Analysis
- **F_0.5 Score (macro, holdout entities never used in training, US + India):** **0.9914** (out-of-fold stage 3;
  thinned pool 0.9910). **Public leaderboard: 0.98832.**

| Version | Holdout macro-F0.5 | Public LB |
|---|---|---|
| v1: retrieval + pruner + 2-stage LightGBM, threshold 0.8 | 0.9856 | 0.975 |
| v2: full-graph context, edit fingerprints, source caps, country-word rule | 0.9889 | 0.98261 |
| v2 + cross-encoder + stage 3 | 0.9910 | 0.98481 |
| + France pseudo-label cross-encoders, thinned-pool stage 3, expected-F0.5 decoder, duplicate consistency | 0.9912 | 0.98643 |
| + French noise-suffix rule (France rows only) | 0.9912 | 0.98697 |
| + full-coverage cross-encoder for US/India; new France cross-encoder blend; full noise-word cell | 0.9913 | 0.98770 |
| + corrected-label full-coverage France cross-encoder (ce_full2), round-4 France labels | — | 0.98781 |
| + round-4 XLM-R base and DistilBERT France CEs, acronym rescue, tie trim (France); twins (US/India) | 0.9914 | **0.98832** |
| + round-5 CEs (MiniLM seed 7, DistilBERT seed 2, multilingual E5) | 0.9914 | 0.98829 |
| **Final submission = the 0.98832 file** (round 5 added no measurable gain) | 0.9914 | **0.98832** |

Rejected submissions, kept for the record: v2 + France swap rule 0.982; v2 + mid-band France adds 0.98175;
category-swap + dropped-word France rescues 0.98609; aggressive France recall adds (+3.1k pairs, entity-aware
thresholds 0.4 / 0.7) 0.98819. The last one confirmed the India-proxy finding that mid-range France probabilities are
over-confident.

- **Leaderboard decomposition.** Countries never interact in the metric, so France-only A/Bs isolate it:
  LB = 0.85·F(US/India) + 0.15·F(France). v2 gave about 0.988 on US/India and 0.953 on France. France is the main gap
  and the target of the adaptation work.
- **France (no labels):** a US/India-trained cross-encoder is close to useless on an unseen country (US-to-India proxy:
  band AUC 0.69). Pseudo-labels recover about 97% of the gain of true labels in that proxy (AUC 0.93 vs 0.94).
  On France the model's self-estimated F0.5 (~0.98) was far above the LB (~0.95). Some France errors are confident,
  so no probability threshold alone can fix them; this motivated the full-coverage cross-encoder and the label-free
  France rules.
- **Common false positives (wrong merges):** 73% involve an S2/S3 record that belongs to *no* S1 entity.
  Typical cases are decoys with an identical or near-identical name and an empty address, or an altered
  house number (6622 vs 662). The rest are replaced or garbled names at an address shared with the S1.
- **Common false negatives (missed matches):** 31% are lost in blocking or pruning. The remainder are
  records with an empty address where only the name is available (the model is conservative because the
  decoys look the same), fully replaced names (`Ve0vantage` for *Massie Family Practice LLC*), and
  names cut down to a generic stem (`Indian Private Limited Center`).

---

## 6. Conclusion
Recasting the problem from the S2/S3 side and exploiting that each record belongs to at most one entity turned entity
resolution into ranking plus collective inference. A fine-tuned bi-encoder solves retrieval well (98.9% R@10). A cheap
learned pruner keeps the candidate set tiny (2.2 per record). A GBDT over the full candidate graph, cross-encoders on
the uncertain band and a stacker trained on out-of-sample holdout rows reach 0.9912 macro-F0.5 on entities never seen
in training.

Lessons:
- Keep a strict fold contract between every pair of stacked models.
- Treat the unseen country as a domain-adaptation problem: confident pseudo-labels from the test inputs, plus rules
  verified label-free and then by isolated France-only A/Bs.
- Distrust a model's self-estimate on a new domain: France's confident errors were invisible to calibration.

Remaining headroom:
- Namesake ties with empty addresses in US/India (irreducible from the record content).
- French sibling-business decoys, which share the address and differ by one category word.

---

## 7. Models, Compliance and Compute
All models are MIT or Apache-2.0 and at most 560M parameters (limit: 8B). No external data or lookup services were
used. Every model is trained only on the provided training data plus pseudo-labels derived from the test inputs.

| Model (Hugging Face checkpoint) | Licence | Parameters | Role |
|---|---|---|---|
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | Apache-2.0 | 118M | bi-encoder retrieval; cross-encoders |
| FacebookAI/xlm-roberta-base | MIT | 278M | cross-encoders |
| FacebookAI/xlm-roberta-large | MIT | 560M | cross-encoders |
| intfloat/multilingual-e5-base | MIT | 278M | cross-encoders |
| distilbert/distilbert-base-multilingual-cased | Apache-2.0 | 135M | cross-encoders |
| LightGBM | MIT | — | pruner, matcher, stage 3 |

Qwen3-4B-Instruct (Apache-2.0) was evaluated as a zero-shot judge and **not** used in any submission.
Compute: one laptop GPU (RTX, 8 GB) plus free Kaggle T4 sessions for the XLM-R large and E5 cross-encoders.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/` contains `src/ber/` (package), `README.md` (exact commands),
`requirements.txt` (pinned) and `ARCHITECTURE.md` (design and experiment log). The entry point is
`python -m ber.pipeline` (prepare, encoder, retrieve, prune, cands), then `ber.v2`, `ber.crossenc` /
`ber.cefull`, `ber.stage3` and `experiments/overnight.py`; every step caches its result under `work/`. The final
submission is assembled by `scripts/wire_final.sh <name> <cross-encoder tags>`, which runs the US/India gate and the
France trust rule, splices the countries and applies the noise-word rescue. `scripts/post_final.sh <name>` then
applies the acronym rescue, tie trim and twins, extends `candidate_pairs.tsv` with the acronym rule-B pairs, and runs
the official validator with the candidate file. The exact command sequence is in `README.md`.

### B. Additional Results
See the experiment log in `ARCHITECTURE.md`.
