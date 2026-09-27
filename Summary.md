# AnchorMatch: Business Entity Resolution (Team Oura Oura)
**Amazon ML Challenge 2026.** Public leaderboard **0.98832** (macro F0.5). Holdout macro F0.5 (US/India, entities never trained on) **0.9914**.

## 1. Approach
The task is solved from the S2/S3 side. Every S2/S3 record belongs to at most one S1 entity, so each record ranks its
possible S1 entities, and uniqueness is enforced as a hard constraint. The pipeline:

1. **Normalisation.** An Indic-to-Latin token dictionary learned from the training pairs, legal-form
   canonicalisation, abbreviation expansion, URL and leetspeak repair, and number extraction.
2. **Candidate generation.** Same-country retrieval from a fine-tuned multilingual bi-encoder (MiniLM-L12, top-30),
   character n-gram LSA (name+address and name only) and a rare-token inverted index. A LightGBM stage-0 pruner cuts
   the union to **2.2 candidates per record** (21.85M test pairs, >99.999% reduction) at **98.9% holdout recall**.
3. **Matcher.** A two-stage LightGBM over the complete candidate graph with ~75 language-agnostic features: name
   and address fuzzy scores, IDF overlaps, house-number agreement, edit fingerprints of the name change and retrieval
   signals. Stage 2 adds cluster context, i.e. the competing candidates and the entity's other matches.
4. **Cross-encoders** re-read the uncertain band (matcher p in [0.01, 0.99], top-4 per record) as raw
   `name ; address` pairs.
5. **Stage-3 stacker.** LightGBM trained only on the out-of-sample holdout fold (grouped 5-fold CV), including
   "thinned" holdout graphs with 19% of S1 removed, which mimic the sparser test pool.
6. **Decoding.** Per-S1 expected-F0.5-optimal subset selection, source caps (5 S2 / 6 S3), a country-word decoy rule,
   and exact-duplicate consistency.

## 2. The unseen country (France)
France appears only in test and was the main gap (LB decomposition: US/India ~0.990, France ~0.95 at v2).
- **Pseudo-label domain adaptation.** Confident France decisions (p >= 0.9 / <= 0.05, ~580k pairs) retrain the
  cross-encoders. Five rounds were run. On a US-to-India proxy, pseudo-labels recover 97% of the gain of true labels
  (AUC 0.69 -> 0.93).
- **Trust rule.** France is decoded from a blend of stage 3 with and without the matcher-probability features, since
  those features transfer worst to an unseen country (proxy 0.9600 -> 0.9630).
- **Label-free France rules**, each screened on the labelled holdout:
  - French noise-suffix words (*fils / service / associés*): +0.0005 LB;
  - an acronym rescue ("RC" = "Rassemblement Club"; initials plus a strict address check): 54/54 correct on holdout;
  - a namesake-tie trim.

## 3. Models (all MIT / Apache-2.0, at most 560M parameters)
| Model | Licence | Params |
|---|---|---|
| paraphrase-multilingual-MiniLM-L12-v2 | Apache-2.0 | 118M |
| xlm-roberta-base / xlm-roberta-large | MIT | 278M / 560M |
| multilingual-e5-base | MIT | 278M |
| distilbert-base-multilingual-cased | Apache-2.0 | 135M |
| LightGBM | MIT | — |

No external data or lookup services were used. Compute: one 8 GB laptop GPU and free Kaggle T4 sessions.

## 4. Experiments and results
| Version | Holdout F0.5 | Public LB |
|---|---|---|
| v1: retrieval + pruner + 2-stage LightGBM | 0.9856 | 0.975 |
| v2: full-graph context, edit fingerprints, caps, decoy rule | 0.9889 | 0.98261 |
| + cross-encoder + stage-3 stacker | 0.9910 | 0.98481 |
| + France pseudo-label CEs, thinned-pool stage 3, expF decoder | 0.9912 | 0.98643 |
| + noise-word rule, trust rule, XLM-R large, E5 | 0.9913 | 0.98770 |
| **+ round-4 CEs, acronym rescue, tie trim, twins (final)** | **0.9914** | **0.98832** |

**Rejected on evidence**, each measured on the labelled holdout, the India proxy or a France-only LB A/B:
- category-swap and dropped-word France rescues: sibling decoys, −0.0009 LB;
- lower France thresholds: over-confident on an unseen country, −0.00013 LB;
- a zero-shot LLM judge (Qwen3-4B): AUC 0.53;
- co-training;
- no-house-number and address-only rescues: 10–58% precise.

## 5. Conclusions
- Formulating from the S2/S3 side with hard uniqueness turns entity resolution into ranking plus collective inference.
- A strict fold contract between stacked models is what makes the stage-3 gain transfer (holdout vs LB consistent).
- An unseen country is a domain-adaptation problem: pseudo-labels and features that transfer help, while the model's
  own confidence on the new domain cannot be trusted. Every France change was validated on a proxy unseen country.
- **Remaining error:** about 80% of the missed pairs are empty-address records that share their name with planted decoys,
  which is irreducible from record content. The next step would be a larger fine-tuned pair model (≤ 8B LLM) with
  namesake hard negatives.
