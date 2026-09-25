# Business Entity Resolution — Pipeline

Matches Source-2 / Source-3 business records to Source-1 (reference) entities across
US, India, and (test-only) France. Tabular ML approach: normalize -> block -> engineer
pairwise features -> LightGBM classify -> decode to one-to-one matches.

## Layout

```
src/
  normalize.py          text normalization (accents, abbreviations, legal suffixes, house numbers)
  blocking.py            candidate generation (4 channels: comb TF-IDF, addr TF-IDF, char-4gram
                          "glue" TF-IDF, exact house-number+locality) + recall diagnostics
  block_diag.py           standalone per-channel recall/miss diagnostics
  features.py             pairwise feature engineering (~60 features) + rank/gap context features
  build_train_pairs.py    builds a labeled pair table from a train S1 subsample -> parquet
  train_model.py          trains LightGBM, tunes the decode threshold on macro F0.5, error analysis
  predict.py              full test-set inference: blocking -> features -> model -> decode -> TSVs
  eval.py                 macro F0.5 scoring + blocking recall utilities
  eda.py, eda2.py         exploratory data analysis scripts (not part of the run pipeline)
```

## Setup

```bash
pip install -r requirements.txt
```

Place the challenge data under `dataset/train/` and `dataset/test/` (see repo root
`docs/README.md` for the expected file layout) before running anything below.

## Reproducing end-to-end

All commands are run from the repository root.

**1. Blocking recall gate** (sanity-checks candidate generation on a 5k-S1 sample before
spending time on the full pipeline; fails fast if recall drops below the given bar):

```bash
python code/business_entity_resolution/src/blocking.py --n 5000 --min-recall 0.97
```

**2. Build the labeled training pair table** (blocking + features on a 100k-S1 train
subsample, cached to `logs/pairs_train.parquet`):

```bash
python code/business_entity_resolution/src/build_train_pairs.py --n 100000
```

**3. Train the classifier** (group-split by S1 stratified by country, LightGBM with
early stopping, threshold sweep on macro F0.5, per-country breakdown, feature
importance, and false-positive/negative error samples printed to stdout):

```bash
python code/business_entity_resolution/src/train_model.py
```

This saves the model to `logs/lgbm.txt`. Steps 1-3 are also wired together in
`run_all.bat` (Windows).

**4. Run full test-set inference and produce the submission files:**

```bash
python code/business_entity_resolution/src/predict.py --thr 0.68
```

This is resumable: each (source, country) unit is checkpointed under
`logs/test_scored/`, and completed units are skipped on rerun. Writes
`output/matching_results.tsv` and `output/candidate_pairs.tsv`, then runs
`utils/validate_submission.py` automatically.

To re-decode at a different threshold without rerunning inference:

```bash
python code/business_entity_resolution/src/predict.py --decode-only --thr 0.65
```

## Method summary

- **Normalization**: lowercase, strip accents, expand abbreviations (Rd -> Road, Pvt ->
  Private), separate legal suffixes from the core business name, extract house numbers —
  country-agnostic (no hard-coded `{US, India}` branching, per the problem statement's
  requirement to treat `country` as an open set).
- **Blocking**: per (source, country) unit, retrieve ~65 candidates per S1 record via the
  union of 4 channels — combined name+address word TF-IDF, address-only word TF-IDF,
  char-4-gram "glue" TF-IDF on the compacted name (catches typos/concatenation), and an
  exact house-number + locality-token index. Validated at ~0.98 pair recall.
- **Features**: ~60 pairwise features per (S1, candidate) pair — fuzzy string similarity
  (token set/sort ratio, Jaro-Winkler, Levenshtein), token/house-number Jaccard overlap,
  legal-suffix agreement, per-channel blocking ranks/scores, plus rank/gap "context"
  features (best rank among candidates for this S1, gap to the runner-up, reverse rank
  of this S1 among candidates competing for the same target record).
- **Classifier**: LightGBM binary classifier (gradient-boosted trees, MIT-licensed,
  well under the 8B-parameter cap) scores each pair's match probability.
- **Decode**: sort all scored pairs by probability, enforce one-to-one by keeping each
  S2/S3 record for only its single best-scoring S1 candidate, then threshold (default
  0.68, tuned on held-out validation macro F0.5).

See `docs/Documentation_template.md` (repo root) for the full methodology write-up,
including per-country validation results and known limitations (India's cross-script
name noise lowers both blocking recall and classifier confidence relative to US/France).
