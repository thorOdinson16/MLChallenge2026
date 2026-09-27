# AnchorMatch — Business Entity Resolution pipeline

**Team Oura Oura** — Abhijna D S, Abhijnan B C, A K Pranav, Abyud Somashekara Shetty

Design, EDA findings and the experiment log live in [ARCHITECTURE.md](ARCHITECTURE.md).

## Results
| version | what changed | holdout macro-F0.5 | public leaderboard |
|---|---|---|---|
| v1 | bi-encoder + lexical retrieval, pruner, 2-stage LightGBM, threshold 0.8 | 0.9856 | 0.975 |
| v2 | full-graph context, edit-fingerprint features, source caps, deterministic decode | 0.9889 | 0.982614 |
| v2 + cross-encoder + stage 3 | cross-encoder on the uncertain band, stacker trained on out-of-sample holdout rows | 0.9910 (5-fold out-of-fold) | 0.984809 |
| **+ France-adapted cross-encoders** (`sub_K`) | cross-encoders fine-tuned with France pseudo-labels (2 rounds, MiniLM + XLM-R base), stage 3 trained on full + thinned pools, expected-F0.5 decoder, exact-duplicate consistency | **0.99120** (thinned pool 0.99083) | **0.986429** |

Holdout is fold 0 of the train S1 entities (US and India only; France has no labels). Leaderboard numbers come from
the portal.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.10
pip install -r requirements.txt                         # install the CUDA torch build first (see file)
```
Hardware used: 16 GB RAM, one 6-8 GB NVIDIA GPU, about 60 GB free disk (caches under `work/` are large). Run GPU
steps one at a time; never share the GPU between jobs. Everything streams in chunks, so larger machines only run faster.

## Data locations
By default the code expects the challenge folder layout:
```
<project>/student_resource/dataset/{train,test}/...   input TSVs
<project>/work/                                       caches (created)
<project>/output/                                     matching_results.tsv, candidate_pairs.tsv
```
To use other locations, set `BER_DATA`, `BER_WORK` or `BER_OUT`.

## Run end to end
All commands run from `src/`.

**1. Retrieval and candidate set (shared by every version)**
```bash
python -m ber.pipeline prepare   # TSV -> parquet, learned transliteration dict, normalisation
python -m ber.pipeline encoder   # fine-tune multilingual MiniLM bi-encoder on ENC folds (about 45 min on the GPU)
python -m ber.pipeline retrieve  # un-pruned retriever passes (LSA, keys, encoder) for train + test (about 4 h)
python -m ber.pipeline prune     # candidate sample -> stage-0 pruner
python -m ber.pipeline cands     # merge + prune -> final candidate set (candidate_pairs.tsv)
```
(`python -m ber.pipeline feats | train | submit` runs the v1 matcher; v2 replaces it.)

**2. v2 matcher** (features and context over the complete candidate graph of each split)
```bash
python -m ber.v2 feats           # pair features for every candidate pair, train and test
python -m ber.v2 train           # stage-1 (out-of-fold) + stage-2 LightGBM -> preds/train_m_v2, models/m_v2_s{1,2}
python -m ber.v2 eval            # decoder selection on the holdout
python -m ber.v2 test            # -> preds/test_m_v2.parquet  (x, s1, src, p1, p2)
```

**3. Cross-encoder** (GPU, about 75 min; reads `preds/{train,test}_m_v2.parquet`)
```bash
python -m ber.crossenc ce_v2 0.01 0.99 4 2 all
#                      tag   lo   hi  k epochs train-folds
```
Writes `models/ce_v2.pt` and `preds/ce_v2_{holdout,test}.parquet` (x, s1, ce).

**4. Stage 3 and the final files**
```bash
python -m ber.stage3 all         # = train (about 3 min) + test (about 10 min) + final
# or step by step:  python -m ber.stage3 train | test | final
```
`final` writes `output/matching_results.tsv` and `output/candidate_pairs.tsv` and runs the official validator.
Reproducibility: retraining the cross-encoder on a GPU (and refitting the stage-3 model) is seeded but not guaranteed to be
bit-identical. Given the shipped `models/m_v2_s3ce2.txt`, `models/ce_v2.pt` (or its scores `preds/ce_v2_*.parquet`) and
`preds/test_m_v2.parquet`, `python -m ber.stage3 final` regenerates the submitted `matching_results.tsv` byte for byte
(checked: identical checksum to the file that scored 0.984809).

**5. France adaptation (current best, `sub_K`)**
```bash
bash ../run_france_adapt.sh      # from src/: thinned-pool simulation, 2 pseudo-label rounds, stage 3 -> output/sub_K.tsv
```
About 7 h on one 8 GB GPU (the cross-encoder runs dominate). Every stage-3 choice (training pools, decoder,
duplicate propagation) is made on the labeled holdout and logged to `output/overnight_report.md`.

`run.sh` is a thin wrapper that sets the environment variables used on the author's machine.

**6. Late additions (27 Sep): extra France cross-encoders and post-steps**
```bash
# ce_full2: full-coverage MiniLM CE with corrected France labels (noise-word pairs = match), France-only test scoring
BER_CE_CORRECTED=1 python -m ber.cefull train ce_full2 && BER_CE_CORRECTED=1 python -m ber.cefull score ce_full2
# ce_r4f: round-4 CE, DistilBERT-multilingual (Apache-2.0) on round-4 France pseudo-labels + train band, band pair set
BER_CE_R4=fr_pseudo4.parquet BER_CE_BASE=distilbert/distilbert-base-multilingual-cased BER_CE_LR=5e-5 \
  BER_CE_BUDGET_MIN=55 BER_CE_SCORE_SET=band BER_CE_SEED=1 python -m ber.cefull train ce_r4f && \
  BER_CE_R4=fr_pseudo4.parquet BER_CE_BASE=distilbert/distilbert-base-multilingual-cased BER_CE_SCORE_SET=band \
  python -m ber.cefull score ce_r4f
# post-steps on the final matching_results.tsv (each keeps rows of other countries byte-identical, then validates)
python -m ber.cn_rescue <in.tsv> <out.tsv> cn_rescue_pairs_full.parquet   # France noise-suffix rescue (adds)
python -m ber.twin <in.tsv> <out.tsv> <test_s3_X.parquet> p3 US,India     # near-duplicate X at one address (US/IN)
python -m ber.tie_trim pairs <france_probs.parquet> pD 0.85 0.6          # France namesake-tie coin flips ...
python -m ber.tie_trim apply <in.tsv> <out.tsv>                           # ... removed
```
Optional zero-shot LLM judge (Qwen/Qwen3-4B-Instruct-2507, Apache-2.0, 4.0B; no fine-tuning) on France grey-zone pairs:
```bash
python -m ber.experiments.llm_kit                        # work/llm_kit/{gate_holdout,france_slices}.parquet
python ber/experiments/llm_judge.py <pairs.parquet> <out> [--shard K --nshards N]   # GPU; P("Yes") per pair
python -m ber.experiments.llm_eval gate <gate_scores.parquet>                       # labelled holdout gate
python -m ber.experiments.llm_eval france <france_scores.parquet> <add_min> <drop_max>
python -m ber.cn_rescue <in.tsv> <out.tsv> fr_llm_add_pairs.parquet && python -m ber.tie_trim apply <out.tsv> <out2.tsv> fr_llm_drop_pairs.parquet
```
Validate the final file with the candidate set too:
`python ../../../student_resource/utils/validate_submission.py --matching <final.tsv> --candidate <candidate_pairs.tsv> --test-dir <test dir>`.

## What each new stage does
- **Cross-encoder** (`crossenc.py`): reads an (X record, S1 record) text pair jointly and outputs a logit, only for pairs the
  matcher is unsure about (stage-1 probability in [0.01, 0.99], top-4 per X). It is weaker than the LightGBM matcher
  alone but complementary to it, and it reads cross-script pairs (Devanagari vs Latin) directly. Because it never sees
  encoder features it can train on every non-holdout fold, unlike the matcher.
- **Stage 3** (`stage3.py`): v2's probabilities are out-of-sample on the holdout, so those labeled rows can train one more
  small LightGBM. Its main inputs are the gap between a pair's final probability and the strongest competing pair for
  the same X, plus the cross-encoder logit. Features tied to the size of the S1 pool are excluded so it transfers to
  the smaller test pool. The final decode is unique assignment per record, threshold 0.75, source caps (at most 5 S2 and
  6 S3 matches per S1) and the country-word decoy rule.

- **France adaptation** (`experiments/fr_pseudo.py`, `crossenc.py` with `BER_CE_EXTRA`): France never appears in
  the training labels, and a cross-encoder that has never seen a country is close to useless there (US-to-India proxy:
  band AUC 0.69). Confident stage-3 decisions on France test pairs (p3 >= 0.9 match, <= 0.05 non-match; the
  ambiguous middle is dropped) are added as extra training rows. In the proxy these pseudo-labels recovered about 97%
  of the gain of true labels (AUC 0.69 -> 0.93, true labels 0.94). Two rounds: labels from the ce_v2 stage 3, then from
  the adapted stage 3. Uses only the provided unlabeled test inputs, no external data.
- **Thinned-pool stage 3** (`experiments/thinned_sim.py`, `experiments/overnight.py`): the test S1 pool is thinner
  than train, so stage 3 is also trained on holdout graphs with 19% of S1 removed (v2 context recomputed). Decoding
  uses the expected-F0.5 decoder on the calibrated stage-3 probabilities, then gives every exact-duplicate X record
  (same normalised name + address; 99.7% share one S1 in train) its group's S1.

## Code map (`src/ber/`)
| module | role |
|---|---|
| `io.py` | TSV → parquet with integer ids; ground-truth pairs |
| `translit.py` | Indic→Latin token dictionary learned from train pairs, plus a vocabulary-resolved fallback |
| `lexicon.py`, `normalize.py` | name/address canonicalisation (legal forms, DBA, URL names, abbreviations, states, numbers) |
| `blocking/dense.py` | LSA char-n-gram embeddings and exact GPU top-k |
| `blocking/keys.py` | rare-token inverted index |
| `blocking/encoder.py` | contrastive bi-encoder (hard-negative batches) |
| `candidates.py`, `prune.py` | retriever union and stage-0 learned pruning = final candidate set |
| `features.py` | language-agnostic pair features, including v2 edit-fingerprint features |
| `matcher.py` | v1 stage-1 LightGBM (OOF) and stage-2 context/cluster LightGBM |
| `matcher2.py`, `v2.py` | v2 matcher (full-graph context) and its driver |
| `crossenc.py` | cross-encoder on the uncertain band |
| `stage3.py` | stacker trained on out-of-sample holdout rows; writes the final files |
| `decode.py` | unique assignment, thresholds, source caps, expected-F0.5 decoder |
| `evaluate.py` | exact macro-F0.5 metric and blocking recall |
| `submit.py` | output TSVs and official validator |
| `experiments/` | one-off studies (recall of each retriever, probability calibration, diagnostics); results are in ARCHITECTURE.md |

## Compliance
Uses only the provided data. No external lookups, geocoding or APIs. Hand-written normalisation
lexicons contain general language knowledge only (abbreviations, legal-form spellings, state names).
Models: LightGBM (MIT) and `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0, 118M parameters),
used as the bi-encoder and, separately fine-tuned, as the cross-encoder. The base weights are downloaded from the
Hugging Face hub once at first use; nothing else touches the network.
