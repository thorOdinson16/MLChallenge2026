# AnchorMatch — Business Entity Resolution pipeline

**Team Oura Oura** — Abhijna D S, Abhijnan B C, A K Pranav, Abyud Somashekara Shetty

Design, EDA findings and the experiment log live in [ARCHITECTURE.md](ARCHITECTURE.md).

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.10
pip install -r requirements.txt                         # install the CUDA torch build first (see file)
```
Hardware used: 16 GB RAM, one 6 GB NVIDIA GPU (RTX 4050 Laptop), about 60 GB free disk. The full run from
raw data takes about 7 hours; run GPU steps one at a time.
Everything streams in chunks, so larger machines only run faster.

## Data locations
By default the code expects the challenge folder layout:
```
<project>/student_resource/dataset/{train,test}/...   input TSVs
<project>/work/                                       caches (created)
<project>/output/                                     matching_results.tsv, candidate_pairs.tsv
```
To use other locations, set `BER_DATA`, `BER_WORK` or `BER_OUT`.

## Run end to end
```bash
cd src
python -m ber.pipeline all
# or step by step (each step caches its result under work/):
python -m ber.pipeline prepare   # TSV -> parquet, learned transliteration dict, normalisation
python -m ber.pipeline encoder   # fine-tune multilingual MiniLM bi-encoder on ENC folds (about 45 min on the GPU)
python -m ber.pipeline retrieve  # un-pruned retriever passes (LSA, keys, encoder) for train + test (about 4 h)
python -m ber.pipeline prune     # candidate sample -> stage-0 pruner
python -m ber.pipeline cands     # merge + prune -> final candidate set (candidate_pairs.tsv)
python -m ber.pipeline feats     # pair features (TRAIN_X, EVAL_X, test)
python -m ber.pipeline train     # 2-stage LightGBM matcher, prints holdout macro-F0.5
python -m ber.pipeline submit    # test predictions -> decoding -> output/*.tsv -> official validator
```
`run.sh` is a thin wrapper that sets the environment variables used on the author's machine.

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
| `features.py` | about 60 language-agnostic pair features |
| `matcher.py` | stage-1 LightGBM (OOF) and stage-2 context/cluster LightGBM |
| `decode.py` | one S1 per record, then per-entity expected-F0.5-optimal selection |
| `evaluate.py` | exact macro-F0.5 metric and blocking recall |
| `submit.py` | output TSVs and official validator |

## Compliance
Uses only the provided data. No external lookups, geocoding or APIs. Hand-written normalisation
lexicons contain general language knowledge only (abbreviations, legal-form spellings, state names).
Models: LightGBM (MIT) and `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
(Apache-2.0, 118M parameters).
