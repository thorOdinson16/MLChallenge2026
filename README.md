# Amazon ML Challenge 2026 — Business Entity Resolution

**Team Oura Oura** — Abhijna D S, Abhijnan B C, A K Pranav, Abyud Somashekara Shetty

Match business records from source files (S2/S3) to canonical entities (S1) across US, India and France.
Metric: macro F0.5. Final public leaderboard score **0.988318** (rank ~221).

## Repository layout
| path | contents |
|---|---|
| `code/business_entity_resolution/` | the pipeline (`src/ber`), run scripts, Kaggle trainer/notebooks, rule lists |
| `code/business_entity_resolution/README.md` | setup, data locations, step-by-step run instructions |
| `code/business_entity_resolution/ARCHITECTURE.md` | design, EDA findings, experiment log |
| `output/matching_results.tsv` | final submission (LB 0.988318) |
| `output/candidate_pairs.tsv` | candidate pair set used by the final build |
| `Summary.md` | 1–2 page summary of the approach |
| `Documentation_template.md` | filled-in competition documentation |

The challenge dataset and the local caches (features, models, predictions) were removed after the competition;
rebuilding requires the original challenge data.

## Approach in brief
1. **Retrieval + pruning** — bi-encoder and lexical candidate generation, learned transliteration, normalisation.
2. **Two-stage LightGBM matcher** (v2) with full-graph context, edit-fingerprint features, source caps and deterministic decoding.
3. **Cross-encoders** (MiniLM, XLM-R, E5) on the uncertain probability band, followed by a **stage-3 LightGBM stacker**
   trained on out-of-sample holdout rows.
4. **France domain adaptation** — cross-encoders fine-tuned on confident pseudo-labels (several rounds), a trust rule
   blending with the no-matcher stack, and noise-word / acronym rescues validated on the leaderboard.
5. **Decoding** — expected-F0.5 decoder, exact-duplicate consistency, tie-trim.

## Results
| version | public LB |
|---|---|
| v1 (bi-encoder + LightGBM) | 0.975 |
| v2 | 0.982614 |
| + cross-encoder + stage 3 | 0.984809 |
| + France-adapted cross-encoders | 0.986429 |
| + noise-word rescue, trust rule, acronym rescue, tie-trim | **0.988318** |

## Reproducing
See `code/business_entity_resolution/README.md` (the exact final-build recipe is `scripts/run_s4.sh` /
`scripts/post_final.sh`). Hardware used: 16 GB RAM, one 6–8 GB GPU, ~60 GB disk.
