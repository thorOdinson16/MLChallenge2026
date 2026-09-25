# Amazon ML Challenge 2026 — Business Entity Resolution

Given noisy business records from three independent sources (S1 = clean deduplicated
reference, S2/S3 = noisy), find which S2/S3 records refer to the same real-world
business as each S1 entity. Scored by macro-averaged F0.5 per S1 entity (precision
weighted 2x over recall).

- **Problem statement**: [`docs/README.md`](docs/README.md)
- **Methodology write-up**: [`docs/Documentation_template.md`](docs/Documentation_template.md)
- **Pipeline code + reproduction steps**: [`code/business_entity_resolution/`](code/business_entity_resolution/)
- **Submission validator**: [`utils/validate_submission.py`](utils/validate_submission.py)

## Approach at a glance

Tabular ML, no graph clustering: normalize text -> block candidates via 4 cheap
retrieval channels (~65 candidates/S1 per source) -> engineer ~60 pairwise features ->
LightGBM binary classifier scores each candidate pair -> threshold + enforce one-to-one
-> emit `matching_results.tsv` and `candidate_pairs.tsv`.

Validated macro F0.5 ≈ 0.95 on a held-out 20% slice of a 100k-S1 training subsample
(threshold 0.68). See the code README for full pipeline details and exact commands to
reproduce end-to-end.

## Repo layout

```
code/business_entity_resolution/   pipeline source, README, requirements.txt
docs/                                problem statement, guidelines, methodology template
utils/                                submission format validator
run_all.bat                          blocking gate -> build train pairs -> train (Windows)
```

`dataset/`, `logs/*.parquet`, `logs/test_scored/`, `backup_v1/`, and `output/*.tsv` are
git-ignored — they're large (multi-GB), regenerable from the pipeline, and/or raw
challenge data not meant for redistribution. The final `matching_results.tsv` /
`candidate_pairs.tsv` ship in the submission zip per the challenge's required format,
not in this repo.
