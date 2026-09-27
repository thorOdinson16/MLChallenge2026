#!/usr/bin/env bash
# France domain adaptation of the cross-encoder by pseudo-labels + thinned-pool stage 3  ->  output/sub_K.tsv
# (public LB 0.986429; previous best 0.984809).  Prerequisite: steps 1-4 of README.md (v2, ce_v2, stage 3 on ce_v2).
# Run from src/.  Model weights are only read from the local Hugging Face cache after the first download.
set -eu
PREDS=$(python -c "from ber import config; print(config.path('preds', 'x').parent)")

python -m ber.experiments.thinned_sim                          # holdout graphs with 19% of S1 removed (v2 context recomputed)

# round 1: pseudo-labels from stage 3 on ce_v2
python -m ber.experiments.fr_pseudo test_m_v2_s3ce2.parquet fr_pseudo
BER_CE_EXTRA=$PREDS/fr_pseudo.parquet python -m ber.crossenc ce_v3 0.01 0.99 4 2 all          # MiniLM, 2 epochs
BER_CE_BASE=xlm-roberta-base BER_CE_BS=32 BER_CE_LR=2e-5 BER_CE_EXTRA=$PREDS/fr_pseudo.parquet \
  python -m ber.crossenc ce_xlmr_fr 0.01 0.99 4 1 all                                          # XLM-R base, 1 epoch
python -m ber.experiments.overnight I ce_v3,ce_xlmr_fr         # -> preds/test_s3_I.parquet, output/sub_I.tsv

# round 2: pseudo-labels from the adapted stage 3
python -m ber.experiments.fr_pseudo test_s3_I.parquet fr_pseudo2
BER_CE_EXTRA=$PREDS/fr_pseudo2.parquet python -m ber.crossenc ce_v4 0.01 0.99 4 2 all
python -m ber.experiments.overnight K ce_v4,ce_xlmr_fr         # -> output/sub_K.tsv (+ report in output/overnight_report.md)
