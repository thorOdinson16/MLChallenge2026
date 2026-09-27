#!/usr/bin/env bash
# S4 autopilot: L4a = S3 set + ce_r5f (already building) ; L4b = + ce_v7 (rescored) ; L4c = + ce_e5r5 (Kaggle) if its files land by 23:00
set -u
ROOT=${BER_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}
LOGS=$ROOT/work/logs
cd $ROOT/work
say() { echo "[$(date +%H:%M:%S)] [s4] $*" | tee -a $LOGS/s4.log; }
BASE=ce_bigfr,ce_full,ce_v6,ce_xlmr_r4,ce_full2,ce_r4f,ce_r5f
until grep -q "DONE" $LOGS/wire_L4a.log 2>/dev/null; do sleep 30; done
bash post_final.sh L4a > $LOGS/post_L4a.out 2>&1; grep FINAL $LOGS/post_L4a.out | tee -a $LOGS/s4.log
until grep -q "DONE ce_v7\|Error\|Traceback" $LOGS/ce_v7_score.log 2>/dev/null; do sleep 30; done
if grep -q "DONE ce_v7" $LOGS/ce_v7_score.log; then
  BASE=$BASE,ce_v7; say "building L4b with $BASE"
  bash wire_final.sh L4b $BASE > $LOGS/wire_L4b.out 2>&1 && bash post_final.sh L4b > $LOGS/post_L4b.out 2>&1
  grep FINAL $LOGS/post_L4b.out | tee -a $LOGS/s4.log
else say "ce_v7 rescore FAILED - no L4b"; fi
until { [ -f $ROOT/ce_e5r5_test.parquet ] || [ -f $ROOT/work/preds/ce_e5r5_test.parquet ]; } && { [ -f $ROOT/ce_e5r5_holdout.parquet ] || [ -f $ROOT/work/preds/ce_e5r5_holdout.parquet ]; }; do
  [ $(date +%H%M) -ge 2300 ] && { say "no E5 by 23:00 - stop"; exit 0; }; sleep 30; done
sleep 20
say "E5 landed; building L4c with $BASE,ce_e5r5"
bash wire_final.sh L4c $BASE,ce_e5r5 > $LOGS/wire_L4c.out 2>&1 && bash post_final.sh L4c > $LOGS/post_L4c.out 2>&1
grep FINAL $LOGS/post_L4c.out | tee -a $LOGS/s4.log
say "S4 DONE"
