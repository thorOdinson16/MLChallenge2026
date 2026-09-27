#!/usr/bin/env bash
# Wire one or more new cross-encoders into a per-country candidate, fully automatic:
#   bash wire_final.sh <out_name> <new_tag>[,<new_tag>...]
# 1) moves <tag>_{holdout,test}.parquet from the repo root into work/preds if found there
# 2) US/IN gate: stage 3 with sub_N's CE set + new tags; used only if BOTH full and thinned beat sub_N (0.99139 / 0.99103)
# 3) France: trust rule on all France-adapted CEs (+ new tags, + ce_xlmr_fr4 / ce_full2 if present), 3-seed average
# 4) splice, full noise-word rescue last, validator with the candidate file
set -u
ROOT=${BER_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}
SRC=$ROOT/code/business_entity_resolution/src
LOGS=$ROOT/work/logs
PREDS=$ROOT/work/preds
OUT=$ROOT/output
NAME=$1; NEW=$2
export PYTHONPATH=$SRC PYTHONIOENCODING=utf-8 BER_S3_SEEDS=3
cd $SRC
dedupe() { echo "$1" | tr ',' '
' | awk 'NF && !s[$0]++' | paste -sd, - ; }
say() { echo "[$(date +%H:%M:%S)] [wire $NAME] $*" | tee -a $LOGS/wire_$NAME.log; }
for t in ${NEW//,/ }; do
  for w in holdout test; do [ -f $ROOT/${t}_$w.parquet ] && mv $ROOT/${t}_$w.parquet $PREDS/; done
  [ -f $PREDS/${t}_holdout.parquet ] && [ -f $PREDS/${t}_test.parquet ] || { say "missing files for $t"; exit 1; }
done
FR=$(dedupe ce_v5,ce_xlmr_fr3,ce_e5fr,$NEW)
for t in ce_xlmr_fr4 ce_full2; do [ -f $PREDS/${t}_test.parquet ] && [[ ",$FR," != *",$t,"* ]] && FR=$FR,$t; done
FRONLY=",ce_full2,"   # models whose TEST scores cover France only: France stage 3 only, never the US/IN gate
USTAGS=$(dedupe ce_v4,ce_xlmr_fr,ce_big,ce_full,$NEW | tr ',' '
' | while read t; do [[ "$FRONLY" == *",$t,"* ]] || echo "$t"; done | paste -sd, -)
say "US/IN gate with $USTAGS ; France set $FR"
python -u -m ber.experiments.overnight ${NAME}_usin $USTAGS > $LOGS/wire_${NAME}_usin.log 2>&1 &
G=$!
python -u -m ber.experiments.fr_trust ${NAME}_fr $OUT/sub_N.tsv $FR > $LOGS/wire_${NAME}_fr.log 2>&1
wait $G
C=$(grep "CHOSEN" $LOGS/wire_${NAME}_usin.log | sed -E 's/.*full ([0-9.]+), thinned ([0-9.]+).*/\1 \2/')
say "US/IN gate result (full thinned): $C   vs sub_N 0.99139 0.99103"
USIN=sub_N.tsv
if python -c "import sys; f,t=map(float,'$C'.split()); sys.exit(0 if f>=0.99139 and t>=0.99103 and (f>0.99139 or t>0.99103) else 1)"; then USIN=sub_${NAME}_usin.tsv; say "new US/IN accepted"; else say "keeping sub_N for US/IN"; fi
python - <<EOF
import polars as pl
from ber import config, io
s1 = io.load("test", "s1", ["eid", "country"]).rename({"eid": "source1_entity_id"})
u = pl.read_csv(str(config.OUT / "$USIN"), separator="\t", infer_schema=False).with_columns(pl.col("matched_entity_ids").fill_null(""))
f = pl.read_csv(str(config.OUT / "${NAME}_fr.tsv"), separator="\t", infer_schema=False).with_columns(pl.col("matched_entity_ids").fill_null(""))
m = f.join(u.rename({"matched_entity_ids": "u"}), on="source1_entity_id").join(s1, on="source1_entity_id")
m = m.with_columns(o=pl.when(pl.col("country") == "France").then(pl.col("matched_entity_ids")).otherwise(pl.col("u")))
with open(str(config.OUT / "${NAME}.tsv"), "w", encoding="utf-8", newline="\n") as fh:
    fh.write("source1_entity_id\tmatched_entity_ids\n")
    for a, b in f.select("source1_entity_id").join(m.select("source1_entity_id", "o"), on="source1_entity_id").iter_rows():
        fh.write(f"{a}\t{b}\n")
EOF
python -m ber.cn_rescue $OUT/${NAME}.tsv $OUT/${NAME}cnF.tsv cn_rescue_pairs_full.parquet 2>&1 | grep -v Warn | tee -a $LOGS/wire_$NAME.log
python - <<EOF | tee -a $LOGS/wire_$NAME.log
import subprocess, sys, hashlib
from ber import config
o = config.OUT / "${NAME}cnF.tsv"
r = subprocess.run([sys.executable, str(config.DATA.parent / "utils" / "validate_submission.py"), "--matching", str(o), "--candidate", str(config.OUT / "candidate_pairs.tsv"), "--test-dir", str(config.DATA / "test")], capture_output=True, text=True, encoding="utf-8", errors="replace")
print("FINAL", o.name, "validator+candidates", "PASS" if r.returncode == 0 else "FAIL", hashlib.md5(o.read_bytes()).hexdigest())
EOF
grep -v Warn $LOGS/wire_${NAME}_fr.log | grep "France\|trust vs" | tee -a $LOGS/wire_$NAME.log
say "DONE"
