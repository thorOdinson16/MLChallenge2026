#!/usr/bin/env bash
# Post-steps on a wire_final build (S3 recipe, LB 0.988318) -> safe + gamble files and the matching candidate file:
#   bash post_final.sh <NAME>     (needs output/<NAME>cnF.tsv, preds/test_fr_trust_<NAME>_fr.parquet)
# safe   = <NAME>cnF + acronym v2 (recomputed) -> tie-trim France (recomputed, .85/.6) -> twin US/IN
# gamble = safe + France clear-winner adds (pD >= 0.3, share >= 0.75, unassigned X)
# cands  = candidate_pairs_orig + acronym rule-B pairs + any matched pair still missing (dup propagation)
set -u
ROOT=${BER_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}
SRC=$ROOT/code/business_entity_resolution/src
LOGS=$ROOT/work/logs
NAME=$1
export PYTHONPATH=$SRC PYTHONIOENCODING=utf-8
cd $SRC
O=$(cd $ROOT/output && pwd -W); P=$(cd $ROOT/work/preds && pwd -W)
say() { echo "[$(date +%H:%M:%S)] [post $NAME] $*" | tee -a $LOGS/post_$NAME.log; }
FRP="$P/test_fr_trust_${NAME}_fr.parquet"
if grep -q "new US/IN accepted" $LOGS/wire_$NAME.log; then TW="$P/test_s3_${NAME}_usin.parquet"; else TW="$P/test_s3_N.parquet"; fi
say "twin scores $TW"
python -m ber.acronym pairs "$FRP" pD 0.85 2>&1 | grep -v Warn | tail -5 | tee -a $LOGS/post_$NAME.log
python -m ber.tie_trim pairs "$FRP" pD 0.85 0.6 2>&1 | grep tie_trim | tee -a $LOGS/post_$NAME.log
python -m ber.cn_rescue "$O/${NAME}cnF.tsv" "$O/${NAME}_a.tsv" fr_acronym_pairs.parquet 2>&1 | grep "cn_rescue\|validator" | tee -a $LOGS/post_$NAME.log
python -m ber.tie_trim apply "$O/${NAME}_a.tsv" "$O/${NAME}_b.tsv" 2>&1 | grep "tie_trim\|validator" | tee -a $LOGS/post_$NAME.log
python -m ber.twin "$O/${NAME}_b.tsv" "$O/${NAME}_SAFE.tsv" "$TW" p3 US,India 2>&1 | grep "twin\|validator" | tee -a $LOGS/post_$NAME.log
python -m ber.experiments.gamble "$FRP" 0.3 0.75 2>&1 | grep gamble | cut -c1-80 | tee -a $LOGS/post_$NAME.log
python -m ber.cn_rescue "$O/${NAME}_SAFE.tsv" "$O/${NAME}_GAMBLE.tsv" fr_gamble_pairs.parquet 2>&1 | grep "cn_rescue\|validator" | tee -a $LOGS/post_$NAME.log
python -m ber.acronym cands "$O/candidate_pairs_orig.tsv" "$O/cands_${NAME}.tsv" 2>&1 | grep cands | tee -a $LOGS/post_$NAME.log
python - <<EOF 2>&1 | grep -v Warn | tee -a $LOGS/post_$NAME.log
import hashlib, subprocess, sys
import polars as pl
from ber import config
def ex(p, c):
    d = pl.read_csv(p, separator="\t", infer_schema=False).with_columns(pl.col(c).fill_null(""))
    return d, d.filter(pl.col(c) != "").with_columns(pl.col(c).str.split(",")).explode(c).rename({c: "xe"})
cp = config.OUT / "cands_${NAME}.tsv"
c, ce = ex(cp, "candidate_entity_ids")
miss = pl.concat([ex(config.OUT / f"${NAME}_{k}.tsv", "matched_entity_ids")[1] for k in ("SAFE", "GAMBLE")]).unique()
miss = miss.join(ce, on=["source1_entity_id", "xe"], how="anti")
print("matched pairs added to candidates:", miss.height)
if miss.height:
    ch = miss["source1_entity_id"].unique()
    lists = pl.concat([ce.filter(pl.col("source1_entity_id").is_in(ch)), miss]).sort("xe").group_by("source1_entity_id").agg(pl.col("xe").str.join(",").alias("m"))
    c = c.join(lists, on="source1_entity_id", how="left").with_columns(candidate_entity_ids=pl.coalesce("m", "candidate_entity_ids")).drop("m")
    with open(cp, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for a, b in c.iter_rows():
            f.write(f"{a}\t{b}\n")
for k in ("SAFE", "GAMBLE"):
    o = config.OUT / f"${NAME}_{k}.tsv"
    r = subprocess.run([sys.executable, str(config.DATA.parent / "utils" / "validate_submission.py"), "--matching", str(o), "--candidate", str(cp), "--test-dir", str(config.DATA / "test")], capture_output=True, text=True, encoding="utf-8", errors="replace")
    warn = [l for l in r.stdout.splitlines() if "not present in candidate" in l]
    print("FINAL", o.name, "PASS" if r.returncode == 0 else "FAIL", "cand-warn" if warn else "clean", hashlib.md5(o.read_bytes()).hexdigest())
EOF
say "DONE"
