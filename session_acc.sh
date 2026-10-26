#!/bin/bash
# GSM8K accuracy of 4-bit snapshots with full-length answers (max_new 512), batch 32, one run each, eslab07 GPU 0
# (youngjoon, 10/24). SpecLA output = exact greedy output (reference); oracle trace from that run.
cd "$(dirname "$0")"
OUT=${OUT:-results_acc512}
run() {  # name rollback extra...
  local name=$1; shift
  GPU=${GPU:-0} src/run_gpu.sh logs/graph/acc_$name.log .venv/bin/python run_all.py --engine graph --rollback "$@" \
    --phases 0,4 --batch-sizes 32 --methods dflash --runs 1 --max-new 512 --out $OUT/$name \
    || { echo "[acc] $name FAILED"; exit 1; }
  echo "[acc] $name done $(TZ=Asia/Seoul date +%H:%M:%S)"
}
mkdir -p $OUT
echo "[acc] start $(TZ=Asia/Seoul date +%H:%M:%S)"
run specla specla
run oracle_fused_q4 oracle_fused_q4 --oracle-from $OUT/specla
run fixed_q4_s2 fixed_q4 --slots 2
run fixed_q4_s6 fixed_q4 --slots 6
run fixed_q4_s16 fixed_q4 --slots 16
echo "[acc] all done $(TZ=Asia/Seoul date +%H:%M:%S)"
