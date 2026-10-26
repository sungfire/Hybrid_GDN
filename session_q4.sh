#!/bin/bash
# Same-session measurement of the slot strategies with fused restore, fp32 vs 4-bit snapshots, eslab07 GPU 0
# (youngjoon, 10/24). SpecLA at both ends; oracle traces from this session's first SpecLA run (same engine, same bs).
cd "$(dirname "$0")"
OUT=${OUT:-results_session_q4}
RUNS=${RUNS:-3}
BS=${BS:-1,8,32}
run() {  # name rollback extra...
  local name=$1; shift
  GPU=${GPU:-0} src/run_gpu.sh logs/graph/q4_$name.log .venv/bin/python run_all.py --engine graph --rollback "$@" \
    --phases 0,4 --batch-sizes $BS --methods dflash --runs $RUNS --out $OUT/$name \
    || { echo "[session] $name FAILED"; exit 1; }
  echo "[session] $name done $(TZ=Asia/Seoul date +%H:%M:%S)"
}
TR=$OUT/specla_c
mkdir -p $OUT
echo "[session] start $(TZ=Asia/Seoul date +%H:%M:%S)"
run specla_c specla
run oracle_fused oracle_fused --oracle-from $TR
run oracle_fused_q4 oracle_fused_q4 --oracle-from $TR
run fixed_q4_s2 fixed_q4 --slots 2
run fixed_q4_s6 fixed_q4 --slots 6
run fixed_q4_s16 fixed_q4 --slots 16
run specla_d specla
echo "[session] all done $(TZ=Asia/Seoul date +%H:%M:%S)"
