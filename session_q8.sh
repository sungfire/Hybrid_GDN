#!/bin/bash
# 8-bit snapshots (youngjoon, 10/24), eslab07 GPU 0. (1) GSM8K accuracy, batch 32, max_new 512, one run, against
# results_acc512/specla (exact output, oracle trace). (2) Same-session speed, batch 1/8/32, SpecLA at both ends;
# oracle_fused_q4 again because the kept rows are now converted to int32 once per step (was once per layer).
cd "$(dirname "$0")"
acc() {  # name rollback extra...
  local name=$1; shift
  GPU=${GPU:-0} src/run_gpu.sh logs/graph/acc_$name.log .venv/bin/python run_all.py --engine graph --rollback "$@" \
    --phases 0,4 --batch-sizes 32 --methods dflash --runs 1 --max-new 512 --out results_acc512/$name \
    || { echo "[q8] acc $name FAILED"; exit 1; }
  echo "[q8] acc $name done $(TZ=Asia/Seoul date +%H:%M:%S)"
}
run() {  # name rollback extra...
  local name=$1; shift
  GPU=${GPU:-0} src/run_gpu.sh logs/graph/q8_$name.log .venv/bin/python run_all.py --engine graph --rollback "$@" \
    --phases 0,4 --batch-sizes 1,8,32 --methods dflash --runs 3 --out results_session_q8/$name \
    || { echo "[q8] $name FAILED"; exit 1; }
  echo "[q8] $name done $(TZ=Asia/Seoul date +%H:%M:%S)"
}
echo "[q8] start $(TZ=Asia/Seoul date +%H:%M:%S)"
acc oracle_fused_q8 oracle_fused_q8 --oracle-from results_acc512/specla
acc fixed_q8_s2 fixed_q8 --slots 2
acc fixed_q8_s6 fixed_q8 --slots 6
acc fixed_q8_s16 fixed_q8 --slots 16
mkdir -p results_session_q8
TR=results_session_q8/specla_e
run specla_e specla
run oracle_fused_q8 oracle_fused_q8 --oracle-from $TR
run oracle_fused_q4 oracle_fused_q4 --oracle-from $TR
run fixed_q8_s2 fixed_q8 --slots 2
run fixed_q8_s6 fixed_q8 --slots 6
run specla_f specla
echo "[q8] all done $(TZ=Asia/Seoul date +%H:%M:%S)"
