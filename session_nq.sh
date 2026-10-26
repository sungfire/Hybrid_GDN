#!/bin/bash
# QA = NQ-open dev, 256 questions (the source of Spec-Bench QA; youngjoon, 10/25): SpecLA, the given methods, SpecLA
# again at batch 1/8/32/64, 3 runs each. One copy per GPU, e.g. ./session_nq.sh 0 confsm_dq8 oracle_dq8 and
# ./session_nq.sh 1 hist_dq8 confhead_dq8.
cd "$(dirname "$0")"
G=$1; shift
OUT=results_nq/gpu$G
CT=${CT:-q4/bench_cost_rtx3090.json}
echo "[nq gpu$G] start $(TZ=Asia/Seoul date +%H:%M:%S)"
run() {  # name rollback extra...
  local name=$1; shift
  GPU=$G src/run_gpu.sh logs/graph/nq_gpu${G}_$name.log .venv/bin/python run_all.py --engine graph --rollback "$@" \
    --dataset nq:256 --cost-table $CT --phases 0,4 --batch-sizes 1,8,32,64 --methods dflash --runs 3 --out $OUT/$name \
    || { echo "[nq gpu$G] $name FAILED"; exit 1; }
  echo "[nq gpu$G] $name done $(TZ=Asia/Seoul date +%H:%M:%S)"
}
run specla_a specla
for m in "$@"; do
  case $m in oracle_*) run $m $m --oracle-from $OUT/specla_a ;; *) run $m $m ;; esac
done
run specla_b specla
echo "[nq gpu$G] all done $(TZ=Asia/Seoul date +%H:%M:%S)"
