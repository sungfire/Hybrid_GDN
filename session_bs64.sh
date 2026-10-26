#!/bin/bash
# Batch 64, GSM8K 256 (youngjoon, 10/25): smoke (memory), then SpecLA, the given methods, SpecLA again; 3 runs each.
# One copy per GPU, e.g. ./session_bs64.sh 0 confsm_dq8 oracle_dq8 and ./session_bs64.sh 1 hist_dq8 confhead_dq8.
cd "$(dirname "$0")"
G=$1; shift
OUT=results_bs64/gpu$G
CT=${CT:-q4/bench_cost_rtx3090.json}
echo "[bs64 gpu$G] start $(TZ=Asia/Seoul date +%H:%M:%S)"
GPU=$G src/run_gpu.sh logs/graph/bs64_smoke_gpu$G.log .venv/bin/python run_all.py --engine graph --rollback $1 \
  --cost-table $CT --phases 0,4 --batch-sizes 64 --methods dflash --runs 1 --warmup 1 --limit 64 --out $OUT/smoke_$1 \
  || { echo "[bs64 gpu$G] smoke FAILED"; exit 1; }
python3 -c "import json; c=json.load(open('$OUT/smoke_$1/cells/dflash_bs64.json')); print('[bs64 gpu$G] smoke', c['status'], 'peak %.1f GB' % c['peak_mem_gb'])"
run() {  # name rollback extra...
  local name=$1; shift
  GPU=$G src/run_gpu.sh logs/graph/bs64_gpu${G}_$name.log .venv/bin/python run_all.py --engine graph --rollback "$@" \
    --cost-table $CT --phases 0,4 --batch-sizes 64 --methods dflash --runs 3 --out $OUT/$name \
    || { echo "[bs64 gpu$G] $name FAILED"; exit 1; }
  echo "[bs64 gpu$G] $name done $(TZ=Asia/Seoul date +%H:%M:%S)"
}
run specla_a specla
for m in "$@"; do
  case $m in oracle_*) run $m $m --oracle-from $OUT/specla_a ;; *) run $m $m ;; esac
done
run specla_b specla
echo "[bs64 gpu$G] all done $(TZ=Asia/Seoul date +%H:%M:%S)"
