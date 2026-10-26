#!/bin/bash
# Deferred restore, extra variants (youngjoon, 10/24), eslab07 GPU 0: fp32 snapshots (no quantisation) and the
# committed state itself kept 8-bit / 4-bit. Smoke, same-session speed at batch 1/8/32 (2 runs, SpecLA at start,
# middle, end), GSM8K accuracy at batch 32, max_new 512.
cd "$(dirname "$0")"
CT=${CT:-q4/bench_cost_rtx3090.json}
SM=results_more_smoke
OUT=results_session_more
smoke() {  # name rollback extra...
  local name=$1; shift
  GPU=${GPU:-0} src/run_gpu.sh logs/graph/more_smoke_$name.log .venv/bin/python run_all.py --engine graph --rollback "$@" \
    --cost-table $CT --phases 0,4 --batch-sizes 8 --methods dflash --runs 1 --warmup 1 --limit 32 \
    --out $SM/$name || { echo "[more] smoke $name FAILED"; exit 1; }
  echo "[more] smoke $name ok $(TZ=Asia/Seoul date +%H:%M:%S)"
}
run() {  # name rollback extra...
  local name=$1; shift
  GPU=${GPU:-0} src/run_gpu.sh logs/graph/more_$name.log .venv/bin/python run_all.py --engine graph --rollback "$@" \
    --cost-table $CT --phases 0,4 --batch-sizes 1,8,32 --methods dflash --runs 2 --out $OUT/$name \
    || { echo "[more] $name FAILED"; exit 1; }
  echo "[more] $name done $(TZ=Asia/Seoul date +%H:%M:%S)"
}
acc() {  # name rollback extra...
  local name=$1; shift
  GPU=${GPU:-0} src/run_gpu.sh logs/graph/acc_$name.log .venv/bin/python run_all.py --engine graph --rollback "$@" \
    --cost-table $CT --phases 0,4 --batch-sizes 32 --methods dflash --runs 1 --max-new 512 --out results_acc512/$name \
    || { echo "[more] acc $name FAILED"; exit 1; }
  echo "[more] acc $name done $(TZ=Asia/Seoul date +%H:%M:%S)"
}
echo "[more] start $(TZ=Asia/Seoul date +%H:%M:%S)"
rm -rf $SM
smoke specla specla
smoke confsm_d32 confsm_d32
for m in oracle_dq8c oracle_dq4c; do smoke $m $m --oracle-from $SM/specla; done
for m in confsm_dq8c hist_dq8c confsm_dq4c hist_dq4c; do smoke $m $m; done
python3 - <<'PY'
import json
ld = lambda d: {r["order"]: r["output"] for r in map(json.loads, open(f"results_more_smoke/{d}/raw/dflash_bs8_run1.jsonl"))}
a, b = ld("specla"), ld("confsm_d32")
print(f"[more] fp32 deferred softmax vs SpecLA: {sum(a[o] == b[o] for o in a)}/{len(a)} identical")
PY
mkdir -p $OUT
TR=$OUT/specla_n
run specla_n specla
run oracle_d32 oracle_d32 --oracle-from $TR
run confsm_d32 confsm_d32
run oracle_dq8c oracle_dq8c --oracle-from $TR
run confsm_dq8c confsm_dq8c
run hist_dq8c hist_dq8c
run specla_o specla
run oracle_dq4c oracle_dq4c --oracle-from $TR
run confsm_dq4c confsm_dq4c
run hist_dq4c hist_dq4c
run specla_p specla
for m in oracle_dq8c oracle_dq4c; do acc $m $m --oracle-from results_acc512/specla; done
for m in confsm_dq8c hist_dq8c confsm_dq4c hist_dq4c; do acc $m $m; done
echo "[more] all done $(TZ=Asia/Seoul date +%H:%M:%S)"
