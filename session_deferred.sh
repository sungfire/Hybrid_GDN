#!/bin/bash
# Deferred restore with 8-bit and 4-bit snapshots, our position-prediction methods (youngjoon, 10/24), eslab07 GPU 0.
# Buffers of all GDN layers in one tensor per kind (retiring a row moves 9 tensors). (1) smoke + correctness (fp32
# deferred oracle == SpecLA outputs) + remove-section check, (2) same-session speed at batch 1/8/32, 2 runs each,
# SpecLA at the start, middle and end, (3) GSM8K accuracy at batch 32, max_new 512.
cd "$(dirname "$0")"
CT=${CT:-q4/bench_cost_rtx3090.json}
SM=results_def_smoke
OUT=results_session_def
smoke() {  # name rollback extra...
  local name=$1; shift
  GPU=${GPU:-0} src/run_gpu.sh logs/graph/def_smoke_$name.log .venv/bin/python run_all.py --engine graph --rollback "$@" \
    --cost-table $CT --phases 0,4 --batch-sizes 8 --methods dflash --runs 1 --warmup 1 --limit 32 \
    --out $SM/$name || { echo "[def] smoke $name FAILED"; exit 1; }
  echo "[def] smoke $name ok $(TZ=Asia/Seoul date +%H:%M:%S)"
}
run() {  # name rollback extra...
  local name=$1; shift
  GPU=${GPU:-0} src/run_gpu.sh logs/graph/def_$name.log .venv/bin/python run_all.py --engine graph --rollback "$@" \
    --cost-table $CT --phases 0,4 --batch-sizes 1,8,32 --methods dflash --runs 2 --out $OUT/$name \
    || { echo "[def] $name FAILED"; exit 1; }
  echo "[def] $name done $(TZ=Asia/Seoul date +%H:%M:%S)"
}
acc() {  # name rollback extra...
  local name=$1; shift
  GPU=${GPU:-0} src/run_gpu.sh logs/graph/acc_$name.log .venv/bin/python run_all.py --engine graph --rollback "$@" \
    --cost-table $CT --phases 0,4 --batch-sizes 32 --methods dflash --runs 1 --max-new 512 --out results_acc512/$name \
    || { echo "[def] acc $name FAILED"; exit 1; }
  echo "[def] acc $name done $(TZ=Asia/Seoul date +%H:%M:%S)"
}
echo "[def] start $(TZ=Asia/Seoul date +%H:%M:%S)"
rm -rf $SM
smoke specla specla
smoke oracle_d32 oracle_d32 --oracle-from $SM/specla
smoke oracle_dq8 oracle_dq8 --oracle-from $SM/specla
python3 - <<'PY' || { echo "[def] smoke check FAILED, stop"; exit 1; }
import json
ld = lambda d: {r["order"]: r["output"] for r in map(json.loads, open(f"results_def_smoke/{d}/raw/dflash_bs8_run1.jsonl"))}
a, b = ld("specla"), ld("oracle_d32")
same = sum(a[o] == b[o] for o in a)
rm = {}
for m in ("specla", "oracle_dq8"):
    r = json.load(open(f"results_def_smoke/{m}/cells/dflash_bs8.json"))["runs"][0]
    rm[m] = r["sections"].get("remove", 0) / r["steps"] * 1e3
print(f"[def] fp32 deferred oracle vs SpecLA: {same}/{len(a)} identical | remove ms/step: SpecLA {rm['specla']:.3f}, "
      f"oracle_dq8 {rm['oracle_dq8']:.3f}")
raise SystemExit(0 if same == len(a) else 1)
PY
mkdir -p $OUT
TR=$OUT/specla_k
run specla_k specla
run oracle_dq8 oracle_dq8 --oracle-from $TR
run confsm_dq8 confsm_dq8
run hist_dq8 hist_dq8
run confhead_dq8 confhead_dq8
run confsm_q8 confsm_q8
run specla_l specla
run oracle_dq4 oracle_dq4 --oracle-from $TR
run confsm_dq4 confsm_dq4
run hist_dq4 hist_dq4
run confhead_dq4 confhead_dq4
run specla_m specla
acc oracle_dq4 oracle_dq4 --oracle-from results_acc512/specla
for m in confsm_dq8 hist_dq8 confhead_dq8 confsm_dq4 hist_dq4 confhead_dq4; do acc $m $m; done
echo "[def] all done $(TZ=Asia/Seoul date +%H:%M:%S)"
