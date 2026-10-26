#!/bin/bash
# Accuracy under state quantisation (youngjoon, 10/25): PPL with the GDN state quantised every 6 / 2 tokens, then
# GSM8K all 1319 questions (batch 32, max_new 512) for SpecLA and softmax-based 8-bit / 4-bit, then the 95% intervals.
cd "$(dirname "$0")"
G=${GPU:-0}
CT=${CT:-q4/bench_cost_rtx3090.json}
mkdir -p results_ppl results_acc_all
echo "[acc_all] start $(TZ=Asia/Seoul date +%H:%M:%S)"
CUDA_VISIBLE_DEVICES=$G .venv/bin/python q4/ppl_state_quant.py --every 6,2 --bits 8,4 --seqs 128 --len 1024 \
  --out results_ppl/ppl.json > logs/ppl.log 2>&1 || { echo "[acc_all] ppl FAILED"; exit 1; }
echo "[acc_all] ppl done $(TZ=Asia/Seoul date +%H:%M:%S)"
for m in specla confsm_dq8 confsm_dq4; do
  GPU=$G src/run_gpu.sh logs/graph/all_$m.log .venv/bin/python run_all.py --engine graph --rollback $m \
    --cost-table $CT --phases 0,4 --batch-sizes 32 --methods dflash --runs 1 --max-new 512 --gsm8k-all \
    --out results_acc_all/$m || { echo "[acc_all] $m FAILED"; exit 1; }
  echo "[acc_all] $m done $(TZ=Asia/Seoul date +%H:%M:%S)"
done
.venv/bin/python q4/acc_ci.py results_acc_all specla confsm_dq8 confsm_dq4
echo "[acc_all] all done $(TZ=Asia/Seoul date +%H:%M:%S)"
