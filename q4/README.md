# GDN state 저장본 8bit 양자화 + 위치 예측

[sungfire/Hybrid_GDN](https://github.com/sungfire/Hybrid_GDN) (Qwen3.5-4B + DFlash block 16, RTX 3090, CUDA graph 엔진) 위에서 한 실험입니다.
원래 있던 rollback(specla, confk, oracle 등)의 동작은 바꾸지 않았고, 새 rollback 이름만 추가했습니다.

**요약:** 검증 중에 GDN state를 한 위치만 8bit로 저장해 두고, 복원은 수락 직후가 아니라 다음 검증 커널을 시작할 때 저장본부터 남은 토큰만 재계산합니다.
원본 SpecLA 대비 batch 8 이상에서 1–3% 빠르고, 정확도 손실은 없습니다.

## 방법

1. **per-channel 양자화.** value 열 v마다 key 128행의 absmax로 scale을 잡습니다.
   `s_v = max_k |S[k, v]| / 127`, `q[k, v] = round(S[k, v] / s_v)`
   8bit는 −127..127이고 key 두 행을 int16 하나에 묶습니다. 4bit는 −7..7이고 key 두 행을 uint8 하나에 묶습니다.
2. **저장 위치 고르기.** 수락 길이 분포 P(A)가 주어지면, 기대 복원 비용이 가장 작은 위치 r 하나를 고릅니다. 저장하지 않는 쪽이 더 싸면 저장하지 않습니다.
   `E[none] = Σ_A P(A)·rep(A)`
   `E[r] = W + Σ_{A≤r} P(A)·rep(A) + Σ_{A>r} P(A)·fused(A−1−r)`
   여기서 rep(n)은 S0에서 n토큰을 재계산하는 비용, fused(n)은 저장본에서 n토큰을 재계산하는 비용, W는 저장 비용입니다. 셋 다 커널 시간이고 `q4/bench_cost_rtx3090.json`에서 읽습니다.
   P(A)를 얻는 방법은 세 가지입니다.
   - 최빈값 기반(`hist_*`): 지금까지 나온 수락 길이의 히스토그램을 씁니다.
   - softmax 기반(`confsm_*`): draft 토큰의 softmax 확률을 씁니다. argmax와 한 커널에서 같이 계산합니다.
   - 학습 head 기반(`confhead_*`): MLP head를 씁니다.
3. **복원을 다음 검증과 합치기.** `restore()`는 위치만 기록합니다. 다음 `gdn_verify_deferred` 커널이 시작할 때 저장본(저장본이 없으면 S0)을 읽고, 이전 block의 남은 토큰을 재계산한 뒤 바로 검증으로 넘어갑니다.

## 결과

원본 SpecLA 대비 tok/s 비율입니다. 8bit 저장을 썼고, SpecLA는 같은 세션, 같은 GPU에서 앞뒤로 측정해 시간으로 보간했습니다.

| 데이터셋 | 방식 | bs1 | bs8 | bs32 | bs64 |
|---|---|---|---|---|---|
| GSM8K 256 | oracle (상한) | 1.006 | 1.019 | 1.018 | 1.023 |
| | softmax 기반 | 1.001 | 1.020 | 1.026 | 1.016 |
| | 최빈값 기반 | 1.005 | 1.018 | 1.011 | 1.008 |
| | 학습 head 기반 | 1.004 | 1.016 | 1.010 | 1.019 |
| NQ-open 256 | oracle (상한) | 1.003 | 1.019 | 1.021 | 1.038 |
| | softmax 기반 | 1.006 | 1.030 | 1.027 | 1.033 |
| | 최빈값 기반 | 1.006 | 1.025 | 1.016 | 1.027 |
| | 학습 head 기반 | 1.001 | 1.016 | 1.022 | 1.028 |

- 평균 수락 길이 τ는 GSM8K 6.2, NQ 2.52입니다. SpecLA에서 복원이 차지하는 비중이 bs8 3.5%, bs32 3.3%라서, 이것이 이론상 최대 이득입니다.
- **정확도 (GSM8K 1319문항, bs32, 512토큰):**
  - SpecLA: 79.7%
  - softmax 기반 8bit: 81.0% (+1.3%p ±1.06)
  - softmax 기반 4bit: 78.9% (−0.8%p ±1.65)
- **PPL** (6토큰마다 state 양자화): 8bit +0.02%, 4bit +2.1%. 4bit는 τ와 PPL에서 손해를 봐서 8bit보다 느립니다.
- 확정 state까지 양자화하는 경우(`*c`)는 이득이 없었습니다.

그림과 요약 데이터는 `q4/results/`에 있습니다.

| 경로 | 내용 |
|---|---|
| `figs/` | fig0 개요, fig1 개념도, fig2–fig10 결과 |
| `results_table.csv` | 위 표의 원자료 |
| `sessions/*/summary.json` | 세션별 tok/s, τ, step당 구간 시간 |
| `accuracy/` | GSM8K 1319 정답률과 95% 구간, PPL |
| `prediction_stats.json` | 위치 예측 적중률 |

run별 cells/raw 파일은 크기 때문에 넣지 않았습니다.

## 바뀐 코드

| 파일 | 추가한 것 |
|---|---|
| `dflash_cb/kernels.py` | `gdn_verify_slots`(검증 중 fp32/8bit/4bit 저장), `gdn_replay_from`(저장본에서 재계산), `gdn_verify_deferred`(이전 block 복원 + 검증), `argmax_maxprob`, `dequant_q4/q8` |
| `dflash_cb/rollback.py` | 아래 rollback 이름들 |
| `dflash_cb/engine.py` | `need_conf == "fused"`일 때만 `argmax_maxprob` 사용 |
| `dflash_cb/data.py`, `run_all.py` | `--dataset nq:256` 또는 `specbench:<category>`, `--gsm8k-all`, `--cost-table`, `--slots` |
| `data/nq_open_dev.jsonl` | NQ-open dev 3610문항 ([google-research-datasets/nq_open](https://huggingface.co/datasets/google-research-datasets/nq_open), CC BY-SA 3.0) |

## rollback 이름

**앞부분: 위치를 고르는 방법**

| 이름 | 방법 |
|---|---|
| `oracle_*` | 정답 위치 (상한, `--oracle-from <SpecLA 결과>` 필요) |
| `confsm_*` | softmax 기반 |
| `hist_*` | 최빈값 기반 |
| `confhead_*` | 학습 head 기반 |
| `fixed*` | 고정 위치 (`--slots`) |

**뒷부분: 저장 bit와 복원 방식**

| 이름 | 뜻 |
|---|---|
| `_dq8`, `_dq4` | 8bit 또는 4bit로 저장하고, 복원을 다음 검증과 합침 |
| `_d32` | fp32로 저장하고, 복원을 다음 검증과 합침 |
| `_dq8c`, `_dq4c` | 위에 더해 확정 state도 같은 bit로 양자화 |
| `_q8`, `_q4` (`d` 없음) | 수락 직후 따로 복원 (`oracle_fused*`, `fixed_q*`, `hist_q8` 등) |

주 결과는 `confsm_dq8`입니다.

## 실행

커널 검사입니다. 마지막 줄에 `ALL CHECKS PASSED`가 나와야 합니다.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python q4/test_q4.py --bs 1,8,32
```

다른 GPU에서는 비용 표를 다시 만듭니다.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python q4/test_q4.py --bs 1,8,32 --full --json q4/bench_cost_rtx3090.json
```

한 번 돌려 보기:

```bash
GPU=0 src/run_gpu.sh logs/try.log .venv/bin/python run_all.py --engine graph --rollback confsm_dq8 --phases 0,4 --batch-sizes 8 --methods dflash --runs 1
```

결과를 만든 세션 스크립트입니다. GPU는 `GPU=N`으로 고르고, 아래 순서대로 돌립니다.

1. `session_acc.sh`: SpecLA 기준 출력과 oracle trace(`results_acc512/specla`)를 만듭니다.
2. `session_q4.sh`, `session_q8.sh`: 4bit와 8bit 저장을 수락 직후 복원으로 돌립니다.
3. `session_deferred.sh`: 주 결과입니다. 8bit/4bit 저장에 복원 합치기를 쓰고, bs 1/8/32와 GSM8K 정확도를 잽니다.
4. `session_more.sh`: fp32 저장과 확정 state 양자화를 돌립니다.
5. 나머지 세 스크립트를 돌립니다.
   - `session_bs64.sh <gpu> <방식...>`: bs64
   - `session_nq.sh <gpu> <방식...>`: NQ-open
   - `session_acc_all.sh`: GSM8K 1319문항과 PPL

결과를 정리하고 그림을 그리는 스크립트입니다.

| 스크립트 | 하는 일 |
|---|---|
| `q4/session_summary.py <세션 dir>` | 세션 안 SpecLA 대비 구간 시간 |
| `q4/compare_outputs.py <REF> <RUN> --bs 32` | 출력 일치율 |
| `q4/acc_ci.py` | 정답률 95% 구간 |
| `q4/ppl_state_quant.py` | PPL |
| `q4/plot_overview.py` | fig0 |
| `q4/plot_concept.py` | fig1 |
| `q4/plot_oracle_al.py` | 1단계 수락 길이별 비용 |
| `q4/plot_deck.py` | fig1_ideal, fig2, fig4, fig6, fig7 |
| `q4/plot_results.py` | fig3, fig8, fig9, fig10 |
