"""Deck figures, one graph or table per figure (youngjoon, 10/25). Writes OUT_DIR/fig*.png.
Usage: python q4/plot_deck.py OUT_DIR"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

OUT = sys.argv[1]
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = lambda p: json.load(open(os.path.join(R, p)))  # noqa: E731
BENCH = J("q4/bench_cost_rtx3090.json")
DEF = J("results_session_def/summary.json")
MORE = J("results_session_more/summary.json")
ACC = J("results_acc_all/acc_ci.json")
PPL = J("results_ppl/ppl.json")["runs"]
L = 24

names = [f.name for f in font_manager.fontManager.ttflist]
for cand in ("Noto Sans CJK KR", "Noto Sans CJK JP", "Noto Sans CJK SC"):
    if cand in names:
        plt.rcParams["font.family"] = cand
        break
plt.rcParams.update({"font.size": 12, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#898781", "figure.dpi": 200})
INK, SEC, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
C = {"specla": "#0b0b0b", "fp32": "#898781", "8bit": "#2a78d6", "4bit": "#eb6834"}
METH = [("oracle", "#0e3a73"), ("softmax 기반", "#2a78d6"), ("최빈값 기반", "#7fb0ea"), ("학습 head 기반", "#4f8fd1")]


def style(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def save(fig, name, note=None):
    if note:
        fig.text(0.01, -0.03, note, fontsize=9.5, color=SEC)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("written", name, flush=True)


# ---------- fig1: ideal (kernel times), oracle fp32 / 8bit / 4bit vs SpecLA, one figure per batch
for b in ("1", "8", "32"):
    t = BENCH[b]
    v0 = t["verify, no store"]
    als = list(range(1, 17))
    spec = [L * t[f"replay {a} tok from S0"] / 1e3 for a in als]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(als, spec, color=C["specla"], lw=2.4, label="SpecLA (원본): 수락 토큰 재계산")
    for kind, ls in (("fp32", "--"), ("8bit", "-"), ("4bit", "-")):
        w = sum(t[f"verify_slots {kind} (AL {a})"] for a in (2, 6, 12)) / 3 - v0
        val = L * (w + t[f"fused {kind} snap + 0 tok"]) / 1e3
        cross = next((a for a, s in zip(als, spec) if s > val), None)
        gain = f"AL ≥ {cross}부터 이득" if cross is not None else "이득 없음"
        name = "fp32 (양자화 없음)" if kind == "fp32" else kind
        ax.plot(als, [val] * 16, color=C[kind], lw=2.2, ls=ls, label=f"oracle {name}: {val:.2f} ms ({gain})")
    ax.set_xlabel("수락 길이 AL")
    ax.set_ylabel("라운드당 rollback 비용 (ms)")
    ax.set_xticks([1, 4, 8, 12, 16])
    ax.set_title(f"이상적 경우 (정답 위치를 앎), batch {b}", fontsize=13, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    style(ax, "both")
    save(fig, f"fig1_ideal_bs{b}.png", "GDN 24층 커널 시간 (RTX 3090, CUDA graph). 비용 = 검증 중 저장 + 저장본에서 복원. "
         "모델 가중치는 bf16, GDN state는 원래 fp32.")

# ---------- fig2: real (engine), rollback cost per step with SpecLA = 100%
fig, ax = plt.subplots(figsize=(8, 4.6))
rows = [("oracle fp32 (양자화 없음)", MORE, "oracle_d32", C["fp32"]), ("oracle 8bit", DEF, "oracle_dq8", C["8bit"]),
        ("oracle 4bit", DEF, "oracle_dq4", C["4bit"])]
w = 0.22
for i, b in enumerate((1, 8, 32)):
    ref = DEF[f"specla_k_bs{b}"]["restore"]
    ax.bar(i - 1.5 * w, 100, width=w * 0.92, color=C["specla"], label="SpecLA (원본)" if i == 0 else None)
    ax.text(i - 1.5 * w, 101, f"{ref:.2f} ms", ha="center", va="bottom", fontsize=8.5, color=SEC)
    for j, (lab, S, key, col) in enumerate(rows):
        v = S[f"{key}_bs{b}"]
        sp_restore = v["restore"] - v["d"]["restore"]
        cost = v["d"]["verify"] + v["restore"]
        y = 100 * cost / sp_restore
        ax.bar(i + (j - 0.5) * w, y, width=w * 0.92, color=col, label=lab if i == 0 else None)
        ax.text(i + (j - 0.5) * w, y + 1, f"{y:.0f}%", ha="center", va="bottom", fontsize=9, color=INK)
ax.set_xticks(range(3))
ax.set_xticklabels(["batch 1", "batch 8", "batch 32"])
ax.set_ylabel("step당 rollback 비용 (SpecLA = 100%)")
ax.set_ylim(0, 125)
ax.set_title("실제 엔진: 정답 위치 저장의 rollback 비용", fontsize=13, color=INK, loc="left")
ax.legend(frameon=False, fontsize=9.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
style(ax)
save(fig, "fig2_real_oracle_cost.png", "rollback 비용 = (검증 시간 증가) + 복원 시간, 같은 세션 SpecLA 대비. "
     "SpecLA 위 숫자는 SpecLA의 복원 시간.")

# ---------- fig3: result, throughput normalised to SpecLA (8-bit methods)
fig, ax = plt.subplots(figsize=(8.5, 4.6))
keys = {"oracle": "oracle_dq8", "softmax 기반": "confsm_dq8", "최빈값 기반": "hist_dq8", "학습 head 기반": "confhead_dq8"}
w = 0.18
for i, b in enumerate((1, 8, 32)):
    for j, (lab, col) in enumerate(METH):
        y = 1 + DEF[f"{keys[lab]}_bs{b}"]["d_tps_pct"] / 100
        x = i + (j - 1.5) * w
        ax.bar(x, y - 1, bottom=1, width=w * 0.92, color=col, label=lab if i == 0 else None)
        ax.text(x, y + 0.001, f"{y:.3f}", ha="center", va="bottom", fontsize=8.5, color=INK)
ax.axhline(1, color=INK, lw=1.4)
ax.text(-0.45, 0.9993, "SpecLA (원본) = 1.000", ha="left", va="top", fontsize=9.5, color=INK)
ax.set_ylim(0.995, 1.032)
ax.set_xticks(range(3))
ax.set_xticklabels(["batch 1", "batch 8", "batch 32"])
ax.set_ylabel("처리량 (SpecLA = 1)")
ax.set_title("결과: 8bit 저장 + 위치 예측, 원본 SpecLA 대비 처리량", fontsize=13, color=INK, loc="left")
ax.legend(frameon=False, fontsize=9.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
style(ax)
save(fig, "fig3_result_tps.png", "같은 세션에서 SpecLA를 처음·중간·끝에 재고 시간에 맞춰 보간한 기준. GSM8K 256문항, 2회 평균. "
     "oracle = 정답 위치(상한).")

# ---------- fig4: accuracy table (GSM8K 1319 + PPL)
fig, ax = plt.subplots(figsize=(10, 2.4))
ax.axis("off")
sp = ACC["specla"]
r8, r4 = ACC["confsm_dq8"], ACC["confsm_dq4"]
p = {k: v["ppl"] for k, v in PPL.items()}
cells = [["SpecLA (원본, bf16)", f"{sp['acc']:.2f}%", "기준", "100%", f"{p['every6_fp32']:.4f}", f"{p['every2_fp32']:.4f}"],
         ["8bit 저장", f"{r8['acc']:.2f}%", f"{r8['diff']:+.2f}%p (±{r8['ci95']:.2f})", f"{r8['identical']:.0f}%",
          f"{p['every6_q8']:.4f} ({100 * (p['every6_q8'] / p['every6_fp32'] - 1):+.2f}%)",
          f"{p['every2_q8']:.4f} ({100 * (p['every2_q8'] / p['every2_fp32'] - 1):+.2f}%)"],
         ["4bit 저장", f"{r4['acc']:.2f}%", f"{r4['diff']:+.2f}%p (±{r4['ci95']:.2f})", f"{r4['identical']:.0f}%",
          f"{p['every6_q4']:.4f} ({100 * (p['every6_q4'] / p['every6_fp32'] - 1):+.2f}%)",
          f"{p['every2_q4']:.4f} ({100 * (p['every2_q4'] / p['every2_fp32'] - 1):+.2f}%)"]]
cols = ["", "GSM8K 정답률\n(1319문항)", "차이\n(95% 구간)", "출력 완전 일치", "PPL\n(6토큰마다 양자화)", "PPL\n(2토큰마다, 가혹)"]
tb = ax.table(cellText=cells, colLabels=cols, loc="center", cellLoc="center")
tb.auto_set_font_size(False)
tb.set_fontsize(10.5)
tb.scale(1, 2.0)
for (r, c), cell in tb.get_celld().items():
    cell.set_edgecolor("#c3c2b7")
    if r == 0:
        cell.set_facecolor("#eef2f7")
        cell.set_text_props(fontweight="bold")
    if r == 2:
        cell.set_facecolor("#eaf2fc")
    if r == 3:
        cell.set_facecolor("#fdf0ea")
ax.set_title("정확도: state 저장본 8bit vs 4bit (softmax 기반, batch 32, 512토큰)", fontsize=13, color=INK, loc="left")
save(fig, "fig4_accuracy_table.png", "PPL: GSM8K 텍스트 1024토큰 × 128개, 정해진 토큰마다 GDN state를 per-channel로 양자화했다 되돌림. "
     "우리 방식은 라운드당 약 1번(평균 6토큰).")

# ---------- fig6: prediction table (batch 32)
PRED = {"oracle": (0.79, 0.90, 0.91), "softmax 기반": (0.29, 0.93, 2.46), "학습 head 기반": (0.23, 0.88, 2.80),
        "최빈값 기반": (0.14, 0.88, 4.40)}
fig, ax = plt.subplots(figsize=(9, 2.6))
ax.axis("off")
cells = [["SpecLA (원본)", "–", "–", "6.2 (수락 토큰 전부)"]] + \
        [[k, f"{v[0] * 100:.0f}%", f"{v[1] * 100:.0f}%", f"{v[2]:.1f}"] for k, v in PRED.items()]
tb = ax.table(cellText=cells, colLabels=["", "위치를 정확히 맞힘", "저장본을 쓸 수 있었음\n(실제 수락보다 앞)", "행당 재계산 토큰"],
              loc="center", cellLoc="center")
tb.auto_set_font_size(False)
tb.set_fontsize(10.5)
tb.scale(1, 1.9)
for (r, c), cell in tb.get_celld().items():
    cell.set_edgecolor("#c3c2b7")
    if r == 0:
        cell.set_facecolor("#eef2f7")
        cell.set_text_props(fontweight="bold")
ax.set_title("위치 예측: 정확히 맞히지 않아도 앞쪽 1칸이면 대부분 쓸 수 있음 (8bit, batch 32)", fontsize=13, color=INK, loc="left")
save(fig, "fig6_prediction_table.png", "oracle이 100%가 아닌 이유: 8bit 저장본으로 출력이 기록과 일부 갈라져 미리 기록한 정답 위치가 틀어짐.")

# ---------- fig7: softmax-based 8-bit vs 4-bit throughput
fig, ax = plt.subplots(figsize=(7.5, 4.4))
w = 0.32
for j, (bits, key) in enumerate((("8bit", "confsm_dq8"), ("4bit", "confsm_dq4"))):
    ys = [DEF[f"{key}_bs{b}"]["d_tps_pct"] for b in (1, 8, 32)]
    xs = [i + (j - 0.5) * w for i in range(3)]
    ax.bar(xs, ys, width=w * 0.92, color=C[bits], label=f"{bits} 저장")
    for x, y, b in zip(xs, ys, (1, 8, 32)):
        ax.text(x, y + 0.04, f"{y:+.1f}%\n(τ {DEF[f'{key}_bs{b}']['tau']:.2f})", ha="center", va="bottom", fontsize=8.5,
                color=INK)
ax.axhline(0, color=INK, lw=1.4)
ax.set_ylim(0, 3.4)
ax.set_xticks(range(3))
ax.set_xticklabels(["batch 1", "batch 8", "batch 32"])
ax.set_ylabel("처리량 변화 vs SpecLA (%)")
ax.set_title("8bit vs 4bit 처리량 (softmax 기반)", fontsize=13, color=INK, loc="left")
ax.legend(frameon=False, fontsize=10, loc="upper left")
style(ax)
save(fig, "fig7_8bit_vs_4bit_tps.png", "4bit는 저장이 조금 더 싸지만 출력이 더 많이 바뀌어 수락 길이 τ가 줄어 처리량이 낮아짐 (SpecLA τ 6.22–6.23).")
