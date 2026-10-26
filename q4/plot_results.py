"""Result figures and tables over GSM8K and NQ, batch 1/8/32/64 (youngjoon, 10/25). Inputs: summary.json of
results_session_def (GSM8K, batch 1/8/32, one session), results_bs64/gpu{0,1} (GSM8K batch 64) and results_nq/gpu{0,1}
(NQ-open dev 256, batch 1/8/32/64). Every method is compared with SpecLA of its own session (and GPU).
Writes OUT_DIR/fig3_result_tps.png (GSM8K), fig8_result_tps_nq.png, fig9_results_table.png, fig10_results_table_paper.png.
Usage: python q4/plot_results.py OUT_DIR"""
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
KEY = {"oracle": "oracle_dq8", "softmax 기반": "confsm_dq8", "최빈값 기반": "hist_dq8", "학습 head 기반": "confhead_dq8"}
GPU = {"oracle": 0, "softmax 기반": 0, "최빈값 기반": 1, "학습 head 기반": 1}
BS = (1, 8, 32, 64)


def spec_rows(S, b):
    return [v for k, v in S.items() if k.startswith("specla") and k.endswith(f"_bs{b}")]


def table():
    """{(dataset, bs, method): dict(x=speedup vs SpecLA, tau, tps)}; method 'SpecLA' included."""
    T = {}
    d = J("results_session_def/summary.json")
    for b in (1, 8, 32):
        sp = spec_rows(d, b)
        T[("GSM8K", b, "SpecLA")] = dict(x=1.0, tau=sum(v["tau"] for v in sp) / len(sp), tps=sum(v["tps"] for v in sp) / len(sp))
        for m, k in KEY.items():
            v = d[f"{k}_bs{b}"]
            T[("GSM8K", b, m)] = dict(x=1 + v["d_tps_pct"] / 100, tau=v["tau"], tps=v["tps"])
    for ds, base, bss in (("GSM8K", "results_bs64", (64,)), ("NQ", "results_nq", BS)):
        S = {g: J(f"{base}/gpu{g}/summary.json") for g in (0, 1)}
        for b in bss:
            sp = spec_rows(S[0], b) + spec_rows(S[1], b)
            T[(ds, b, "SpecLA")] = dict(x=1.0, tau=sum(v["tau"] for v in sp) / len(sp), tps=sum(v["tps"] for v in sp) / len(sp))
            for m, k in KEY.items():
                v = S[GPU[m]][f"{k}_bs{b}"]
                T[(ds, b, m)] = dict(x=1 + v["d_tps_pct"] / 100, tau=v["tau"], tps=v["tps"])
    return T


T = table()
json.dump({f"{k[0]}_bs{k[1]}_{k[2]}": v for k, v in T.items()}, open(os.path.join(OUT, "results_table.json"), "w"), indent=1)
names = [f.name for f in font_manager.fontManager.ttflist]
for cand in ("Noto Sans CJK KR", "Noto Sans CJK JP", "Noto Sans CJK SC"):
    if cand in names:
        plt.rcParams["font.family"] = cand
        break
plt.rcParams.update({"font.size": 12, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#898781", "figure.dpi": 200})
INK, SEC, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
METH = [("oracle", "#0e3a73"), ("softmax 기반", "#2a78d6"), ("최빈값 기반", "#7fb0ea"), ("학습 head 기반", "#4f8fd1")]


def tps_figure(ds, name, title):
    fig, ax = plt.subplots(figsize=(10, 4.6))
    w = 0.18
    ys_all = []
    for i, b in enumerate(BS):
        for j, (lab, col) in enumerate(METH):
            if (ds, b, lab) not in T:
                continue
            y = T[(ds, b, lab)]["x"]
            ys_all.append(y)
            x = i + (j - 1.5) * w
            ax.bar(x, y - 1, bottom=1, width=w * 0.92, color=col, label=lab if i == 0 else None)
            ax.text(x, y + (0.0008 if y >= 1 else -0.0008), f"{y:.3f}", ha="center", va="bottom" if y >= 1 else "top",
                    fontsize=8.3, color=INK)
    ax.axhline(1, color=INK, lw=1.4)
    lo, hi = min(ys_all + [1]), max(ys_all + [1])
    ax.set_ylim(min(0.995, lo - 0.004), hi + 0.006)
    ax.text(-0.45, min(0.995, lo - 0.004) + 0.0008, "SpecLA (원본) = 1.000", ha="left", va="bottom", fontsize=9.5, color=INK)
    ax.set_xticks(range(len(BS)))
    ax.set_xticklabels([f"batch {b}" for b in BS])
    ax.set_ylabel("처리량 (SpecLA = 1)")
    ax.set_title(title, fontsize=13, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.text(0.01, -0.03, "8bit 저장 + 위치 예측. 각 방식은 같은 세션(같은 GPU)에서 앞뒤로 잰 SpecLA를 시간에 맞춰 보간한 기준. "
             "oracle = 정답 위치(상한).", fontsize=9.3, color=SEC)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("written", name, flush=True)


tps_figure("GSM8K", "fig3_result_tps.png", "결과 (GSM8K): 원본 SpecLA 대비 처리량")
tps_figure("NQ", "fig8_result_tps_nq.png", "결과 (QA, NQ-open 256문항): 원본 SpecLA 대비 처리량")


def styled_table(ax, cells, cols, bold=(), shade_rows=(), italic=()):
    tb = ax.table(cellText=cells, colLabels=cols, loc="center", cellLoc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(10)
    tb.scale(1, 1.55)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor("#c3c2b7")
        if r == 0:
            cell.set_facecolor("#eef2f7")
            cell.set_text_props(fontweight="bold")
        elif (r, c) in bold:
            cell.set_text_props(fontweight="bold")
        if r in shade_rows:
            cell.set_facecolor("#f6f6f3")
        if r in italic and r != 0:
            cell.set_text_props(color=SEC, style="italic")
    return tb


# ---------- fig9: plain table, rows dataset x batch, SpecLA tok/s and speedups
mnames = [m for m, _ in METH]
cells, bold, shade = [], set(), set()
for ds in ("GSM8K", "NQ"):
    for b in BS:
        if (ds, b, "SpecLA") not in T:
            continue
        row = [ds, f"{b}", f"{T[(ds, b, 'SpecLA')]['tps']:.0f}"] + [f"{T[(ds, b, m)]['x']:.3f}×" for m in mnames]
        r = len(cells) + 1
        best = max(mnames[1:], key=lambda m: T[(ds, b, m)]["x"])
        bold.add((r, 3 + mnames.index(best)))
        if ds == "NQ":
            shade.add(r)
        cells.append(row)
fig, ax = plt.subplots(figsize=(11, 0.33 * len(cells) + 1.2))
ax.axis("off")
styled_table(ax, cells, ["데이터", "batch", "SpecLA tok/s"] + [f"{m}" for m in mnames], bold=bold, shade_rows=shade)
ax.set_title("결과 표: 원본 SpecLA 대비 처리량 배율 (8bit 저장 + 위치 예측)", fontsize=13, color=INK, loc="left", pad=18)
fig.text(0.01, 0.0, "배율 = tok/s ÷ 같은 세션 SpecLA tok/s. 굵은 글씨 = oracle 제외 최고. GSM8K 256문항, NQ-open dev 256문항, 최대 128토큰.",
         fontsize=9.3, color=SEC)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig9_results_table.png"), bbox_inches="tight", facecolor="white")
plt.close(fig)
print("written fig9_results_table.png", flush=True)

# ---------- fig10: paper-style table (EAGLE / DFlash): per dataset Speedup and tau, mean speedup; grouped by batch
rows_m = ["SpecLA (원본)", "oracle (상한)", "softmax 기반", "최빈값 기반", "학습 head 기반"]
mkey = {"SpecLA (원본)": "SpecLA", "oracle (상한)": "oracle", "softmax 기반": "softmax 기반", "최빈값 기반": "최빈값 기반",
        "학습 head 기반": "학습 head 기반"}
cells, bold, italic, shade = [], set(), set(), set()
for b in BS:
    block_start = len(cells) + 1
    for m in rows_m:
        k = mkey[m]
        g, n = T[("GSM8K", b, k)], T[("NQ", b, k)]
        mean = (g["x"] + n["x"]) / 2
        cells.append([f"{b}" if m == rows_m[0] else "", m, f"{g['x']:.3f}×", f"{g['tau']:.2f}", f"{n['x']:.3f}×",
                      f"{n['tau']:.2f}", f"{mean:.3f}×"])
        r = len(cells)
        if m == "oracle (상한)":
            italic.add(r)
        if b in (8, 64):
            shade.add(r)
    ours = [block_start + rows_m.index(m) for m in rows_m[2:]]
    for c in (2, 4, 6):
        best = max(ours, key=lambda r: float(cells[r - 1][c].rstrip("×")))
        bold.add((best, c))
fig, ax = plt.subplots(figsize=(11, 0.3 * len(cells) + 1.2))
ax.axis("off")
styled_table(ax, cells, ["Batch", "Method", "GSM8K\nSpeedup", "GSM8K\nτ", "QA (NQ)\nSpeedup", "QA (NQ)\nτ", "Mean\nSpeedup"],
             bold=bold, shade_rows=shade, italic=italic)
ax.set_title("Qwen3.5-4B + DFlash (block 16), RTX 3090, greedy: GDN rollback 방식별 결과", fontsize=13, color=INK, loc="left", pad=18)
fig.text(0.01, 0.0, "EAGLE·DFlash 논문 표 형식. Speedup = 같은 세션 원본 SpecLA 대비 tok/s 배율 (논문의 AR 대비와 다름). "
         "τ = 라운드당 평균 수락 토큰. 굵은 글씨 = oracle 제외 최고. 우리 방식은 모두 8bit 저장.", fontsize=9.3, color=SEC)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig10_results_table_paper.png"), bbox_inches="tight", facecolor="white")
plt.close(fig)
print("written fig10_results_table_paper.png", flush=True)
