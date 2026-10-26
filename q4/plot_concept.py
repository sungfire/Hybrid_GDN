"""Concept figure (not measured data): quantising the stored state lowers the storing cost, so the acceptance length
from which storing beats recomputation moves earlier (youngjoon, 10/25). Usage: python q4/plot_concept.py OUT_PNG"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

OUT = sys.argv[1]
names = [f.name for f in font_manager.fontManager.ttflist]
for cand in ("Noto Sans CJK KR", "Noto Sans CJK JP", "Noto Sans CJK SC"):
    if cand in names:
        plt.rcParams["font.family"] = cand
        break
plt.rcParams.update({"font.size": 12, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e",
                     "xtick.color": "#898781", "ytick.color": "#898781", "figure.dpi": 200})
INK, SEC = "#0b0b0b", "#52514e"
LINES = [("fp32 (양자화 없음)", 3.8, "#898781", "--"), ("8bit", 2.4, "#2a78d6", "-"), ("4bit", 1.75, "#eb6834", "-")]
rec = lambda a: 1.0 + 0.25 * a  # noqa: E731

fig, ax = plt.subplots(figsize=(9, 5.2))
x = np.linspace(0.5, 16.5, 200)
ax.plot(x, rec(x), color=INK, lw=2.8, label="재계산 (SpecLA): 수락 토큰이 많을수록 비쌈")
crosses = []
for name, val, col, ls in LINES:
    ax.plot(x, np.full_like(x, val), color=col, lw=2.6, ls=ls, label=f"저장 {name}")
    xc = (val - 1.0) / 0.25
    crosses.append((xc, val, col, name))
    ax.plot([xc, xc], [0, val], color=col, lw=1.2, ls=":")
    ax.scatter([xc], [val], s=70, color=col, zorder=5, edgecolor="white", linewidth=1.5)
    ax.text(xc, -0.32, f"교차점\n({name.split(' ')[0]})", ha="center", va="top", fontsize=9.5, color=col)
ax.set_xlim(0.5, 16.5)
ax.set_ylim(0, 5.4)
ax.set_xticks([1, 4, 8, 12, 16])
ax.set_yticks([])
ax.set_xlabel("수락 길이 AL (다시 계산할 토큰 수)", labelpad=34)
ax.set_ylabel("라운드당 rollback 비용")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=10.5, loc="upper left", bbox_to_anchor=(0.0, 1.0))
ax.set_title("아이디어: state 저장본을 양자화하면 저장이 싸져서, 저장이 재계산을 이기는 지점이 앞으로 옴", fontsize=13,
             color=INK, loc="left")
fig.text(0.01, -0.02, "개념도 (실측값 아님). 정답 위치에 1칸 저장한다고 가정할 때의 비용 비교. 교차점보다 오른쪽(수락이 길 때)은 저장이 쌈.",
         fontsize=9.5, color=SEC)
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print("written", OUT)
