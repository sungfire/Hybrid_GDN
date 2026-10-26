"""Method overview (youngjoon, 10/25): DFlash round with GDN rollback (a), SpecLA (original) vs our common structure
(b), and the three position predictors (c)-(e): each gives P(A), the row with the smallest expected rollback cost is
kept (one 8-bit slot, or none), and the restore is finished at the start of the next verify. Writes
OUT_DIR/overview_arch.png. Usage: python q4/plot_overview.py OUT_DIR"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

OUT = sys.argv[1]
names = [f.name for f in font_manager.fontManager.ttflist]
for cand in ("Noto Sans CJK KR", "Noto Sans CJK JP", "Noto Sans CJK SC"):
    if cand in names:
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["figure.dpi"] = 220
plt.rcParams["mathtext.fontset"] = "dejavusans"
INK, SEC, MUT = "#0b0b0b", "#52514e", "#898781"
MOD_FC, MOD_EC = "#dbe8f9", "#2a78d6"          # model modules
NEW_FC, NEW_EC = "#d4f1e6", "#1baf7a"          # modules our ideas add
ST_FC, ST_EC = "#f6b48f", "#d0612b"            # stored state
EMPTY_FC, EMPTY_EC = "#ffffff", "#b4b2a9"
ANCHOR, MASK, DTOK = "#f5b400", "#8fdc9e", "#9cc3ef"
PANEL = "#1e3a5f"

fig, ax = plt.subplots(figsize=(12, 7.0))
ax.set_xlim(0, 12)
ax.set_ylim(-0.8, 6.7)
ax.axis("off")


def panel(x, y, w, h, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.18", fc="none", ec=PANEL,
                                lw=1.3, ls=(0, (5, 3))))
    ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="center", fontsize=12.5, color=INK, fontweight="bold")


def box(x, y, w, h, text, fc=MOD_FC, ec=MOD_EC, fs=10, bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.07", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.25,
            fontweight="bold" if bold else "normal")


def arrow(x0, y0, x1, y1, color="#3d4a5c"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4,
                                                                  shrinkA=0, shrinkB=0))


def line(xs, ys, color="#3d4a5c"):
    ax.plot(xs, ys, color=color, lw=1.4, solid_capstyle="round")


def squares(x, y, n, s, gap, fills, edges=None):
    for i in range(n):
        ax.add_patch(Rectangle((x + i * (s + gap), y), s, s, fc=fills[i], ec=(edges[i] if edges else "#5f5e5a"), lw=0.9))
    return x + n * (s + gap) - gap


def state_row(x, y, stored, s=0.165, gap=0.05, label=True):
    fills = [ST_FC if i + 1 in stored else EMPTY_FC for i in range(16)]
    edges = [ST_EC if i + 1 in stored else EMPTY_EC for i in range(16)]
    end = squares(x, y, 16, s, gap, fills, edges)
    if label:
        ax.text(x + s / 2, y - 0.13, "$S_1$", ha="center", va="center", fontsize=8.5, color=SEC)
        ax.text(end - s / 2, y - 0.13, "$S_{16}$", ha="center", va="center", fontsize=8.5, color=SEC)
    return end


# (a) DFlash round
panel(0.12, 3.5, 7.5, 3.1, "(a) DFlash 한 라운드와 GDN rollback")
y1 = 5.55
end = squares(0.35, y1, 1, 0.26, 0.05, [ANCHOR])
end = squares(end + 0.05, y1, 4, 0.26, 0.05, [MASK] * 4)
ax.text(end + 0.17, y1 + 0.13, "⋯", ha="center", va="center", fontsize=12, color=SEC)
ax.text(1.15, y1 - 0.2, "anchor + mask 15개", ha="center", va="center", fontsize=8.5, color=SEC)
arrow(2.12, y1 + 0.13, 2.4, y1 + 0.13)
box(2.4, y1 - 0.17, 1.35, 0.6, "DFlash\nDrafter")
arrow(3.75, y1 + 0.13, 4.02, y1 + 0.13)
box(4.02, y1 - 0.07, 0.55, 0.4, "$h_k$", fc="#ececec", ec="#888780", fs=11)
arrow(4.57, y1 + 0.13, 4.84, y1 + 0.13)
box(4.84, y1 - 0.17, 0.95, 0.6, "LM Head\n$W_{lm}$")
arrow(5.79, y1 + 0.13, 6.06, y1 + 0.13)
end = squares(6.06, y1, 4, 0.26, 0.05, [DTOK] * 4)
ax.text(end + 0.17, y1 + 0.13, "⋯", ha="center", va="center", fontsize=12, color=SEC)
ax.text(6.75, y1 - 0.2, "draft $d_1 \\cdots d_{15}$", ha="center", va="center", fontsize=8.5, color=SEC)
# down to verify
line([6.75, 6.75, 1.15, 1.15], [y1 - 0.33, 4.95, 4.95, 4.8])
arrow(1.15, 4.8, 1.15, 4.72)
y2 = 4.05
box(0.35, y2 - 0.02, 1.6, 0.7, "Target Verify\n(GDN 24층)")
arrow(1.95, y2 + 0.33, 2.25, y2 + 0.33)
squares(2.25, y2 + 0.24, 1, 0.19, 0, ["#5f5e5a"])
ax.text(2.345, y2 + 0.06, "$S_0$", ha="center", va="center", fontsize=8.5, color=SEC)
xs = 2.58
sz, gp = 0.17, 0.045
end = squares(xs, y2 + 0.24, 16, sz, gp, [EMPTY_FC] * 16, [EMPTY_EC] * 16)
ax.text((xs + end) / 2, y2 + 0.06, "검증 중 위치별 State $S_1 \\cdots S_{16}$", ha="center", va="center", fontsize=8.5, color=SEC)
a = 6
bx1 = xs + a * (sz + gp) - gp
line([xs, xs, bx1, bx1], [y2 + 0.5, y2 + 0.58, y2 + 0.58, y2 + 0.5], color=SEC)
ax.text((xs + bx1) / 2, y2 + 0.7, "수락 a개", ha="center", va="center", fontsize=9, color=INK)
arrow(end + 0.05, y2 + 0.33, end + 0.3, y2 + 0.33)
box(end + 0.3, y2 - 0.02, 1.0, 0.7, "Rollback\n$S_a$로 복원", fc="#fde3d8", ec="#eb6834")
ax.text(3.9, 3.72, "거절된 토큰까지 State에 반영되므로, 수락 위치의 State $S_a$로 되돌려야 함", ha="center", va="center",
        fontsize=9, color=SEC)

# (b) SpecLA (original) vs our common structure
panel(7.75, 3.5, 4.13, 3.1, "(b) SpecLA (원본) vs 우리 공통 구조")
ax.text(7.95, 5.85, "SpecLA (원본)", ha="left", va="center", fontsize=10.5, color=INK, fontweight="bold")
state_row(7.95, 5.42, set(), s=0.15, gap=0.04)
ax.text(11.68, 5.5, "저장 없음", ha="right", va="center", fontsize=9.5, color=INK)
ax.text(7.95, 5.05, "수락 직후 따로 복원: $S_0$에서 a개 재계산, fp32 State 다시 씀", ha="left", va="center", fontsize=8.4, color=SEC)
ax.text(7.95, 4.65, "우리 (세 아이디어 공통)", ha="left", va="center", fontsize=10.5, color=INK, fontweight="bold")
state_row(7.95, 4.22, {4}, s=0.15, gap=0.04)
ax.text(11.68, 4.3, "8bit 1칸", ha="right", va="center", fontsize=9.5, color=INK)
ax.text(7.95, 3.74, "고른 위치 r만 8bit 저장, 다음 검증 시작에서\n저장본부터 남은 a−r개만 재계산 (따로 복원 없음)",
        ha="left", va="center", fontsize=8.4, color=SEC, linespacing=1.25)


# (c)-(e) ideas
# illustration with the measured batch-8 kernel times (us per GDN layer, rounded)
REP = [43.9, 44.5, 45.1, 48, 51, 54.1, 57, 60, 63, 66, 69, 72, 75.5, 78.5, 81.5, 84.6]   # replay A tokens from S_0
FUS = [29.0, 29.5, 31.5, 33.5, 35.7, 39, 42, 45.5, 49, 52, 54.6, 57.5, 60.5, 63.5, 66.5, 69.5]  # 8-bit snapshot + m tokens
WRT = 13.7                                                                              # 8-bit snapshot write in verify


def choose(P):
    """row r (0-based) with the smallest expected cost, or -1 (no snapshot)."""
    best, arg = sum(p * REP[A - 1] for A, p in zip(range(1, 17), P)), -1
    for r in range(16):
        c = WRT + sum(p * (REP[A - 1] if A <= r else FUS[A - 1 - r]) for A, p in zip(range(1, 17), P))
        if c < best:
            best, arg = c, r
    return arg


def pa_chart(x, y, w, h, P):
    bw = w / 16
    m = max(P)
    for i, p in enumerate(P):
        ax.add_patch(Rectangle((x + i * bw + bw * 0.12, y), bw * 0.76, h * p / m, fc="#cfcdc5", ec="none"))
    ax.plot([x, x + w], [y, y], color=MUT, lw=0.8)
    ax.text(x, y - 0.12, "A = 1", ha="left", va="center", fontsize=8, color=MUT)
    ax.text(x + w, y - 0.12, "16", ha="right", va="center", fontsize=8, color=MUT)


def pa_from_c(c):
    P, surv = [], 1.0
    for j in range(15):
        P.append(surv * (1 - c[j]))
        surv *= c[j]
    return P + [surv]


W3, X3 = 3.84, [0.12, 4.08, 8.04]
c_ex = [0.97, 0.93, 0.6, 0.85, 0.5, 0.9, 0.7, 0.8, 0.85, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
for i, title in enumerate(["(c) Static: 최빈값 기반", "(d) Dynamic: softmax 기반", "(e) Dynamic: 학습 head 기반"]):
    panel(X3[i], 0.08, W3, 3.3, title)


# (c)
x = X3[1] + 0.2
box(x, 2.38, 0.5, 0.4, "$h_k$", fc="#ececec", ec="#888780", fs=11)
arrow(x + 0.5, 2.58, x + 0.72, 2.58)
box(x + 0.72, 2.28, 0.95, 0.6, "LM Head\n$W_{lm}$")
arrow(x + 1.67, 2.58, x + 1.89, 2.58)
box(x + 1.89, 2.28, 1.05, 0.6, "softmax\n1등 확률", fc=NEW_FC, ec=NEW_EC)
ax.text(x + 2.415, 2.17, "argmax와 한 번에 계산", ha="center", va="center", fontsize=7.6, color=SEC)
arrow(x + 2.94, 2.58, x + 3.16, 2.58)
ax.text(x + 3.3, 2.58, "$c_k$", ha="center", va="center", fontsize=12, color=INK)
P = pa_from_c(c_ex)
pa_chart(x + 0.1, 1.25, 3.2, 0.55, P)
ax.text(x + 1.7, 1.95, "$P(A)=c_1\\cdots c_{A-1}(1-c_A)$", ha="center", va="center", fontsize=8.8, color=SEC)
r = choose(P)
state_row(x + 0.1, 0.55, {r + 1} if r >= 0 else set())
ax.text(x + 1.7, 0.22, "기대 비용이 가장 작은 위치 1칸만 저장", ha="center", va="center", fontsize=8.8, color=SEC)

# (d)
x = X3[2] + 0.2
box(x, 2.6, 0.62, 0.32, "$h_k$", fc="#ececec", ec="#888780", fs=10)
box(x, 2.2, 0.62, 0.32, "$E(t_{k-1})$", fc="#fff1c7", ec="#d8a200", fs=9)
line([x + 0.62, x + 0.74, x + 0.74], [2.76, 2.76, 2.36])
line([x + 0.62, x + 0.74], [2.36, 2.36])
arrow(x + 0.74, 2.56, x + 0.92, 2.56)
box(x + 0.92, 2.26, 1.15, 0.6, "MLP head\n5120-256-1", fc=NEW_FC, ec=NEW_EC, fs=9.5)
arrow(x + 2.07, 2.56, x + 2.29, 2.56)
box(x + 2.29, 2.36, 0.5, 0.4, "σ", fc=NEW_FC, ec=NEW_EC, fs=11)
arrow(x + 2.79, 2.56, x + 3.01, 2.56)
ax.text(x + 3.17, 2.56, "$c_k$", ha="center", va="center", fontsize=12, color=INK)
c_ex2 = [0.96, 0.6, 0.92, 0.88, 0.85, 0.55, 0.8, 0.85, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
P = pa_from_c(c_ex2)
pa_chart(x + 0.1, 1.25, 3.2, 0.55, P)
ax.text(x + 1.7, 1.95, "같은 식으로 $P(A)$ (head는 BCE로 학습)", ha="center", va="center", fontsize=8.8, color=SEC)
r = choose(P)
state_row(x + 0.1, 0.55, {r + 1} if r >= 0 else set())
ax.text(x + 1.7, 0.22, "기대 비용이 가장 작은 위치 1칸만 저장", ha="center", va="center", fontsize=8.8, color=SEC)

# (e)
x = X3[0] + 0.2
box(x, 2.3, 1.5, 0.55, "지난 라운드들의\n수락 길이", fc="#ececec", ec="#888780", fs=9.5)
arrow(x + 1.5, 2.575, x + 1.75, 2.575)
box(x + 1.75, 2.3, 1.45, 0.55, "횟수 세기\n$n(A)$ += 1", fc=NEW_FC, ec=NEW_EC, fs=9.5)
counts = [11, 13, 11, 9, 8, 7, 6, 5, 4, 4, 3, 3, 2, 2, 2, 9]
P = [cnt / sum(counts) for cnt in counts]
pa_chart(x + 0.1, 1.25, 3.2, 0.55, P)
ax.text(x + 1.7, 1.95, "$P(A) = n(A) / \\sum n$ (모든 시퀀스 공통)", ha="center", va="center", fontsize=8.8, color=SEC)
r = choose(P)
state_row(x + 0.1, 0.55, {r + 1} if r >= 0 else set())
ax.text(x + 1.7, 0.22, "기대 비용이 가장 작은 위치 1칸만 저장", ha="center", va="center", fontsize=8.8, color=SEC)

ax.add_patch(Rectangle((0.3, -0.2), 0.17, 0.17, fc=ST_FC, ec=ST_EC, lw=0.9))
ax.text(0.55, -0.115, "저장한 State (8bit)", ha="left", va="center", fontsize=9.5, color=SEC)
ax.add_patch(Rectangle((2.15, -0.2), 0.17, 0.17, fc=EMPTY_FC, ec=EMPTY_EC, lw=0.9))
ax.text(2.4, -0.115, "저장 안 함", ha="left", va="center", fontsize=9.5, color=SEC)
ax.add_patch(FancyBboxPatch((3.45, -0.2), 0.3, 0.17, boxstyle="round,pad=0,rounding_size=0.03", fc=NEW_FC, ec=NEW_EC, lw=0.9))
ax.text(3.85, -0.115, "아이디어가 추가하는 부분", ha="left", va="center", fontsize=9.5, color=SEC)
ax.text(0.3, -0.4, "세 아이디어 공통: 기대 비용 = 8bit 저장 비용 + 기대 재계산 비용. 재계산이 더 싸면 저장하지 않고 SpecLA와 같게 동작",
        ha="left", va="center", fontsize=9.5, color=SEC)
ax.text(0.3, -0.68, "Static: 지난 라운드 통계로 정한 같은 위치를 모든 시퀀스가 씀 (예측 비용 0).   "
        "Dynamic: 라운드마다 지금 draft를 보고 시퀀스별로 위치를 정함", ha="left", va="center", fontsize=9.5, color=SEC)
fig.savefig(f"{OUT}/overview_arch.png", bbox_inches="tight", facecolor="white")
print("written", OUT)
