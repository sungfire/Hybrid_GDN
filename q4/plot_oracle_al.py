"""Step 1 figure (youngjoon, 10/24): rollback cost per acceptance length, recompute (SpecLA) vs oracle snapshot
(fp32 / 4-bit, fused restore), batch 1 / 8 / 32. Lines: GPU kernel times (CUDA-graph replay, one GDN layer x 24
layers; q4/test_q4.py --full --json). Batch-1 markers: engine steps of the same session grouped by AL (oracle runs:
steps whose oracle was right). cost = snapshot write added to verify + restore.
Usage: python q4/plot_oracle_al.py BENCH_JSON SESSION_DIR OUT_PNG"""
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

BJ, D, OUT = sys.argv[1:4]
SUMMARY = sys.argv[4] if len(sys.argv) > 4 else None       # session summary.json: y = tok/s change vs SpecLA
L = 24
names = [f.name for f in font_manager.fontManager.ttflist]
for cand in ("Noto Sans CJK KR", "Noto Sans CJK JP", "Noto Sans CJK SC"):
    if cand in names:
        plt.rcParams["font.family"] = cand
        break
plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e",
                     "xtick.color": "#898781", "ytick.color": "#898781", "figure.dpi": 200})
INK, SEC, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
BLUE, RED, GREEN = "#2a78d6", "#c94f1f", "#1baf7a"
bench = json.load(open(BJ))


def kernel_curves(t):
    v0 = t["verify, no store"]
    w = lambda kind: sum(t[f"verify_slots {kind} (AL {a})"] for a in (2, 6, 12)) / 3 - v0  # noqa: E731
    als = list(range(1, 17))
    spec = [L * t[f"replay {a} tok from S0"] / 1e3 for a in als]
    snap = {k: L * (w(k) + t[f"fused {k} snap + 0 tok"]) / 1e3 for k in ("fp32", "8bit", "4bit")}
    return als, spec, snap


def engine_bs1():
    def steps(name, hit_only):
        p = os.path.join(D, name, "cells", "dflash_bs1.json")
        if not os.path.exists(p):
            return []
        cell, out = json.load(open(p)), []
        for r in cell["runs"]:
            hits = r.get("prediction", {}).get("oracle_hit_trace")
            for k, line in enumerate(open(os.path.join(D, name, "raw", f"dflash_bs1_run{r['run']}_steps.jsonl"))):
                s = json.loads(line)
                if hit_only and hits is not None and not hits[k][0]:
                    continue
                out.append((s["produced"][0], s["verify"] * 1e3, s["restore"] * 1e3))
        return out
    spec = [x for nm in sorted(os.listdir(D)) if nm.startswith("specla") for x in steps(nm, False)]
    v0 = sum(v for _, v, _ in spec) / len(spec)
    res = {}
    for key, nm in (("spec", None), ("f32", "oracle_fused"), ("q4", "oracle_fused_q4")):
        st = spec if nm is None else steps(nm, True)
        by = defaultdict(list)
        for a, v, r in st:
            by[a].append(v - v0 + r)
        res[key] = {a: sum(x) / len(x) for a, x in by.items() if len(x) >= 20}
    return res


STEP = None
if SUMMARY:
    sm = json.load(open(SUMMARY))
    STEP = {b: sum(v["step"] for k, v in sm.items() if k.startswith("specla") and k.endswith(f"_bs{b}"))
            / sum(1 for k in sm if k.startswith("specla") and k.endswith(f"_bs{b}")) for b in ("1", "8", "32")}


def to_tps(cost, spec, b):
    """tok/s change (%) of a round with this AL: same tokens, round time SpecLA step + (cost - SpecLA rollback)."""
    return 100 * (STEP[b] / (STEP[b] + cost - spec) - 1)


fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
STYLE = {"fp32": ("#898781", "--"), "8bit": ("#2a78d6", "-"), "4bit": ("#eb6834", "-")}
for ax, b in zip(axes, ("1", "8", "32")):
    als, spec, snap = kernel_curves(bench[b])
    if STEP:
        ax.axhline(0, color=INK, lw=2.2, label="SpecLA (원본) = 0")
    else:
        ax.plot(als, spec, color=INK, lw=2.2, label="SpecLA (원본): 수락 토큰 재계산")
    for kind, val in snap.items():
        col, ls = STYLE[kind]
        cross = next((a for a, s_ in zip(als, spec) if s_ > val), None)
        gain = f"AL ≥ {cross}부터 이득" if cross is not None else "이득 없음"
        if STEP:
            ys = [to_tps(val, sp, b) for sp in spec]
            ax.plot(als, ys, color=col, lw=2, ls=ls, label=f"oracle {kind} 저장 ({gain})")
        else:
            ax.plot(als, [val] * 16, color=col, lw=2, ls=ls, label=f"oracle {kind} 저장: {val:.2f} ms ({gain})")
        if cross is not None:
            ax.axvline(cross, color=col, lw=1, ls=":")
    ax.set_title(f"batch {b}" + (f" (SpecLA step {STEP[b]:.1f} ms)" if STEP else ""), fontsize=12, color=INK, loc="left")
    ax.set_xlabel("수락 길이 AL")
    ax.set_xticks([1, 4, 8, 12, 16])
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.6, loc="upper left")
axes[0].set_ylabel("원본 SpecLA 대비 처리량 (%)" if STEP else "라운드당 rollback 비용 (ms)")
fig.suptitle(("AL별 처리량: oracle(정답 위치) 저장 vs SpecLA(원본) 재계산" if STEP else
              "oracle(정답 위치) 저장이 SpecLA(원본) 재계산보다 싸지는 AL")
             + " (RTX 3090, GDN 24층 커널 시간, 검증 중 저장 + 복원)", x=0.01, ha="left", fontsize=12.5, color=INK)
fig.text(0.01, -0.04, ("처리량 = SpecLA step ÷ (SpecLA step + oracle 비용 − SpecLA 재계산 비용) − 1. 비용은 커널만 잰 값(CUDA graph), "
                       "SpecLA step은 같은 엔진 실측. " if STEP else "커널만 잰 값 (CUDA graph). ")
         + "oracle = 정답 위치 1칸 저장 후 그 저장본에서 복원.", fontsize=9.5, color=SEC)
fig.tight_layout(w_pad=2.5)
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print("written", OUT)
