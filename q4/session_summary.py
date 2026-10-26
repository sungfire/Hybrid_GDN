"""Per-step section times of same-session runs (youngjoon, 10/24). Usage: python q4/session_summary.py DIR [names...] [x.json]
Prints, per batch size and method, mean over runs of tok/s, tau and ms per step (draft, verify, restore, step wall),
and the difference to SpecLA of the session: the specla* runs are interpolated linearly in time (cell file mtime) to
the method's own time, so a drift over the session cancels (d_draft, identical work for every method, shows what
drift is left)."""
import json
import os
import sys

d = sys.argv[1]
names = [a for a in sys.argv[2:] if not a.endswith(".json")] or sorted(x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x)))
dump = next((a for a in sys.argv[2:] if a.endswith(".json")), None)
rows = {}
for nm in names:
    for f in sorted(os.listdir(os.path.join(d, nm, "cells"))):
        c = json.load(open(os.path.join(d, nm, "cells", f)))
        if c.get("status") != "done":
            continue
        rs = c["runs"]
        m = lambda fn: sum(fn(r) for r in rs) / len(rs)  # noqa: E731
        sec = lambda k: m(lambda r: r["sections"].get(k, 0) / r["steps"] * 1e3)  # noqa: E731
        rows[(c["batch"], nm)] = dict(time=os.path.getmtime(os.path.join(d, nm, "cells", f)),
                                      tps=m(lambda r: r["throughput"]), tau=m(lambda r: r["tau"]),
                                      draft=sec("draft"), verify=sec("verify"), save=sec("save"), restore=sec("restore"),
                                      step=m(lambda r: r["decode_s"] / r["steps"] * 1e3),
                                      hit=rs[0].get("prediction", {}).get("hit_rate"))
KEYS = ("tps", "draft", "verify", "restore", "step")


def baseline(b, t):
    """SpecLA values at time t: linear interpolation between the session's specla runs (constant outside them)."""
    base = sorted((v["time"], v) for (bb, nm), v in rows.items() if bb == b and nm.startswith("specla"))
    if not base:
        return None
    if t <= base[0][0] or len(base) == 1:
        return {k: base[0][1][k] for k in KEYS}
    for (t0, v0), (t1, v1) in zip(base, base[1:]):
        if t <= t1:
            w = (t - t0) / (t1 - t0)
            return {k: v0[k] + w * (v1[k] - v0[k]) for k in KEYS}
    return {k: base[-1][1][k] for k in KEYS}


for b in sorted({k[0] for k in rows}):
    sp = [v for (bb, nm), v in rows.items() if bb == b and nm.startswith("specla")]
    print(f"\nbatch {b}  (SpecLA: " + ", ".join(f"{v['tps']:.1f}" for v in sorted(sp, key=lambda x: x["time"])) + " tok/s)")
    print(f"{'method':18s} {'tok/s':>7s} {'tau':>6s} {'draft':>7s} {'verify':>8s} {'restore':>7s} {'step':>8s} "
          f"{'d_draft':>7s} {'d_verify':>8s} {'d_rest':>7s} {'d_step':>7s} {'d_tps%':>7s} {'hit':>5s}")
    for (bb, nm), v in sorted(rows.items(), key=lambda x: x[1]["time"]):
        if bb != b:
            continue
        ref = baseline(b, v["time"])
        d = {k: v[k] - ref[k] for k in KEYS} if ref else {k: 0 for k in KEYS}
        v["d"] = d
        v["d_tps_pct"] = 100 * (v["tps"] / ref["tps"] - 1) if ref else 0
        hit = f"{v['hit']:.2f}" if v["hit"] is not None else "-"
        print(f"{nm:18s} {v['tps']:7.1f} {v['tau']:6.3f} {v['draft']:7.3f} {v['verify']:8.3f} {v['restore']:7.3f} "
              f"{v['step']:8.3f} {d['draft']:+7.3f} {d['verify']:+8.3f} {d['restore']:+7.3f} {d['step']:+7.3f} "
              f"{v['d_tps_pct']:+7.2f} {hit:>5s}")
if dump:
    json.dump({f"{nm}_bs{b}": v for (b, nm), v in rows.items()}, open(dump, "w"), indent=1)
