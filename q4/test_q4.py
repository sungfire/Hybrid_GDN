"""Checks for the 8-bit / 4-bit snapshot kernels (youngjoon, 10/24-25).

1. gdn_verify_slots: outputs bit-identical to gdn_verify; fp32 slots == the per-token states; 4-bit / 8-bit slots ==
   torch quantisation of the fp32 state.
2. gdn_replay_from (fp32 / 8-bit / 4-bit source): bit-identical to copy (or torch dequant) + gdn_replay; rows with
   src < 0 behave as gdn_replay.
3. gdn_verify_deferred: the previous block's restore done in the verify prologue bit-identical to restore first, then
   gdn_verify; with the committed state kept 8-bit / 4-bit, against a torch reference. argmax_maxprob == argmax and
   the softmax probability of it.
4. GPU time per kernel call on one GDN layer (x 24 layers printed), measured as CUDA-graph replays (no host launch
   overhead, as in the graph engine): verify without / with one fp32 / 8-bit / 4-bit slot, restores.
Usage: python q4/test_q4.py [--bs 1,8] [--full --json q4/bench_cost_rtx3090.json]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dflash_cb.kernels import (argmax_maxprob, cond_copy_rows, dequant_q4, dequant_q8, gdn_replay,  # noqa: E402
                               gdn_replay_from, gdn_verify, gdn_verify_deferred, gdn_verify_slots)

HV, K, V, S, L = 32, 128, 128, 16, 24
FULL, BENCH = False, {}


def inputs(B, seed=0):
    g_ = torch.Generator(device="cuda").manual_seed(seed)
    r = lambda *s: torch.randn(*s, device="cuda", generator=g_)  # noqa: E731
    q, k, v = r(B, S, HV, K).bfloat16(), r(B, S, HV, K).bfloat16(), r(B, S, HV, V).bfloat16()
    g = -torch.nn.functional.softplus(r(B, S, HV) - 4)                      # mostly close to 0, like trained gates
    beta = torch.sigmoid(r(B, S, HV)).bfloat16()
    h0 = (r(B, HV, K, V) * 0.05).contiguous()
    return q, k, v, g, beta, h0


def quant_ref(x, G):
    """torch version of the kernel's quantiser: x [..., K, V] fp32 -> (packed uint8 [..., K/2, V], scale [..., K/G, V])."""
    x3 = x.unflatten(-2, (x.shape[-2] // G, G))
    s = x3.abs().amax(-2).clamp_min(1e-30) / 7.0
    i = torch.floor(x3 / s.unsqueeze(-2) + 8.5).clamp(0, 15).to(torch.int32).flatten(-3, -2)
    return (i[..., 0::2, :] | (i[..., 1::2, :] << 4)).to(torch.uint8), s


def quant8_ref(x, G):
    x3 = x.unflatten(-2, (x.shape[-2] // G, G))
    s = x3.abs().amax(-2).clamp_min(1e-30) / 127.0
    i = torch.floor(x3 / s.unsqueeze(-2) + 0.5).clamp(-127, 127).to(torch.int32).flatten(-3, -2)
    return ((i[..., 0::2, :] & 255) | (i[..., 1::2, :] << 8)).to(torch.int16), s


def check8(B, G):
    """8-bit path: same checks as check() (verify output, stored bytes vs torch, fused restore vs torch dequant)."""
    q, k, v, g, beta, h0 = inputs(B)
    full = torch.empty(B, S, HV, K, V, device="cuda")
    o_ref = gdn_verify(q, k, v, g, beta, h0, full, store=True)
    keep = [5, 11]
    rows = torch.tensor(keep, device="cuda").expand(B, -1)
    hq = torch.empty(B, len(keep), HV, K // 2, V, dtype=torch.int16, device="cuda")
    hs = torch.empty(B, len(keep), HV, K // G, V, device="cuda")
    o8 = gdn_verify_slots(q, k, v, g, beta, h0, hq, rows, hs)
    ok_o = torch.equal(o8, o_ref)
    o_diff = (o8.float() - o_ref.float()).abs().max().item()
    pq, ps = quant8_ref(full[:, keep], G)
    byte_mis = (pq != hq).float().mean().item()
    deq = dequant_q8(hq, hs)
    rel_rms = ((deq - full[:, keep]).norm() / full[:, keep].norm()).item()
    produced = torch.tensor([[14, 8, 3, 6, 12, 1, 16, 9][r % 8] for r in range(B)], device="cuda")
    best = torch.where(produced >= 12, 11, torch.where(produced >= 6, 5, -1))
    slot = torch.where(produced >= 12, 1, 0)
    src = torch.where(best >= 0, torch.arange(B, device="cuda") * 2 + slot, -1)
    offs = torch.where(best >= 0, best + 1, 0)
    lens = produced - offs
    a = h0.clone()
    for r in range(B):
        if best[r] >= 0:
            a[r] = deq[r, int(slot[r])]
    gdn_replay(k, v, g, beta, a, offs, lens)
    c = h0.clone()
    gdn_replay_from(k, v, g, beta, c, offs, lens, src, hq.flatten(0, 1), hs.flatten(0, 1))
    ok_q8 = torch.equal(a, c)
    print(f"[check8 B={B} G={G}] verify o identical: {ok_o} (max diff {o_diff:.1e}), restore max diff "
          f"{(a - c).abs().max().item():.1e} | int8 values differing: {byte_mis:.2e} | scale max rel "
          f"diff {((ps - hs).abs() / ps).max().item():.1e} | 8-bit snapshot rel RMS err {rel_rms:.4f} | fused restore "
          f"8-bit identical: {ok_q8}", flush=True)
    return ok_o and ok_q8 and byte_mis < 1e-4


def check_deferred(B, bits):
    """Two steps, deferred restore vs restore-then-verify: step-2 outputs and the committed state bit-identical."""
    q1, k1, v1, g1, b1, h0 = inputs(B, seed=1)
    q2, k2, v2, g2, b2, _ = inputs(B, seed=2)
    dev = "cuda"
    dt = {8: torch.int16, 4: torch.uint8}.get(bits)
    mk = lambda: (torch.empty(B, 1, HV, K // 2, V, dtype=dt, device=dev),                             # noqa: E731
                  torch.empty(B, 1, HV, 1, V, device=dev)) if bits else (torch.empty(B, 1, HV, K, V, device=dev), None)
    rows1 = torch.full((B, 1), 5, dtype=torch.long, device=dev)
    rows2 = torch.full((B, 1), 9, dtype=torch.long, device=dev)
    produced = torch.tensor([[14, 3, 6, 1, 16, 5, 9, 2][r % 8] for r in range(B)], device=dev)
    have = produced >= 6
    src = torch.where(have, torch.arange(B, device=dev), -1)
    offs = torch.where(have, 6, 0)
    lens = produced - offs
    # reference: verify, fused restore, verify
    sa, ca = mk()
    o1_ref = gdn_verify_slots(q1, k1, v1, g1, b1, h0, sa, rows1, ca)
    h1 = h0.clone()
    gdn_replay_from(k1, v1, g1, b1, h1, offs, lens, src, sa.flatten(0, 1), ca.flatten(0, 1) if ca is not None else None)
    sb, cb = mk()
    o2_ref = gdn_verify_slots(q2, k2, v2, g2, b2, h1, sb, rows2, cb)
    # deferred, with the buffers as layer 1 of a two-layer buffer (row stride != row size, as in the rollback)
    sd = torch.zeros(B, 2, *sb.shape[1:], dtype=sb.dtype, device=dev)[:, 1]
    cd = torch.zeros(B, 2, *cb.shape[1:], device=dev)[:, 1] if cb is not None else None
    h = h0.clone()
    pk = torch.zeros(B, 2, 2, S, HV, K, dtype=torch.bfloat16, device=dev)[:, 1]
    pv = torch.zeros(B, 2, 2, S, HV, V, dtype=torch.bfloat16, device=dev)[:, 1]
    pg = torch.zeros(B, 2, 2, S, HV, device=dev)[:, 1]
    pb = torch.zeros(B, 2, 2, S, HV, dtype=torch.bfloat16, device=dev)[:, 1]
    par = torch.zeros((), dtype=torch.int32, device=dev)
    i32 = lambda x: x.to(torch.int32).contiguous()  # noqa: E731
    o1 = gdn_verify_deferred(q1, k1, v1, g1, b1, h, sd, i32(rows1[:, 0]), pk, pv, pg, pb, par,
                             i32(torch.full((B,), 3, device=dev)), i32(torch.zeros(B, device=dev)),
                             i32(torch.full((B,), 7, device=dev)), i32(torch.ones(B, device=dev)), cd)  # fresh rows
    ok1 = torch.equal(o1, o1_ref) and torch.equal(h, h0)
    par.fill_(1)
    o2 = gdn_verify_deferred(q2, k2, v2, g2, b2, h, sd, i32(rows2[:, 0]), pk, pv, pg, pb, par, i32(torch.where(have, 0, -1)),
                             i32(offs), i32(lens), i32(torch.zeros(B, device=dev)), cd)
    ok2 = torch.equal(o2, o2_ref) and torch.equal(h, h1) and torch.equal(sd, sb)
    print(f"[deferred B={B} bits={bits}] step 1 identical: {ok1} | step 2 output, committed state, snapshot identical: "
          f"{ok2} (max |o2 diff| {(o2.float() - o2_ref.float()).abs().max().item():.1e}, max |h diff| "
          f"{(h - h1).abs().max().item():.1e})", flush=True)
    return ok1 and ok2


def check_deferred_cq(B, bits):
    """Committed state kept quantised (same bits as the snapshot): two steps vs a torch reference (rows without a
    snapshot start from dequant(quant(committed))). Rounding-boundary bytes may differ (GPU division), so: share of
    differing bytes and the output difference."""
    q1, k1, v1, g1, b1, h0 = inputs(B, seed=3)
    q2, k2, v2, g2, b2, _ = inputs(B, seed=4)
    dev = "cuda"
    dt = {8: torch.int16, 4: torch.uint8}[bits]
    qf, dqf = (quant8_ref, dequant_q8) if bits == 8 else (quant_ref, dequant_q4)
    mk = lambda: (torch.empty(B, 1, HV, K // 2, V, dtype=dt, device=dev), torch.empty(B, 1, HV, 1, V, device=dev))  # noqa: E731
    rows1 = torch.full((B, 1), 5, dtype=torch.long, device=dev)
    rows2 = torch.full((B, 1), 9, dtype=torch.long, device=dev)
    produced = torch.tensor([[14, 3, 6, 1, 16, 5, 9, 2][r % 8] for r in range(B)], device=dev)
    have = produced >= 6
    src = torch.where(have, torch.arange(B, device=dev), -1)
    offs = torch.where(have, 6, 0)
    lens = produced - offs
    sa, ca = mk()
    gdn_verify_slots(q1, k1, v1, g1, b1, h0, sa, rows1, ca)
    h1c = dqf(*qf(h0, K)).contiguous()
    gdn_replay_from(k1, v1, g1, b1, h1c, offs, lens, src, sa.flatten(0, 1), ca.flatten(0, 1))
    sb, cb = mk()
    o2r = gdn_verify_slots(q2, k2, v2, g2, b2, h1c, sb, rows2, cb)
    sd, cd = mk()
    cq = torch.zeros(B, HV, K // 2, V, dtype=dt, device=dev)
    cqs = torch.zeros(B, HV, 1, V, device=dev)
    h = h0.clone()
    pk = torch.zeros(B, 2, S, HV, K, dtype=torch.bfloat16, device=dev)
    pv = torch.zeros(B, 2, S, HV, V, dtype=torch.bfloat16, device=dev)
    pg = torch.zeros(B, 2, S, HV, device=dev)
    pb = torch.zeros(B, 2, S, HV, dtype=torch.bfloat16, device=dev)
    par = torch.zeros((), dtype=torch.int32, device=dev)
    i32 = lambda x: x.to(torch.int32).contiguous()  # noqa: E731
    gdn_verify_deferred(q1, k1, v1, g1, b1, h, sd, i32(rows1[:, 0]), pk, pv, pg, pb, par, i32(torch.full((B,), 3, device=dev)),
                        i32(torch.zeros(B, device=dev)), i32(torch.full((B,), 7, device=dev)), i32(torch.ones(B, device=dev)),
                        cd, cq, cqs)
    mis1 = (cq != qf(h0, K)[0]).float().mean().item()
    par.fill_(1)
    o2 = gdn_verify_deferred(q2, k2, v2, g2, b2, h, sd, i32(rows2[:, 0]), pk, pv, pg, pb, par, i32(torch.where(have, 0, -1)),
                             i32(offs), i32(lens), i32(torch.zeros(B, device=dev)), cd, cq, cqs)
    mis2 = (cq != qf(h1c, K)[0]).float().mean().item()
    od = (o2.float() - o2r.float()).abs().max().item()
    ok = mis1 < 1e-4 and mis2 < 1e-4 and od < 5e-2 and torch.equal(h, h0)
    print(f"[deferred-cq B={B} bits={bits}] committed bytes differing step1 {mis1:.1e} step2 {mis2:.1e} | step-2 output "
          f"max diff {od:.1e} | h untouched {torch.equal(h, h0)} -> {'ok' if ok else 'FAIL'}", flush=True)
    return ok


def check_argmax(R=15 * 32, Vv=248320):
    """argmax_maxprob == (argmax, max softmax probability) on bf16 logits, ties included (first index wins)."""
    g_ = torch.Generator(device="cuda").manual_seed(1)
    x = (torch.randn(R, Vv, device="cuda", generator=g_) * 3).bfloat16()
    x[0, 1000] = x[0, 200000] = 40.0                      # tie across splits
    x[1, 5] = x[1, 6] = 40.0                              # tie inside a block
    tok, p = argmax_maxprob(x)
    ref_t = x.argmax(-1)
    lg = x.float()
    ref_p = (lg.amax(-1) - lg.logsumexp(-1)).exp()
    ok_t = torch.equal(tok, ref_t)
    err = ((p - ref_p).abs() / ref_p).max().item()
    print(f"[argmax] tokens identical: {ok_t} | max rel err of the probability {err:.1e}", flush=True)
    return ok_t and err < 1e-4


def check(B, G):
    q, k, v, g, beta, h0 = inputs(B)
    full = torch.empty(B, S, HV, K, V, device="cuda")
    o_ref = gdn_verify(q, k, v, g, beta, h0, full, store=True)                # state after every token
    keep = [5, 11]                                                          # rows (AL 6 and 12)
    rows = torch.tensor(keep, device="cuda").expand(B, -1)
    hq = torch.empty(B, len(keep), HV, K // 2, V, dtype=torch.uint8, device="cuda")
    hs = torch.empty(B, len(keep), HV, K // G, V, device="cuda")
    o_q = gdn_verify_slots(q, k, v, g, beta, h0, hq, rows, hs)
    h32 = torch.empty(B, len(keep), HV, K, V, device="cuda")
    o_32 = gdn_verify_slots(q, k, v, g, beta, h0, h32, rows)
    ok_o = torch.equal(o_q, o_ref) and torch.equal(o_32, o_ref) and torch.equal(h32, full[:, keep])
    pq, ps = quant_ref(full[:, keep], G)
    byte_mis = (pq != hq).float().mean().item()
    sc_rel = ((ps - hs).abs() / ps).max().item()
    deq = dequant_q4(hq, hs)
    err = deq - full[:, keep]
    rel_rms = (err.norm() / full[:, keep].norm()).item()
    # fused restore: rows alternate between slot 1 (AL >= 12), slot 0 (6 <= AL < 12) and S_0
    produced = torch.tensor([[14, 8, 3, 6, 12, 1, 16, 9][r % 8] for r in range(B)], device="cuda")
    best = torch.where(produced >= 12, 11, torch.where(produced >= 6, 5, -1))
    slot = torch.where(produced >= 12, 1, 0)
    src = torch.where(best >= 0, torch.arange(B, device="cuda") * 2 + slot, -1)
    offs = torch.where(best >= 0, best + 1, 0)
    lens = produced - offs
    snap32 = full[:, keep].contiguous()
    a = h0.clone()                                                          # reference: copy + gdn_replay
    for r in range(B):
        if best[r] >= 0:
            a[r] = snap32[r, int(slot[r])]
    gdn_replay(k, v, g, beta, a, offs, lens)
    b = h0.clone()
    gdn_replay_from(k, v, g, beta, b, offs, lens, src, snap32.flatten(0, 1))
    ok_f32 = torch.equal(a, b)
    a = h0.clone()
    for r in range(B):
        if best[r] >= 0:
            a[r] = deq[r, int(slot[r])]
    gdn_replay(k, v, g, beta, a, offs, lens)
    c = h0.clone()
    gdn_replay_from(k, v, g, beta, c, offs, lens, src, hq.flatten(0, 1), hs.flatten(0, 1))
    ok_q4 = torch.equal(a, c)
    exact = full[torch.arange(B), produced - 1]                             # exact state after `produced` tokens
    print(f"[check B={B} G={G}] verify o identical: {ok_o} | packed bytes differing: {byte_mis:.2e} | scale max rel "
          f"diff {sc_rel:.1e} | 4-bit snapshot rel RMS err {rel_rms:.4f} | fused restore fp32 identical: {ok_f32}, "
          f"4-bit identical: {ok_q4} | fp32 fused vs exact max abs {(b - exact).abs().max().item():.1e}", flush=True)
    return ok_o and ok_f32 and ok_q4 and byte_mis < 1e-4


def timeit(fn, calls=20, reps=20):
    """us of GPU time per fn() call: fn captured `calls` times in one CUDA graph, the graph replayed `reps` times."""
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(st)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for _ in range(calls):
            fn()
    for _ in range(3):
        gr.replay()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    e0.record()
    for _ in range(reps):
        gr.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / (reps * calls) * 1e3


def bench(B):
    q, k, v, g, beta, h0 = inputs(B)
    s1 = torch.empty(B, 1, HV, K, V, device="cuda")
    q1 = torch.empty(B, 1, HV, K // 2, V, dtype=torch.uint8, device="cuda")
    c1 = torch.empty(B, 1, HV, 1, V, device="cuda")
    q8 = torch.empty(B, 1, HV, K // 2, V, dtype=torch.int16, device="cuda")
    t = {}
    t["verify, no store"] = timeit(lambda: gdn_verify(q, k, v, g, beta, h0, h0, store=False))
    none = torch.full((B, 1), -1, dtype=torch.long, device="cuda")
    t["verify_slots, no row kept"] = timeit(lambda: gdn_verify_slots(q, k, v, g, beta, h0, q1, none, c1))
    for al in (2, 6, 12):
        smap = torch.full((B, S), -1, dtype=torch.int32, device="cuda")
        smap[:, al - 1] = 0
        rows = torch.full((B, 1), al - 1, dtype=torch.long, device="cuda")
        t[f"verify, slot map fp32 (AL {al})"] = timeit(lambda: gdn_verify(q, k, v, g, beta, h0, s1, slot_map=smap))
        t[f"verify_slots fp32 (AL {al})"] = timeit(lambda: gdn_verify_slots(q, k, v, g, beta, h0, s1, rows))
        t[f"verify_slots 4bit (AL {al})"] = timeit(lambda: gdn_verify_slots(q, k, v, g, beta, h0, q1, rows, c1))
        t[f"verify_slots 8bit (AL {al})"] = timeit(lambda: gdn_verify_slots(q, k, v, g, beta, h0, q8, rows, c1))
    h = h0.clone()
    zero = torch.zeros(B, dtype=torch.long, device="cuda")
    src0 = torch.arange(B, device="cuda")
    for n in range(1, S + 1) if FULL else (1, 3, 6, 10, 16):
        lens = torch.full((B,), n, dtype=torch.long, device="cuda")
        t[f"replay {n} tok from S0"] = timeit(lambda: gdn_replay(k, v, g, beta, h, zero, lens))
    t["copy fp32 snap (cond_copy_rows)"] = timeit(lambda: cond_copy_rows(h, s1.flatten(0, 1), src0))
    for n in range(0, S) if FULL else (0, 1, 4, 10):
        lens = torch.full((B,), n, dtype=torch.long, device="cuda")
        t[f"fused fp32 snap + {n} tok"] = timeit(lambda: gdn_replay_from(k, v, g, beta, h, zero, lens, src0,
                                                                         s1.flatten(0, 1)))
        t[f"fused 4bit snap + {n} tok"] = timeit(lambda: gdn_replay_from(k, v, g, beta, h, zero, lens, src0,
                                                                         q1.flatten(0, 1), c1.flatten(0, 1)))
        t[f"fused 8bit snap + {n} tok"] = timeit(lambda: gdn_replay_from(k, v, g, beta, h, zero, lens, src0,
                                                                         q8.flatten(0, 1), c1.flatten(0, 1)))
    BENCH[B] = t
    print(f"[bench B={B}] GPU time per call, one GDN layer | x{L} layers", flush=True)
    for name, us in t.items():
        print(f"  {name:34s} {us:8.1f} us  {us * L / 1e3:7.3f} ms", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", default="1,8")
    ap.add_argument("--bench-only", action="store_true")
    ap.add_argument("--full", action="store_true", help="replay / fused restore for every token count")
    ap.add_argument("--json", default=None, help="write the bench times (us per layer) here")
    a = ap.parse_args()
    FULL = a.full
    if not a.bench_only:
        ok = check_argmax()
        for B in map(int, a.bs.split(",")):
            for bits in (0, 8, 4):
                ok &= check_deferred(B, bits)
            for bits in (8, 4):
                ok &= check_deferred_cq(B, bits)
        for B in map(int, a.bs.split(",")):
            for G in (128, 32):
                ok &= check(B, G)
                ok &= check8(B, G)
        print("ALL CHECKS PASSED" if ok else "CHECK FAILED", flush=True)
    for B in map(int, a.bs.split(",")):
        bench(B)
    if a.json:
        import json
        json.dump({str(b): t for b, t in BENCH.items()}, open(a.json, "w"), indent=1)
