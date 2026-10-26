"""Logic check of gdn_verify_deferred in the Triton interpreter (CPU, tiny shapes): TRITON_INTERPRET=1."""
import os
import sys

import torch
import triton.runtime.interpreter as _I

_orig_patch = _I._patch_lang_tensor


def _patch(tensor, scope):
    _orig_patch(tensor, scope)
    scope.set_attr(tensor, "__index__", lambda self: int(self.handle.data.reshape(-1)[0]))


_I._patch_lang_tensor = _patch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dflash_cb.kernels import gdn_replay_from, gdn_verify_deferred, gdn_verify_slots  # noqa: E402

torch.manual_seed(0)
B, S, HV, K, V = 3, 6, 2, 16, 16


def inp():
    return (torch.randn(B, S, HV, K).bfloat16(), torch.randn(B, S, HV, K).bfloat16(), torch.randn(B, S, HV, V).bfloat16(),
            -torch.rand(B, S, HV) * 0.1, torch.rand(B, S, HV).bfloat16())


for bits in (0, 8, 4):
    (q1, k1, v1, g1, b1), (q2, k2, v2, g2, b2) = inp(), inp()
    h0 = torch.randn(B, HV, K, V) * 0.1
    mk = lambda: (torch.zeros(B, 1, HV, K // 2, V, dtype={8: torch.int16, 4: torch.uint8}[bits]),  # noqa: E731
                  torch.zeros(B, 1, HV, 1, V)) if bits else (torch.zeros(B, 1, HV, K, V), None)
    rows1, rows2 = torch.full((B, 1), 2), torch.full((B, 1), 4)
    produced = torch.tensor([5, 1, 3])
    have = produced >= 3
    src = torch.where(have, torch.arange(B), -1)
    offs = torch.where(have, 3, 0)
    lens = produced - offs
    sa, ca = mk()
    o1r = gdn_verify_slots(q1, k1, v1, g1, b1, h0, sa, rows1, ca)
    h1 = h0.clone()
    gdn_replay_from(k1, v1, g1, b1, h1, offs, lens, src, sa.flatten(0, 1), ca.flatten(0, 1) if ca is not None else None)
    sb, cb = mk()
    o2r = gdn_verify_slots(q2, k2, v2, g2, b2, h1, sb, rows2, cb)
    sd, cd = mk()
    h = h0.clone()
    pk, pv = torch.zeros(B, 2, S, HV, K).bfloat16(), torch.zeros(B, 2, S, HV, V).bfloat16()
    pg, pb = torch.zeros(B, 2, S, HV), torch.zeros(B, 2, S, HV).bfloat16()
    par = torch.zeros((), dtype=torch.int32)
    i32 = lambda x: x.to(torch.int32).contiguous()  # noqa: E731
    o1 = gdn_verify_deferred(q1, k1, v1, g1, b1, h, sd, i32(rows1[:, 0]), pk, pv, pg, pb, par, i32(torch.full((B,), 0)),
                             i32(torch.zeros(B)), i32(torch.full((B,), 2)), i32(torch.ones(B)), cd)
    ok1 = torch.allclose(o1.float(), o1r.float(), atol=1e-2) and torch.equal(h, h0)
    par.fill_(1)
    o2 = gdn_verify_deferred(q2, k2, v2, g2, b2, h, sd, i32(rows2[:, 0]), pk, pv, pg, pb, par, i32(torch.where(have, 0, -1)),
                             i32(offs), i32(lens), i32(torch.zeros(B)), cd)
    print(f"bits={bits}: step1 ok {ok1} | step2 o max diff {(o2.float() - o2r.float()).abs().max().item():.1e}, "
          f"h max diff {(h - h1).abs().max().item():.1e}, snapshot equal {torch.equal(sd, sb)}, "
          f"saved k equal {torch.equal(pk[:, 1], k2)}, saved v equal {torch.equal(pv[:, 1], v2)}")

# committed state kept quantised (cq): step 1 fresh rows write quant(h0); step 2 rows without a snapshot start from it
from dflash_cb.kernels import dequant_q4, dequant_q8  # noqa: E402
from test_q4 import quant8_ref, quant_ref  # noqa: E402

for bits in (8, 4):
    (q1, k1, v1, g1, b1), (q2, k2, v2, g2, b2) = inp(), inp()
    h0 = torch.randn(B, HV, K, V) * 0.1
    dt = {8: torch.int16, 4: torch.uint8}[bits]
    qf, dqf = (quant8_ref, dequant_q8) if bits == 8 else (quant_ref, dequant_q4)
    mk = lambda: (torch.zeros(B, 1, HV, K // 2, V, dtype=dt), torch.zeros(B, 1, HV, 1, V))  # noqa: E731
    rows1, rows2 = torch.full((B, 1), 2), torch.full((B, 1), 4)
    produced = torch.tensor([5, 1, 3])
    have = produced >= 3
    src = torch.where(have, torch.arange(B), -1)
    offs = torch.where(have, 3, 0)
    lens = produced - offs
    sa, ca = mk()
    gdn_verify_slots(q1, k1, v1, g1, b1, h0, sa, rows1, ca)
    h1c = dqf(*qf(h0, K)).contiguous()                       # rows without a snapshot start from the quantised state
    gdn_replay_from(k1, v1, g1, b1, h1c, offs, lens, src, sa.flatten(0, 1), ca.flatten(0, 1))
    sb, cb = mk()
    o2r = gdn_verify_slots(q2, k2, v2, g2, b2, h1c, sb, rows2, cb)
    sd, cd = mk()
    cq, cqs = torch.zeros(B, HV, K // 2, V, dtype=dt), torch.zeros(B, HV, 1, V)
    h = h0.clone()
    pk, pv = torch.zeros(B, 2, S, HV, K).bfloat16(), torch.zeros(B, 2, S, HV, V).bfloat16()
    pg, pb = torch.zeros(B, 2, S, HV), torch.zeros(B, 2, S, HV).bfloat16()
    par = torch.zeros((), dtype=torch.int32)
    i32 = lambda x: x.to(torch.int32).contiguous()  # noqa: E731
    gdn_verify_deferred(q1, k1, v1, g1, b1, h, sd, i32(rows1[:, 0]), pk, pv, pg, pb, par, i32(torch.full((B,), 0)),
                        i32(torch.zeros(B)), i32(torch.full((B,), 2)), i32(torch.ones(B)), cd, cq, cqs)
    p0, s0 = qf(h0, K)
    ok1 = torch.equal(cq, p0) and torch.allclose(cqs, s0) and torch.equal(h, h0)
    par.fill_(1)
    o2 = gdn_verify_deferred(q2, k2, v2, g2, b2, h, sd, i32(rows2[:, 0]), pk, pv, pg, pb, par, i32(torch.where(have, 0, -1)),
                             i32(offs), i32(lens), i32(torch.zeros(B)), cd, cq, cqs)
    p1, s1 = qf(h1c, K)
    print(f"committed {bits}-bit: step1 cq == quant(h0) {ok1} | step2 o max diff {(o2.float() - o2r.float()).abs().max().item():.1e}, "
          f"cq == quant(committed) {torch.equal(cq, p1)} (scale close {torch.allclose(cqs, s1)}), h untouched {torch.equal(h, h0)}")
