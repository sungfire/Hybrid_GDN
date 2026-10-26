"""Logic check of the 4-bit kernels in the Triton interpreter (CPU, tiny shapes). TRITON_INTERPRET=1 must be set."""
import os
import sys

import torch
import triton.runtime.interpreter as _I

_orig_patch = _I._patch_lang_tensor


def _patch(tensor, scope):          # NumPy >= 2.4: int() of a size-1 array that is not 0-d raises
    _orig_patch(tensor, scope)
    scope.set_attr(tensor, "__index__", lambda self: int(self.handle.data.reshape(-1)[0]))


_I._patch_lang_tensor = _patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dflash_cb.kernels import dequant_q4, gdn_replay, gdn_replay_from, gdn_verify, gdn_verify_slots  # noqa: E402
from test_q4 import quant_ref  # noqa: E402

torch.manual_seed(0)
B, S, HV, K, V = 2, 4, 2, 16, 16
q, k, v = torch.randn(B, S, HV, K), torch.randn(B, S, HV, K), torch.randn(B, S, HV, V)
g = -torch.rand(B, S, HV) * 0.1
beta = torch.rand(B, S, HV)
h0 = torch.randn(B, HV, K, V) * 0.1
full = torch.empty(B, S, HV, K, V)
o_ref = gdn_verify(q, k, v, g, beta, h0, full, store=True)
for G in (16, 4):
    rows = torch.tensor([[1, 3]] * B)
    hq = torch.zeros(B, 2, HV, K // 2, V, dtype=torch.uint8)
    hs = torch.zeros(B, 2, HV, K // G, V)
    o = gdn_verify_slots(q, k, v, g, beta, h0, hq, rows, hs)
    pq, ps = quant_ref(full[:, [1, 3]], G)
    deq = dequant_q4(hq, hs)
    rel = ((deq - full[:, [1, 3]]).norm() / full[:, [1, 3]].norm()).item()
    print(f"G={G}: o close {torch.allclose(o, o_ref, atol=1e-5)}, bytes equal {(pq == hq).float().mean().item():.4f}, "
          f"scales close {torch.allclose(ps, hs)}, dequant rel err {rel:.3f}")
    produced = torch.tensor([3, 4])
    src = torch.tensor([0 * 2 + 0, 1 * 2 + 1])      # row 0 from slot 0 (state after token 1), row 1 from slot 1 (after token 3)
    offs = torch.tensor([2, 4])
    lens = produced - offs
    a = torch.stack([deq[0, 0], deq[1, 1]]).contiguous()
    gdn_replay(k, v, g, beta, a, offs, lens)
    c = h0.clone()
    gdn_replay_from(k, v, g, beta, c, offs, lens, src, hq.flatten(0, 1), hs.flatten(0, 1))
    d = h0.clone()
    gdn_replay_from(k, v, g, beta, d, offs, lens, src, full[:, [1, 3]].contiguous().flatten(0, 1))
    e = h0.clone()
    gdn_replay_from(k, v, g, beta, e, torch.zeros(2, dtype=torch.long), produced, torch.tensor([-1, -1]),
                    full[:, [1, 3]].contiguous().flatten(0, 1))
    exact = full[torch.arange(B), produced - 1]
    print(f"  replay_from q4 == dequant+replay {torch.allclose(a, c, atol=1e-6)}, fp32 snap vs exact "
          f"{(d - exact).abs().max().item():.1e}, from S0 vs exact {(e - exact).abs().max().item():.1e}")
