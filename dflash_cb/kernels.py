"""Triton kernels for Gated DeltaNet verification with state snapshots, and for replaying accepted tokens."""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["T", "store_t"])
def _gdn_verify_kernel(q, k, v, g, beta, o, h0, hs, scale, T, store_t, slot_map, HV: tl.constexpr, K: tl.constexpr,
                       V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, USE_QK_L2NORM: tl.constexpr,
                       MODE: tl.constexpr, NS: tl.constexpr):
    """Gated delta rule over T tokens from h0 (read only), same op order as fla fused_recurrent.
    MODE 0: no snapshot. MODE 1: state after every token -> hs[n, t] (full state).
    MODE 2: state after token index store_t only -> hs[n] (single snapshot).
    MODE 4: per-row slots. slot_map[n, t] = slot index s (or -1): state after token index t -> hs[n, s] (NS slots)."""
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]
    bos = i_n * T
    p_q = q + (bos * HV + i_hv) * K + o_k
    p_k = k + (bos * HV + i_hv) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    p_g = g + bos * HV + i_hv
    p_b = beta + bos * HV + i_hv
    p_o = o + (bos * HV + i_hv) * V + o_v
    if MODE == 1:
        p_h = hs + (bos * HV + i_hv) * K * V + o_k[:, None] * V + o_v[None, :]
    else:
        p_h = hs + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
    b_h = tl.load(h0 + i_nh * K * V + o_k[:, None] * V + o_v[None, :], mask=mask_h, other=0).to(tl.float32)
    for t in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        if USE_QK_L2NORM:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        b_beta = tl.load(p_b).to(tl.float32)
        b_h *= tl.exp(tl.load(p_g).to(tl.float32))
        b_v = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
        b_h += b_k[:, None] * b_v[None, :]
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
        if MODE == 1:
            tl.store(p_h, b_h, mask=mask_h)
            p_h += HV * K * V
        if MODE == 2:
            if t == store_t:
                tl.store(p_h, b_h, mask=mask_h)
        if MODE == 4:
            s = tl.load(slot_map + i_n * T + t)
            if s >= 0:
                tl.store(p_h + (i_n * (NS - 1) + s) * HV * K * V, b_h, mask=mask_h)
        p_q += HV * K
        p_k += HV * K
        p_v += HV * V
        p_g += HV
        p_b += HV
        p_o += HV * V


def gdn_verify(q, k, v, g, beta, h0, hs, store=True, store_t=None, slot_map=None):
    """q, k: [B, T, HV, K]; v: [B, T, HV, V]; g, beta: [B, T, HV]; h0: [B, HV, K, V] fp32 (not modified).
    store_t None: hs [B, T, HV, K, V] gets the state after every token (when store).
    store_t = t : hs [B, HV, K, V] gets the state after token index t (when store).
    slot_map    : int tensor [B, T] (slot or -1), hs [B, NS, HV, K, V] gets row n's state after token t in slot
                  slot_map[n, t]. Returns o [B, T, HV, V]."""
    q, k, v, g, beta = (x.contiguous() for x in (q, k, v, g, beta))
    B, T, HV, K = k.shape
    V = v.shape[-1]
    assert h0.is_contiguous() and hs.is_contiguous()
    mode = 0 if not store else (4 if slot_map is not None else (1 if store_t is None else 2))
    smap = slot_map.to(torch.int32).contiguous() if slot_map is not None else h0     # dummy pointer unless MODE 4
    ns = hs.shape[1] if mode == 4 else 1
    o = torch.empty_like(v)
    grid = (triton.cdiv(V, 8), B * HV)
    _gdn_verify_kernel[grid](q, k, v, g, beta, o, h0, hs, K ** -0.5, T, -1 if store_t is None else store_t, smap,
                             HV=HV, K=K, V=V, BK=triton.next_power_of_2(K), BV=8, USE_QK_L2NORM=True, MODE=mode, NS=ns,
                             num_warps=1, num_stages=3)
    return o


@triton.jit(do_not_specialize=["T"])
def _gdn_replay_kernel(k, v, g, beta, h, offs, lens, T, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                       BK: tl.constexpr, BV: tl.constexpr, USE_QK_L2NORM: tl.constexpr):
    """In place: h[n] <- h[n] advanced over block tokens [offs[n], offs[n] + lens[n]) (fla fused_recurrent order)."""
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    n_t = tl.load(lens + i_n)
    if n_t > 0:
        off = tl.load(offs + i_n)
        o_k = tl.arange(0, BK)
        o_v = i_v * BV + tl.arange(0, BV)
        mask_k = o_k < K
        mask_v = o_v < V
        mask_h = mask_k[:, None] & mask_v[None, :]
        bos = i_n * T + off
        p_k = k + (bos * HV + i_hv) * K + o_k
        p_v = v + (bos * HV + i_hv) * V + o_v
        p_g = g + bos * HV + i_hv
        p_b = beta + bos * HV + i_hv
        p_h = h + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        b_h = tl.load(p_h, mask=mask_h, other=0).to(tl.float32)
        for _ in range(0, n_t):
            b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
            b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
            if USE_QK_L2NORM:
                b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
            b_beta = tl.load(p_b).to(tl.float32)
            b_h *= tl.exp(tl.load(p_g).to(tl.float32))
            b_v = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
            b_h += b_k[:, None] * b_v[None, :]
            p_k += HV * K
            p_v += HV * V
            p_g += HV
            p_b += HV
        tl.store(p_h, b_h, mask=mask_h)


def gdn_replay(k, v, g, beta, h, offs, lens):
    """k: [B, T, HV, K]; v: [B, T, HV, V]; g, beta: [B, T, HV]; h: [B, HV, K, V] fp32 updated in place."""
    k, v, g, beta = (x.contiguous() for x in (k, v, g, beta))
    B, T, HV, K = k.shape
    V = v.shape[-1]
    assert h.is_contiguous()
    grid = (triton.cdiv(V, 8), B * HV)
    _gdn_replay_kernel[grid](k, v, g, beta, h, offs.to(torch.int32), lens.to(torch.int32), T, HV=HV, K=K, V=V,
                             BK=triton.next_power_of_2(K), BV=8, USE_QK_L2NORM=True, num_warps=1, num_stages=3)


@triton.jit
def _cond_copy_kernel(dst, src, idx, D, BLOCK: tl.constexpr):
    """dst[r, :] = src[idx[r], :] where idx[r] >= 0; rows with idx[r] < 0 are left untouched (no memory traffic)."""
    i_b, r = tl.program_id(0), tl.program_id(1)
    s = tl.load(idx + r)
    if s >= 0:
        o = i_b * BLOCK + tl.arange(0, BLOCK)
        m = o < D
        tl.store(dst + r.to(tl.int64) * D + o, tl.load(src + s.to(tl.int64) * D + o, mask=m), mask=m)


def cond_copy_rows(dst, src, idx):
    """Fixed-shape gather of whole rows for CUDA-graph capture. dst: [B, ...], src: [M, ...] (same row shape, both
    contiguous), idx: int [B] (row of src, or -1 to skip)."""
    assert dst.is_contiguous() and src.is_contiguous()
    B, D = dst.shape[0], dst[0].numel()
    BLOCK = 4096
    _cond_copy_kernel[(triton.cdiv(D, BLOCK), B)](dst, src, idx.contiguous(), D, BLOCK=BLOCK)


# ---------------------------------------------------------------- 4-bit / 8-bit snapshots (youngjoon, 10/24)
# A quantised snapshot of a [HV, K, V] state: symmetric round-to-nearest with one fp32 scale (absmax / qmax) per G
# consecutive key rows of each value column (G = K: one scale per value channel), scale [HV, K/G, V] fp32.
#   4-bit: -7..7 stored +8 as a nibble, two key rows per byte (low nibble = even row): q [HV, K/2, V] uint8.
#   8-bit: -127..127, two key rows per int16 (low byte = even row, both signed): q [HV, K/2, V] int16 (same tile
#          handling as 4-bit; a plain int8 store changes the register layout of the whole recurrence).

@triton.jit(do_not_specialize=["T"])
def _gdn_verify_slots_kernel(q, k, v, g, beta, o, h0, hs, hsc, rows, scale, T, HV: tl.constexpr, K: tl.constexpr,
                             V: tl.constexpr, BV: tl.constexpr, G: tl.constexpr, NS: tl.constexpr, NSP: tl.constexpr,
                             BITS: tl.constexpr, USE_QK_L2NORM: tl.constexpr):
    """_gdn_verify_kernel with per-row slots given as block rows (K a power of two): rows[n, j] = t >= 0 -> the state
    after token t goes to slot j of row n. The NS kept rows are read once before the token loop (no per-token slot-map
    load). BITS 4 / 8: the state is quantised in registers and written as hs[n, j] and hsc[n, j] (scales); BITS 0:
    hs[n, j] gets the fp32 state."""
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    o_s = tl.arange(0, NSP)
    mask_v = o_v < V
    bos = i_n * T
    p_q = q + (bos * HV + i_hv) * K + o_k
    p_k = k + (bos * HV + i_hv) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    p_g = g + bos * HV + i_hv
    p_b = beta + bos * HV + i_hv
    p_o = o + (bos * HV + i_hv) * V + o_v
    if NS == 1:
        t_keep = tl.load(rows + i_n)                                   # scalar: the one kept row
    else:
        b_rows = tl.load(rows + i_n * NS + o_s, mask=o_s < NS, other=-1)
    if BITS != 0:
        o_k2 = tl.arange(0, K // 2)
        p_hq = hs + (i_n * NS * HV + i_hv) * (K // 2) * V + o_k2[:, None] * V + o_v[None, :]
    else:
        p_h = hs + (i_n * NS * HV + i_hv) * K * V + o_k[:, None] * V + o_v[None, :]
    if BITS != 0:
        o_g = tl.arange(0, K // G)
        p_hs = hsc + (i_n * NS * HV + i_hv) * (K // G) * V + o_g[:, None] * V + o_v[None, :]
    b_h = tl.load(h0 + i_nh * K * V + o_k[:, None] * V + o_v[None, :], mask=mask_v[None, :], other=0).to(tl.float32)
    for t in range(0, T):
        b_q = tl.load(p_q).to(tl.float32)
        b_k = tl.load(p_k).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        if USE_QK_L2NORM:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        b_beta = tl.load(p_b).to(tl.float32)
        b_h *= tl.exp(tl.load(p_g).to(tl.float32))
        b_v = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
        b_h += b_k[:, None] * b_v[None, :]
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
        if NS == 1:
            s = tl.where(t == t_keep, 0, -1)
        else:
            s = tl.max(tl.where(b_rows == t, o_s, -1))
        if s >= 0:
            if BITS == 4:
                b3 = tl.reshape(b_h, (K // G, G, BV))
                b_s = tl.maximum(tl.max(tl.abs(b3), 1), 1e-30) / 7.0                         # [K/G, BV]
                b_i = tl.minimum(tl.maximum(tl.floor(b3 / b_s[:, None, :] + 8.5), 0.0), 15.0).to(tl.int32)
                lo, hi = tl.split(tl.permute(tl.reshape(b_i, (K // 2, 2, BV)), (0, 2, 1)))   # even / odd key rows
                tl.store(p_hq + s * HV * (K // 2) * V, (lo | (hi << 4)).to(tl.uint8), mask=mask_v[None, :])
                tl.store(p_hs + s * HV * (K // G) * V, b_s, mask=mask_v[None, :])
            elif BITS == 8:
                b3 = tl.reshape(b_h, (K // G, G, BV))
                b_s = tl.maximum(tl.max(tl.abs(b3), 1), 1e-30) / 127.0                       # [K/G, BV]
                b_i = tl.minimum(tl.maximum(tl.floor(b3 / b_s[:, None, :] + 0.5), -127.0), 127.0).to(tl.int32)
                lo, hi = tl.split(tl.permute(tl.reshape(b_i, (K // 2, 2, BV)), (0, 2, 1)))   # even / odd key rows
                tl.store(p_hq + s * HV * (K // 2) * V, ((lo & 255) | (hi << 8)).to(tl.int16), mask=mask_v[None, :])
                tl.store(p_hs + s * HV * (K // G) * V, b_s, mask=mask_v[None, :])
            else:
                tl.store(p_h + s * HV * K * V, b_h, mask=mask_v[None, :])
        p_q += HV * K
        p_k += HV * K
        p_v += HV * V
        p_g += HV
        p_b += HV
        p_o += HV * V


BITS_OF = {torch.float32: 0, torch.uint8: 4, torch.int16: 8}


def gdn_verify_slots(q, k, v, g, beta, h0, hs, rows, hsc=None):
    """gdn_verify keeping the state after block row rows[n, j] (>= 0) in slot j of row n. rows: int [B, NS]
    (int32 contiguous avoids a conversion per call). hs: [B, NS, HV, K, V] fp32; with hsc [B, NS, HV, K/G, V] fp32
    scales (G = K / hsc.shape[-2]): hs [B, NS, HV, K/2, V] uint8 (4-bit) or int16 (8-bit).
    Returns o [B, T, HV, V]."""
    q, k, v, g, beta = (x.contiguous() for x in (q, k, v, g, beta))
    B, T, HV, K = k.shape
    V = v.shape[-1]
    bits = BITS_OF[hs.dtype]
    quant = bits != 0
    assert h0.is_contiguous() and hs.is_contiguous() and K & (K - 1) == 0 and quant == (hsc is not None)
    assert not quant or hs.shape[-2] * 2 == K
    NS = rows.shape[1]
    o = torch.empty_like(v)
    _gdn_verify_slots_kernel[(triton.cdiv(V, 8), B * HV)](
        q, k, v, g, beta, o, h0, hs, hsc if quant else hs, rows.to(torch.int32).contiguous(), K ** -0.5, T, HV=HV,
        K=K, V=V, BV=8, G=K // hsc.shape[-2] if quant else K, NS=NS, NSP=triton.next_power_of_2(NS), BITS=bits,
        USE_QK_L2NORM=True, num_warps=1, num_stages=3)
    return o


@triton.jit(do_not_specialize=["T"])
def _gdn_replay_from_kernel(k, v, g, beta, h, offs, lens, src, hq, hsc, T, HV: tl.constexpr, K: tl.constexpr,
                            V: tl.constexpr, BV: tl.constexpr, G: tl.constexpr, BITS: tl.constexpr,
                            USE_QK_L2NORM: tl.constexpr):
    """_gdn_replay_kernel whose starting state is snapshot row src[n] (if >= 0; fp32 hq [M, HV, K, V] when BITS 0,
    else quantised hq with scales hsc) instead of h[n]: one read of the snapshot, the replay, one write of h[n]."""
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    n_t = tl.load(lens + i_n)
    s = tl.load(src + i_n)
    if (n_t > 0) | (s >= 0):
        off = tl.load(offs + i_n)
        o_k = tl.arange(0, K)
        o_v = i_v * BV + tl.arange(0, BV)
        mask_v = o_v < V
        m2 = mask_v[None, :]
        bos = i_n * T + off
        p_k = k + (bos * HV + i_hv) * K + o_k
        p_v = v + (bos * HV + i_hv) * V + o_v
        p_g = g + bos * HV + i_hv
        p_b = beta + bos * HV + i_hv
        p_h = h + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        if s >= 0:
            row = s.to(tl.int64) * HV + i_hv
            if BITS == 4:
                o_k2 = tl.arange(0, K // 2)
                o_g = tl.arange(0, K // G)
                b_p = tl.load(hq + row * (K // 2) * V + o_k2[:, None] * V + o_v[None, :], mask=m2, other=0).to(tl.int32)
                b_i = tl.reshape(tl.permute(tl.join((b_p & 15) - 8, (b_p >> 4) - 8), (0, 2, 1)), (K // G, G, BV))
                b_s = tl.load(hsc + row * (K // G) * V + o_g[:, None] * V + o_v[None, :], mask=m2, other=0)
                b_h = tl.reshape(b_i.to(tl.float32) * b_s[:, None, :], (K, BV))
            elif BITS == 8:
                o_k2 = tl.arange(0, K // 2)
                o_g = tl.arange(0, K // G)
                b_p = tl.load(hq + row * (K // 2) * V + o_k2[:, None] * V + o_v[None, :], mask=m2, other=0).to(tl.int32)
                lo = ((b_p & 255) ^ 128) - 128                                                 # signed low byte
                b_i = tl.reshape(tl.permute(tl.join(lo, b_p >> 8), (0, 2, 1)), (K // G, G, BV))
                b_s = tl.load(hsc + row * (K // G) * V + o_g[:, None] * V + o_v[None, :], mask=m2, other=0)
                b_h = tl.reshape(b_i.to(tl.float32) * b_s[:, None, :], (K, BV))
            else:
                b_h = tl.load(hq + row * K * V + o_k[:, None] * V + o_v[None, :], mask=m2, other=0).to(tl.float32)
        else:
            b_h = tl.load(p_h, mask=m2, other=0).to(tl.float32)
        for _ in range(0, n_t):
            b_k = tl.load(p_k).to(tl.float32)
            b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
            if USE_QK_L2NORM:
                b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
            b_beta = tl.load(p_b).to(tl.float32)
            b_h *= tl.exp(tl.load(p_g).to(tl.float32))
            b_v = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
            b_h += b_k[:, None] * b_v[None, :]
            p_k += HV * K
            p_v += HV * V
            p_g += HV
            p_b += HV
        tl.store(p_h, b_h, mask=m2)


def gdn_replay_from(k, v, g, beta, h, offs, lens, src, snap, scale=None):
    """gdn_replay where row n starts from snapshot row src[n] of snap (when src[n] >= 0) instead of h[n].
    snap: [M, HV, K, V] fp32, or with scale [M, HV, K/G, V]: [M, HV, K/2, V] uint8 (4-bit) / int16 (8-bit)."""
    k, v, g, beta = (x.contiguous() for x in (k, v, g, beta))
    B, T, HV, K = k.shape
    V = v.shape[-1]
    assert h.is_contiguous() and snap.is_contiguous() and K & (K - 1) == 0
    bits = BITS_OF[snap.dtype]
    quant = bits != 0
    assert quant == (scale is not None)
    _gdn_replay_from_kernel[(triton.cdiv(V, 8), B * HV)](
        k, v, g, beta, h, offs.to(torch.int32), lens.to(torch.int32), src.to(torch.int32), snap,
        scale if quant else snap, T, HV=HV, K=K, V=V, BV=8, G=K // scale.shape[-2] if quant else K, BITS=bits,
        USE_QK_L2NORM=True, num_warps=1, num_stages=3)


def dequant_q4(hq, hsc):
    """Reference (torch) dequantisation: hq [..., K/2, V] uint8, hsc [..., K/G, V] -> [..., K, V] fp32."""
    lo, hi = (hq & 15).int() - 8, (hq >> 4).int() - 8
    x = torch.stack([lo, hi], -2).flatten(-3, -2).float()                     # rows 2j, 2j+1
    G = x.shape[-2] // hsc.shape[-2]
    return (x.unflatten(-2, (hsc.shape[-2], G)) * hsc.unsqueeze(-2)).flatten(-3, -2)


def dequant_q8(hq, hsc):
    """Reference (torch) dequantisation: hq [..., K/2, V] int16 (two signed bytes), hsc [..., K/G, V] -> [..., K, V]."""
    x = hq.int()
    lo, hi = ((x & 255) ^ 128) - 128, x >> 8
    y = torch.stack([lo, hi], -2).flatten(-3, -2).float()                    # rows 2j, 2j+1
    G = y.shape[-2] // hsc.shape[-2]
    return (y.unflatten(-2, (hsc.shape[-2], G)) * hsc.unsqueeze(-2)).flatten(-3, -2)


# ---------------------------------------------------------------- argmax + its softmax probability in one pass
@triton.jit
def _argmax_lse_part_kernel(x, pm, pi, ps, Vv, CH, NSPLIT: tl.constexpr, BLOCK: tl.constexpr):
    """Row r, split sp of x [R, Vv]: max, first index of the max and sum(exp(x - max)) over [sp*CH, (sp+1)*CH)."""
    r, sp = tl.program_id(0), tl.program_id(1)
    base = x + r.to(tl.int64) * Vv
    lo = sp * CH
    hi = tl.minimum(lo + CH, Vv)
    m = tl.full((), float("-inf"), tl.float32)
    idx = tl.full((), 0, tl.int32)
    acc = tl.full((), 0.0, tl.float32)
    for off in range(lo, hi, BLOCK):
        o = off + tl.arange(0, BLOCK)
        v = tl.load(base + o, mask=o < hi, other=float("-inf")).to(tl.float32)
        cm = tl.max(v, 0)
        ci = tl.argmax(v, 0, tie_break_left=True) + off
        nm = tl.maximum(m, cm)
        acc = acc * tl.exp(m - nm) + tl.sum(tl.exp(v - nm), 0)
        idx = tl.where(cm > m, ci, idx)                  # strict: the first maximum wins, as torch.argmax
        m = nm
    tl.store(pm + r * NSPLIT + sp, m)
    tl.store(pi + r * NSPLIT + sp, idx)
    tl.store(ps + r * NSPLIT + sp, acc)


def argmax_maxprob(x, block=4096, nsplit=16):
    """x [R, Vv] logits -> (argmax [R] int64, softmax probability of the argmax [R] fp32), reading x once.
    Equal to (x.argmax(-1), softmax(x.float()).amax(-1)) up to fp32 rounding of the probability."""
    x = x.contiguous()
    R, Vv = x.shape
    ch = triton.cdiv(triton.cdiv(Vv, nsplit), block) * block
    ns = triton.cdiv(Vv, ch)                                     # no empty split
    pm = torch.empty(R, ns, device=x.device)
    pi = torch.empty(R, ns, device=x.device, dtype=torch.int32)
    ps = torch.empty(R, ns, device=x.device)
    _argmax_lse_part_kernel[(R, ns)](x, pm, pi, ps, Vv, ch, NSPLIT=ns, BLOCK=block, num_warps=8)
    m, j = pm.max(1)                                             # first split holding the maximum
    p = 1.0 / (ps * torch.exp(pm - m[:, None])).sum(1)
    return pi.gather(1, j[:, None])[:, 0].long(), p


# ---------------------------------------------------------------- deferred restore (youngjoon, 10/24)
@triton.jit(do_not_specialize=["T"])
def _gdn_verify_deferred_kernel(q, k, v, g, beta, o, h, hs, hsc, rows, pk, pv, pg, pb, par, pslot, poffs, plens, fresh,
                                cq, cqs, scale, T, RSK, RSV, RSG, RSH, RSC, RSQ, RSQS, HV: tl.constexpr, K: tl.constexpr,
                                V: tl.constexpr, BV: tl.constexpr, G: tl.constexpr, NS: tl.constexpr, BITS: tl.constexpr,
                                CQB: tl.constexpr, USE_QK_L2NORM: tl.constexpr):
    """_gdn_verify_slots_kernel (NS = 1) that first finishes the previous step's restore, deferred to here: the start
    state is snapshot slot pslot[n] of row n (fp32 when BITS 0, else 8-bit / 4-bit) or h[n] (pslot < 0), advanced over the
    previous block's tokens [poffs[n], poffs[n] + plens[n]) read from half 1 - par of pk / pv / pg / pb; it is written
    to h[n] (the committed state, kept for a later fallback) and the new block is verified from it. Rows with fresh[n]
    (inserted since the last step) start from h[n] as is. The block's raw k, v, g, beta go to half par for the next
    step. pk [B, 2, T, HV, K], pv [B, 2, T, HV, V], pg / pb [B, 2, T, HV], hs / hsc [B, 1, ...]: each contiguous within a
    row; RS*: row strides (elements) of pk, pv, pg / pb, hs, hsc (the rollback keeps all layers in one buffer per kind).
    CQB 8 / 4: the committed state is kept quantised in cq / cqs ([B, HV, K/2, V] int16 / uint8, [B, HV, K/G, V])
    instead of fp32 in h: a non-fresh row without a usable snapshot starts from cq, every row writes its committed state
    to cq (fresh rows start from h, the prefill state), h is not written."""
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_v = o_v < V
    m2 = mask_v[None, :]
    n64 = i_n.to(tl.int64)
    pk = pk + n64 * RSK                                          # this row's buffers
    pv = pv + n64 * RSV
    pg = pg + n64 * RSG
    pb = pb + n64 * RSG
    hs = hs + n64 * RSH
    hsc = hsc + n64 * RSC
    cq = cq + n64 * RSQ
    cqs = cqs + n64 * RSQS
    cur = tl.load(par)
    prv = 1 - cur
    fr = tl.load(fresh + i_n)
    sl = tl.where(fr != 0, -1, tl.load(pslot + i_n))
    nt = tl.where(fr != 0, 0, tl.load(plens + i_n))
    p_h = h + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
    # ---- the previous step's restore
    if sl >= 0:
        row = sl.to(tl.int64) * HV + i_hv
        if BITS != 0:
            o_k2 = tl.arange(0, K // 2)
            o_g = tl.arange(0, K // G)
            b_p = tl.load(hs + row * (K // 2) * V + o_k2[:, None] * V + o_v[None, :], mask=m2, other=0).to(tl.int32)
            if BITS == 8:
                lo = ((b_p & 255) ^ 128) - 128
                hi = b_p >> 8
            else:
                lo = (b_p & 15) - 8
                hi = (b_p >> 4) - 8
            b_i = tl.reshape(tl.permute(tl.join(lo, hi), (0, 2, 1)), (K // G, G, BV))
            b_s = tl.load(hsc + row * (K // G) * V + o_g[:, None] * V + o_v[None, :], mask=m2, other=0)
            b_h = tl.reshape(b_i.to(tl.float32) * b_s[:, None, :], (K, BV))
        else:
            b_h = tl.load(hs + row * K * V + o_k[:, None] * V + o_v[None, :], mask=m2, other=0).to(tl.float32)
    else:
        if CQB != 0:
            if fr == 0:
                o_k2 = tl.arange(0, K // 2)
                o_g = tl.arange(0, K // G)
                b_p = tl.load(cq + i_hv * (K // 2) * V + o_k2[:, None] * V + o_v[None, :], mask=m2, other=0).to(tl.int32)
                if CQB == 8:
                    lo = ((b_p & 255) ^ 128) - 128
                    hi = b_p >> 8
                else:
                    lo = (b_p & 15) - 8
                    hi = (b_p >> 4) - 8
                b_i = tl.reshape(tl.permute(tl.join(lo, hi), (0, 2, 1)), (K // G, G, BV))
                b_s = tl.load(cqs + i_hv * (K // G) * V + o_g[:, None] * V + o_v[None, :], mask=m2, other=0)
                b_h = tl.reshape(b_i.to(tl.float32) * b_s[:, None, :], (K, BV))
            else:
                b_h = tl.load(p_h, mask=m2, other=0).to(tl.float32)
        else:
            b_h = tl.load(p_h, mask=m2, other=0).to(tl.float32)
    if nt > 0:
        off = tl.load(poffs + i_n)
        pb0 = prv * T + off
        p_k = pk + (pb0 * HV + i_hv) * K + o_k
        p_v = pv + (pb0 * HV + i_hv) * V + o_v
        p_g = pg + pb0 * HV + i_hv
        p_b = pb + pb0 * HV + i_hv
        for _ in range(0, nt):
            b_k = tl.load(p_k).to(tl.float32)
            b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
            if USE_QK_L2NORM:
                b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
            b_beta = tl.load(p_b).to(tl.float32)
            b_h *= tl.exp(tl.load(p_g).to(tl.float32))
            b_v = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
            b_h += b_k[:, None] * b_v[None, :]
            p_k += HV * K
            p_v += HV * V
            p_g += HV
            p_b += HV
    if CQB != 0:
        o_k2 = tl.arange(0, K // 2)
        o_g = tl.arange(0, K // G)
        b3 = tl.reshape(b_h, (K // G, G, BV))
        p_cq = cq + i_hv * (K // 2) * V + o_k2[:, None] * V + o_v[None, :]
        if CQB == 8:
            b_s = tl.maximum(tl.max(tl.abs(b3), 1), 1e-30) / 127.0
            b_i = tl.minimum(tl.maximum(tl.floor(b3 / b_s[:, None, :] + 0.5), -127.0), 127.0).to(tl.int32)
            lo, hi = tl.split(tl.permute(tl.reshape(b_i, (K // 2, 2, BV)), (0, 2, 1)))
            tl.store(p_cq, ((lo & 255) | (hi << 8)).to(tl.int16), mask=m2)
        else:
            b_s = tl.maximum(tl.max(tl.abs(b3), 1), 1e-30) / 7.0
            b_i = tl.minimum(tl.maximum(tl.floor(b3 / b_s[:, None, :] + 8.5), 0.0), 15.0).to(tl.int32)
            lo, hi = tl.split(tl.permute(tl.reshape(b_i, (K // 2, 2, BV)), (0, 2, 1)))
            tl.store(p_cq, (lo | (hi << 4)).to(tl.uint8), mask=m2)
        tl.store(cqs + i_hv * (K // G) * V + o_g[:, None] * V + o_v[None, :], b_s, mask=m2)
    elif (sl >= 0) | (nt > 0):
        tl.store(p_h, b_h, mask=m2)
    # ---- this step's verify (as _gdn_verify_slots_kernel), saving the raw inputs for the next step
    bos = i_n * T
    p_q = q + (bos * HV + i_hv) * K + o_k
    p_k = k + (bos * HV + i_hv) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    p_g = g + bos * HV + i_hv
    p_b = beta + bos * HV + i_hv
    p_o = o + (bos * HV + i_hv) * V + o_v
    sb = cur * T
    s_k = pk + (sb * HV + i_hv) * K + o_k
    s_v = pv + (sb * HV + i_hv) * V + o_v
    s_g = pg + sb * HV + i_hv
    s_b = pb + sb * HV + i_hv
    t_keep = tl.load(rows + i_n)
    if BITS != 0:
        o_k2 = tl.arange(0, K // 2)
        o_g = tl.arange(0, K // G)
        p_hq = hs + i_hv * (K // 2) * V + o_k2[:, None] * V + o_v[None, :]
        p_hsc = hsc + i_hv * (K // G) * V + o_g[:, None] * V + o_v[None, :]
    else:
        p_hf = hs + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
    for t in range(0, T):
        r_k = tl.load(p_k)
        r_v = tl.load(p_v, mask=mask_v, other=0)
        r_g = tl.load(p_g)
        r_b = tl.load(p_b)
        tl.store(s_v, r_v.to(s_v.dtype.element_ty), mask=mask_v)
        if i_v == 0:
            tl.store(s_k, r_k.to(s_k.dtype.element_ty))
            tl.store(s_g, r_g.to(s_g.dtype.element_ty))
            tl.store(s_b, r_b.to(s_b.dtype.element_ty))
        b_q = tl.load(p_q).to(tl.float32)
        b_k = r_k.to(tl.float32)
        b_v = r_v.to(tl.float32)
        if USE_QK_L2NORM:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        b_beta = r_b.to(tl.float32)
        b_h *= tl.exp(r_g.to(tl.float32))
        b_v = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
        b_h += b_k[:, None] * b_v[None, :]
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
        if t == t_keep:
            if BITS == 8:
                b3 = tl.reshape(b_h, (K // G, G, BV))
                b_s = tl.maximum(tl.max(tl.abs(b3), 1), 1e-30) / 127.0
                b_i = tl.minimum(tl.maximum(tl.floor(b3 / b_s[:, None, :] + 0.5), -127.0), 127.0).to(tl.int32)
                lo, hi = tl.split(tl.permute(tl.reshape(b_i, (K // 2, 2, BV)), (0, 2, 1)))
                tl.store(p_hq, ((lo & 255) | (hi << 8)).to(tl.int16), mask=m2)
                tl.store(p_hsc, b_s, mask=m2)
            elif BITS == 4:
                b3 = tl.reshape(b_h, (K // G, G, BV))
                b_s = tl.maximum(tl.max(tl.abs(b3), 1), 1e-30) / 7.0
                b_i = tl.minimum(tl.maximum(tl.floor(b3 / b_s[:, None, :] + 8.5), 0.0), 15.0).to(tl.int32)
                lo, hi = tl.split(tl.permute(tl.reshape(b_i, (K // 2, 2, BV)), (0, 2, 1)))
                tl.store(p_hq, (lo | (hi << 4)).to(tl.uint8), mask=m2)
                tl.store(p_hsc, b_s, mask=m2)
            else:
                tl.store(p_hf, b_h, mask=m2)
        p_q += HV * K
        p_k += HV * K
        p_v += HV * V
        p_g += HV
        p_b += HV
        p_o += HV * V
        s_k += HV * K
        s_v += HV * V
        s_g += HV
        s_b += HV


def gdn_verify_deferred(q, k, v, g, beta, h, hs, rows, pk, pv, pg, pb, par, pslot, poffs, plens, fresh, hsc=None,
                        cq=None, cqs=None):
    """See _gdn_verify_deferred_kernel. h [B, HV, K, V] fp32 (updated: committed state of the previous step); hs one
    slot per row (fp32 [B, 1, HV, K, V], or int16 8-bit / uint8 4-bit [B, 1, HV, K/2, V] with hsc); rows / pslot /
    poffs / plens / fresh int32 [B]; par int32 scalar tensor; cq / cqs: quantised committed state instead of h, int16
    8-bit or uint8 4-bit (see the kernel). Buffers other than h only need to be contiguous within a row. Returns o [B, T, HV, V]."""
    q, k, v, g, beta = (x.contiguous() for x in (q, k, v, g, beta))
    B, T, HV, K = k.shape
    V = v.shape[-1]
    bits = BITS_OF[hs.dtype]
    assert bits in (0, 4, 8) and hs.shape[1] == 1 and (bits == 0) == (hsc is None) and K & (K - 1) == 0
    assert pk.shape[2] == T and h.is_contiguous() and all(x[0].is_contiguous() for x in (hs, pk, pv, pg, pb))
    assert pg.stride(0) == pb.stride(0) and (hsc is None or hsc[0].is_contiguous())
    use_cq = cq is not None
    cqb = BITS_OF[cq.dtype] if use_cq else 0
    assert not use_cq or (cqb in (4, 8) and cq[0].is_contiguous() and cqs[0].is_contiguous())
    o = torch.empty_like(v)
    _gdn_verify_deferred_kernel[(triton.cdiv(V, 8), B * HV)](
        q, k, v, g, beta, o, h, hs, hsc if bits else hs, rows, pk, pv, pg, pb, par, pslot, poffs, plens, fresh,
        cq if use_cq else hs, cqs if use_cq else hs, K ** -0.5, T, pk.stride(0), pv.stride(0), pg.stride(0),
        hs.stride(0), hsc.stride(0) if bits else 0, cq.stride(0) if use_cq else 0, cqs.stride(0) if use_cq else 0,
        HV=HV, K=K, V=V, BV=8, G=K // hsc.shape[-2] if bits else K, NS=1, BITS=bits, CQB=cqb, USE_QK_L2NORM=True,
        num_warps=1, num_stages=3)
    return o
