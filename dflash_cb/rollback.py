"""GDN rollback strategies behind one interface (selected by name, e.g. --rollback fullstate).

The engine calls, for every DFlash step over the n active rows:
  verify_gdn(i, q, k, v, g, beta, conv_full, n) -> o   inside the verify forward of GDN layer i. Must only READ the
                                                         durable states engine.rec[i] / engine.conv[i].
  save(n)                                               after the verify forward (timed as "state save").
  restore(produced, n, accepted)                        after acceptance: durable state of row r <- state after the
                                                         first produced[r] tokens of the block (timed as "restore").
conv_full[r] = [k-1 previous conv inputs | the block's conv inputs]  ([n, C, k-1+S]).
"""
import json
import os

import torch

from .kernels import (cond_copy_rows, gdn_replay, gdn_replay_from, gdn_verify, gdn_verify_deferred,
                      gdn_verify_slots)


class Rollback:
    name = "base"
    stores_snapshots = True     # False: nothing is written during verification (no store-overhead calibration)
    time_kernels = True         # per-kernel CUDA events around the verify kernels (off inside CUDA graphs)

    def _ev(self):
        if not self.time_kernels:
            return None
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return e

    def _ev_end(self, e0):
        if e0 is not None:
            e1 = torch.cuda.Event(enable_timing=True)
            e1.record()
            self.events.append((e0, e1))

    def __init__(self, engine, bs, block):
        self.eng, self.bs, self.S = engine, bs, block

    def buffer_bytes(self):
        return {}

    def verify_gdn(self, i, q, k, v, g, beta, conv_full, n):
        raise NotImplementedError

    def save(self, n):
        pass

    def restore(self, produced, n, accepted=None):
        raise NotImplementedError

    def reset_stats(self):
        pass

    def store_kwargs(self):
        """gdn_verify keyword arguments that reproduce this strategy's snapshot writes (calibration)."""
        return {}


class FullStateRollback(Rollback):
    """Keep the recurrent state and the short-conv state after every block position; restore = copy, no recompute.

    Recurrent snapshots are written by the verify kernel itself (one fused pass: cheapest way to obtain per-position
    states). Conv snapshots are materialised from conv_full in save(). Buffers are allocated once for `bs` rows and
    reused every step (stale positions are simply overwritten)."""
    name = "fullstate"

    def __init__(self, engine, bs, block):
        super().__init__(engine, bs, block)
        e = engine
        self.snap_rec = {i: torch.empty(bs, block, *e.rec[i].shape[1:], device=e.dev, dtype=e.rec[i].dtype)
                         for i in e.gdn}
        self.snap_conv = {i: torch.empty(bs, block, *e.conv[i].shape[1:], device=e.dev, dtype=e.conv[i].dtype)
                          for i in e.gdn}
        self.conv_full = {}
        self.events = []          # (start, end) CUDA events around the fused verify kernels of the current step
        self.capture = None       # dict to stash kernel inputs (calibration / state check), or None

    def buffer_bytes(self):
        rec = sum(x.numel() * x.element_size() for x in self.snap_rec.values())
        conv = sum(x.numel() * x.element_size() for x in self.snap_conv.values())
        per_row_layer = (self.snap_rec[self.eng.gdn[0]][0].numel() * 4
                         + self.snap_conv[self.eng.gdn[0]][0].numel() * self.snap_conv[self.eng.gdn[0]].element_size())
        return dict(rec_bytes=rec, conv_bytes=conv, total_bytes=rec + conv, layers=len(self.snap_rec),
                    positions=self.S, batch=self.bs, bytes_per_row_per_layer=per_row_layer)

    def verify_gdn(self, i, q, k, v, g, beta, conv_full, n):
        e0 = self._ev()
        o = gdn_verify(q, k, v, g, beta, self.eng.rec[i][:n], self.snap_rec[i][:n], store=True)
        self._ev_end(e0)
        self.conv_full[i] = conv_full
        if self.capture is not None:
            self.capture[i] = (q, k, v, g, beta)
        return o

    def save(self, n):
        for i, full in self.conv_full.items():
            km1 = full.shape[-1] - self.S
            # conv state after t+1 block tokens = conv inputs [t+1, t+1+km1) of conv_full
            win = full.unfold(2, km1, 1)[:, :, 1:self.S + 1]                       # [n, C, S, km1]
            self.snap_conv[i][:n].copy_(win.permute(0, 2, 1, 3))
        self.conv_full = {}

    def restore(self, produced, n, accepted=None):
        idx = torch.arange(n, device=produced.device) * self.S + (produced - 1)
        for i in self.snap_rec:
            torch.index_select(self.snap_rec[i][:n].flatten(0, 1), 0, idx, out=self.eng.rec[i][:n])
            torch.index_select(self.snap_conv[i][:n].flatten(0, 1), 0, idx, out=self.eng.conv[i][:n])

    def kernel_ms(self):
        """GPU time (ms) of this step's fused verify kernels (call after a synchronize)."""
        t = sum(a.elapsed_time(b) for a, b in self.events)
        self.events = []
        return t


class ModeALRollback(Rollback):
    """Snapshot only at the most frequent acceptance length so far (m); restore = copy, or replay (SpecLA-style
    recomputation) from the nearest state at or below the accepted length:
        AL == m : copy the snapshot
        AL >  m : copy the snapshot, replay block tokens [m, AL)
        AL <  m : replay block tokens [0, AL) from the durable state S_0 (never modified by verification)
    AL = tokens committed in a step (pending token + accepted drafts). The histogram is online (past steps only),
    reset per run by reset_stats(); before the first observation m = block // 2; ties go to the smaller AL.
    Conv state: window at m snapshotted; other ALs take their window from the verify pass's conv inputs."""
    name = "modeal"

    def __init__(self, engine, bs, block):
        super().__init__(engine, bs, block)
        e = engine
        self.snap_rec = {i: torch.empty(bs, *e.rec[i].shape[1:], device=e.dev, dtype=e.rec[i].dtype) for i in e.gdn}
        self.snap_conv = {i: torch.empty(bs, *e.conv[i].shape[1:], device=e.dev, dtype=e.conv[i].dtype) for i in e.gdn}
        self.inputs, self.conv_full, self.events, self.capture = {}, {}, [], None
        self.reset_stats()

    def store_kwargs(self):
        return dict(store_t=self.m - 1)

    def reset_stats(self):
        self.hist = [0] * (self.S + 1)
        self.m = self.S // 2
        self.m_trace = []

    def buffer_bytes(self):
        rec = sum(x.numel() * x.element_size() for x in self.snap_rec.values())
        conv = sum(x.numel() * x.element_size() for x in self.snap_conv.values())
        return dict(rec_bytes=rec, conv_bytes=conv, total_bytes=rec + conv, layers=len(self.snap_rec), positions=1,
                    batch=self.bs)

    def verify_gdn(self, i, q, k, v, g, beta, conv_full, n):
        e0 = self._ev()
        o = gdn_verify(q, k, v, g, beta, self.eng.rec[i][:n], self.snap_rec[i][:n], store=True, store_t=self.m - 1)
        self._ev_end(e0)
        self.inputs[i] = (k, v, g, beta)
        self.conv_full[i] = conv_full
        if self.capture is not None:
            self.capture[i] = (q, k, v, g, beta)
        return o

    def save(self, n):
        for i, full in self.conv_full.items():
            km1 = full.shape[-1] - self.S
            self.snap_conv[i][:n].copy_(full[:, :, self.m:self.m + km1])

    def restore(self, produced, n, accepted=None):
        m = self.m
        dev = produced.device
        ge = produced >= m
        rows_ge = ge.nonzero().flatten()
        offs = torch.where(ge, m, 0)
        lens = produced - offs
        km1 = self.eng.conv[self.eng.gdn[0]].shape[-1]
        widx = (produced[:, None] + torch.arange(km1, device=dev)[None])[:, None, :]
        for i in self.snap_rec:
            rec = self.eng.rec[i][:n]
            rec.index_copy_(0, rows_ge, self.snap_rec[i][:n].index_select(0, rows_ge))
            k, v, g, beta = self.inputs[i]
            gdn_replay(k, v, g, beta, rec, offs, lens)
            full = self.conv_full[i]
            self.eng.conv[i][:n].copy_(torch.gather(full, 2, widx.expand(-1, full.shape[1], -1)))
        self.inputs, self.conv_full = {}, {}
        # update the acceptance-length histogram with this step's (uncapped) acceptance lengths
        for a in (accepted if accepted is not None else produced).tolist():
            self.hist[a] += 1
        best = max(range(1, self.S + 1), key=lambda a: (self.hist[a], -a))
        self.m = best
        self.m_trace.append(best)

    def kernel_ms(self):
        t = sum(a.elapsed_time(b) for a, b in self.events)
        self.events = []
        return t


class SpecLARollback(Rollback):
    """Pure recomputation: verification writes no snapshot; after acceptance every row replays its first AL block
    tokens from the durable state S_0 (never modified by verification), reusing the verify pass's k, v, g, beta.
    Conv state = the conv-input window ending at the last committed token."""
    name = "specla"
    stores_snapshots = False

    def __init__(self, engine, bs, block):
        super().__init__(engine, bs, block)
        self.inputs, self.conv_full, self.events, self.capture = {}, {}, [], None

    def buffer_bytes(self):
        return dict(rec_bytes=0, conv_bytes=0, total_bytes=0, layers=len(self.eng.gdn), positions=0, batch=self.bs)

    def verify_gdn(self, i, q, k, v, g, beta, conv_full, n):
        e0 = self._ev()
        h0 = self.eng.rec[i][:n]
        o = gdn_verify(q, k, v, g, beta, h0, h0, store=False)     # hs unused when store=False
        self._ev_end(e0)
        self.inputs[i] = (k, v, g, beta)
        self.conv_full[i] = conv_full
        if self.capture is not None:
            self.capture[i] = (q, k, v, g, beta)
        return o

    def restore(self, produced, n, accepted=None):
        offs = torch.zeros_like(produced)
        km1 = self.eng.conv[self.eng.gdn[0]].shape[-1]
        widx = (produced[:, None] + torch.arange(km1, device=produced.device)[None])[:, None, :]
        for i, (k, v, g, beta) in self.inputs.items():
            gdn_replay(k, v, g, beta, self.eng.rec[i][:n], offs, produced)
            full = self.conv_full[i]
            self.eng.conv[i][:n].copy_(torch.gather(full, 2, widx.expand(-1, full.shape[1], -1)))
        self.inputs, self.conv_full = {}, {}

    def kernel_ms(self):
        t = sum(a.elapsed_time(b) for a, b in self.events)
        self.events = []
        return t


class ConfKRollback(Rollback):
    """modeal-style restore, but the snapshot rows are chosen per sequence from the draft's confidences, before
    verification. c_i = draft probability of its own token at block position i (i = 1..S-1).
      select "topk": P(j) = (c_1 ... c_j)(1 - c_{j+1}), j = 0..S-2 (j = accepted drafts); keep the K largest
                     (ties -> smaller j)
      select "thr" : j = number of leading drafts with c_i >= thr (consecutive from i = 1)
    Row j = state after the pending token + j drafts (block token index j). Row S-1 (all accepted) is always kept,
    so a sequence has K+1 (topk) or 2 (thr) slots. The verify kernel writes only those rows (slot map [n, S]).
    Restore with A = committed tokens (row A-1): hit -> copy the slot; miss -> copy the largest kept row r < A-1
    (or start from S_0) and replay GDN rows r+1..A-1 with the verify pass's k, v, g, beta. Conv state = conv-input
    window ending at row A-1; attention KV / draft features come from the verify pass (as modeal)."""
    name = "confk"
    needs_confidence = True

    def __init__(self, engine, bs, block, select="topk", k=1, thr=0.5, slots=None):
        super().__init__(engine, bs, block)
        assert select in ("topk", "thr")
        e = engine
        self.select, self.k, self.thr = select, k, thr
        self.ns = slots or ((k + 1) if select == "topk" else 2)
        self.snap_rec = {i: torch.empty(bs, self.ns, *e.rec[i].shape[1:], device=e.dev, dtype=e.rec[i].dtype)
                         for i in e.gdn}
        self.inputs, self.conv_full, self.events, self.capture = {}, {}, [], None
        self.slot_map = self.rows = None
        self.reset_stats()

    def reset_stats(self):
        if getattr(self, "stat", None) is not None:     # in place: a captured CUDA graph keeps adding to these tensors
            for x in self.stat.values():
                x.zero_()
            return
        z = lambda: torch.zeros((), dtype=torch.long, device=self.eng.dev)  # noqa: E731
        self.stat = dict(n=z(), hit=z(), from_s0=z(), replay=z(),
                         replay_hist=torch.zeros(self.S + 1, dtype=torch.long, device=self.eng.dev))

    def pred_stats(self):
        s = {k: v.tolist() for k, v in self.stat.items()}
        n = max(s["n"], 1)
        return dict(select=self.select, k=self.k if self.select == "topk" else None,
                    thr=self.thr if self.select == "thr" else None, slots_per_seq=self.ns, row_steps=s["n"],
                    hit_rate=s["hit"] / n, from_s0_rate=s["from_s0"] / n, replay_tokens_per_row_step=s["replay"] / n,
                    replay_hist={str(i): c for i, c in enumerate(s["replay_hist"]) if c})

    def buffer_bytes(self):
        rec = sum(x.numel() * x.element_size() for x in self.snap_rec.values())
        return dict(rec_bytes=rec, conv_bytes=0, total_bytes=rec, layers=len(self.snap_rec), positions=self.ns,
                    batch=self.bs)

    def store_kwargs(self):
        return dict(slot_map=self.slot_map)

    def predict(self, conf):
        """conf [n, S-1] -> rows [n, ns] (kept block rows; -1 = unused) and slot_map [n, S]."""
        n, S = conf.shape[0], self.S
        dev = conf.device
        if self.select == "topk":
            P = torch.cumprod(conf, 1)                                          # c_1 .. c_1...c_{S-1}
            Pa = torch.cat([torch.ones_like(P[:, :1]), P[:, :-1]], 1)           # P_0 .. P_{S-2}
            pj = Pa * (1 - conf)                                                # P(j), j = 0..S-2
            chosen = torch.argsort(-pj, dim=1, stable=True)[:, : self.k]        # ties -> smaller j
        else:
            chosen = (conf >= self.thr).long().cumprod(1).sum(1, keepdim=True)  # leading run length, 0..S-1
        rows = torch.cat([chosen, torch.full((n, 1), S - 1, device=dev, dtype=torch.long)], 1)
        rows = torch.where((rows == S - 1) & (torch.arange(rows.shape[1], device=dev) < rows.shape[1] - 1), -1, rows)
        smap = torch.full((n, S), -1, device=dev, dtype=torch.int32)
        valid = rows >= 0
        slot = torch.arange(self.ns, device=dev).expand(n, -1)
        smap[torch.arange(n, device=dev)[:, None].expand(-1, self.ns)[valid], rows[valid]] = slot[valid].int()
        self.rows, self.slot_map = rows, smap

    def verify_gdn(self, i, q, k, v, g, beta, conv_full, n):
        e0 = self._ev()
        o = gdn_verify(q, k, v, g, beta, self.eng.rec[i][:n], self.snap_rec[i][:n], store=True, slot_map=self.slot_map)
        self._ev_end(e0)
        self.inputs[i] = (k, v, g, beta)
        self.conv_full[i] = conv_full
        if self.capture is not None:
            self.capture[i] = (q, k, v, g, beta)
        return o

    def restore(self, produced, n, accepted=None):
        dev = produced.device
        last = (produced - 1)[:, None]                                          # row A-1
        cand = torch.where((self.rows >= 0) & (self.rows <= last), self.rows, -1)
        best, slot = cand.max(1)                                                # largest kept row <= A-1
        have = best >= 0
        rows_have = have.nonzero().flatten()
        flat = rows_have * self.ns + slot[rows_have]
        offs = torch.where(have, best + 1, 0)
        lens = produced - offs
        km1 = self.eng.conv[self.eng.gdn[0]].shape[-1]
        widx = (produced[:, None] + torch.arange(km1, device=dev)[None])[:, None, :]
        for i in self.snap_rec:
            rec = self.eng.rec[i][:n]
            rec.index_copy_(0, rows_have, self.snap_rec[i][:n].flatten(0, 1).index_select(0, flat))
            k, v, g, beta = self.inputs[i]
            gdn_replay(k, v, g, beta, rec, offs, lens)
            full = self.conv_full[i]
            self.eng.conv[i][:n].copy_(torch.gather(full, 2, widx.expand(-1, full.shape[1], -1)))
        self.inputs, self.conv_full = {}, {}
        self.stat["n"] += n
        self.stat["hit"] += (lens == 0).sum()
        self.stat["from_s0"] += (~have).sum()
        self.stat["replay"] += lens.sum()
        self.stat["replay_hist"].index_add_(0, lens, torch.ones_like(lens))

    def kernel_ms(self):
        t = sum(a.elapsed_time(b) for a, b in self.events)
        self.events = []
        return t


class ModeALStatic(ModeALRollback):
    """CUDA-graph version of ModeALRollback (same decisions): m and the AL histogram live on the device and are updated
    inside the captured step; the snapshot position reaches the verify kernel as a slot map; the restore copies the
    snapshot only for rows with AL >= m (conditional row copy). Before the first observation the histogram holds a
    prior of 0.5 at block//2, so m = block//2 until then; argmax keeps the first (smaller) AL on ties."""

    def __init__(self, engine, bs, block):
        dev = engine.dev
        self.m_dev = torch.full((), block // 2, dtype=torch.long, device=dev)
        self.hist_dev = torch.zeros(block + 1, device=dev)
        super().__init__(engine, bs, block)

    def reset_stats(self):
        super().reset_stats()
        self.hist_dev.zero_()
        self.hist_dev[self.S // 2] = 0.5
        self.m_dev.fill_(self.S // 2)

    def observed_hist(self):
        h = self.hist_dev.clone()
        h[self.S // 2] -= 0.5
        return [int(round(x)) for x in h.tolist()]

    def store_kwargs(self):
        return dict(slot_map=self._smap(self.eng.n))

    def _smap(self, n):
        pos = torch.arange(self.S, device=self.m_dev.device)[None]
        return torch.where(pos == self.m_dev - 1, 0, -1).to(torch.int32).expand(n, self.S).contiguous()

    def verify_gdn(self, i, q, k, v, g, beta, conv_full, n):
        e0 = self._ev()
        o = gdn_verify(q, k, v, g, beta, self.eng.rec[i][:n], self.snap_rec[i][:n].unsqueeze(1), store=True,
                       slot_map=self._smap(n))
        self._ev_end(e0)
        self.inputs[i] = (k, v, g, beta)
        self.conv_full[i] = conv_full
        if self.capture is not None:
            self.capture[i] = (q, k, v, g, beta)
        return o

    def save(self, n):
        for i, full in self.conv_full.items():
            km1 = full.shape[-1] - self.S
            idx = (self.m_dev + torch.arange(km1, device=full.device)).view(1, 1, km1).expand(n, full.shape[1], km1)
            self.snap_conv[i][:n].copy_(torch.gather(full, 2, idx))

    def restore(self, produced, n, accepted=None):
        dev = produced.device
        m = self.m_dev
        have = produced >= m
        src = torch.where(have, torch.arange(n, device=dev), -1)
        offs = torch.where(have, m, 0)
        lens = produced - offs
        km1 = self.eng.conv[self.eng.gdn[0]].shape[-1]
        widx = (produced[:, None] + torch.arange(km1, device=dev)[None])[:, None, :]
        for i in self.snap_rec:
            rec = self.eng.rec[i][:n]
            cond_copy_rows(rec, self.snap_rec[i][:n], src)
            k, v, g, beta = self.inputs[i]
            gdn_replay(k, v, g, beta, rec, offs, lens)
            full = self.conv_full[i]
            self.eng.conv[i][:n].copy_(torch.gather(full, 2, widx.expand(-1, full.shape[1], -1)))
        self.inputs, self.conv_full = {}, {}
        valid = torch.arange(n, device=dev) < self.eng.n_dev          # padded rows of a batch bucket do not count
        a = accepted if accepted is not None else produced
        self.hist_dev.index_add_(0, torch.where(valid, a, 0), valid.float())
        self.m_dev.copy_(torch.argmax(self.hist_dev[1:]) + 1)


class ConfKStatic(ConfKRollback):
    """CUDA-graph version of ConfKRollback (same selection and restore): the slot map is built with a fixed-shape
    scatter and the snapshot is gathered with a conditional row copy; statistics skip padded rows."""

    def predict(self, conf):
        n, S = conf.shape[0], self.S
        dev = conf.device
        if self.select == "topk":
            P = torch.cumprod(conf, 1)
            Pa = torch.cat([torch.ones_like(P[:, :1]), P[:, :-1]], 1)
            chosen = torch.argsort(-(Pa * (1 - conf)), dim=1, stable=True)[:, : self.k]
        else:
            chosen = (conf >= self.thr).long().cumprod(1).sum(1, keepdim=True)
        rows = torch.cat([chosen, torch.full((n, 1), S - 1, device=dev, dtype=torch.long)], 1)
        rows = torch.where((rows == S - 1) & (torch.arange(rows.shape[1], device=dev) < rows.shape[1] - 1), -1, rows)
        smap = torch.full((n, S + 1), -1, device=dev, dtype=torch.int32)      # column S collects unused slots
        slot = torch.arange(self.ns, device=dev, dtype=torch.int32).expand(n, -1)
        smap.scatter_(1, torch.where(rows >= 0, rows, S), slot)
        self.rows, self.slot_map = rows, smap[:, :S].contiguous()

    def restore(self, produced, n, accepted=None):
        dev = produced.device
        last = (produced - 1)[:, None]
        cand = torch.where((self.rows >= 0) & (self.rows <= last), self.rows, -1)
        best, slot = cand.max(1)
        have = best >= 0
        src = torch.where(have, torch.arange(n, device=dev) * self.ns + slot, -1)
        offs = torch.where(have, best + 1, 0)
        lens = produced - offs
        km1 = self.eng.conv[self.eng.gdn[0]].shape[-1]
        widx = (produced[:, None] + torch.arange(km1, device=dev)[None])[:, None, :]
        for i in self.snap_rec:
            rec = self.eng.rec[i][:n]
            cond_copy_rows(rec, self.snap_rec[i][:n].flatten(0, 1), src)
            k, v, g, beta = self.inputs[i]
            gdn_replay(k, v, g, beta, rec, offs, lens)
            full = self.conv_full[i]
            self.eng.conv[i][:n].copy_(torch.gather(full, 2, widx.expand(-1, full.shape[1], -1)))
        self.inputs, self.conv_full = {}, {}
        valid = torch.arange(n, device=dev) < self.eng.n_dev
        vl = valid.long()
        self.stat["n"] += vl.sum()
        self.stat["hit"] += ((lens == 0) & valid).sum()
        self.stat["from_s0"] += ((~have) & valid).sum()
        self.stat["replay"] += (lens * vl).sum()
        self.stat["replay_hist"].index_add_(0, lens, vl)


def load_conf_head(path, device):
    """conf-K method B head (confk-head-B): net = Linear(5120, 1) ("linear") or Linear(5120, 256) - SiLU -
    Linear(256, 1) ("mlp"), with the input normalisation mu / sd stored next to the weights."""
    d = torch.load(path, map_location=device, weights_only=True)
    dim = d["mu"].shape[0]
    if d["kind"] == "linear":
        net = torch.nn.Linear(dim, 1)
    elif d["kind"] == "mlp":
        net = torch.nn.Sequential(torch.nn.Linear(dim, 256), torch.nn.SiLU(), torch.nn.Linear(256, 1))
    else:
        raise ValueError(f"conf head must be linear or mlp, got {d['kind']}")
    net.load_state_dict(d["state"])
    return dict(kind=d["kind"], net=net.to(device).float().eval().requires_grad_(False), mu=d["mu"].float().to(device),
                sd=d["sd"].float().to(device))


class _ConfHead:
    """conf-K method B: the softmax probability c_k is replaced by a learned head on the draft side,
    c_k = sigmoid(net(([h_k ; E[prev_k]] - mu) / sd)), k = 1..S-1, where h_k is the draft's final-norm hidden at draft
    position k (the lm_head input), prev_k the verify-block token before it (the pending token for k = 1) and E the
    target's input embeddings. Selection and restore are those of conf-K."""
    name = "confkb"
    needs_confidence = "hidden"

    def __init__(self, engine, bs, block, select="topk", k=1, thr=0.5, head=None):
        super().__init__(engine, bs, block, select, k, thr)
        self.head_path = head
        self.head = load_conf_head(head, engine.dev)

    def confidence(self, hidden, prev):
        E = self.eng.t.get_input_embeddings().weight
        x = torch.cat([hidden.float(), E[prev].float()], -1)
        c = torch.sigmoid(self.head["net"]((x - self.head["mu"]) / self.head["sd"])[..., 0])
        self.last = (hidden, prev, c)       # references only (static graph tensors after capture): integration check
        return c

    def pred_stats(self):
        return dict(super().pred_stats(), conf_source="head", head=self.head_path, head_kind=self.head["kind"])


class ConfKBRollback(_ConfHead, ConfKRollback):
    pass


class ConfKBStatic(_ConfHead, ConfKStatic):
    pass


class _Oracle:
    """Upper bound for AL prediction: one snapshot slot per sequence, at the row the step will actually commit (A - 1),
    read from a recorded run of the same engine and batch size (outputs and acceptance are identical across
    strategies). load_oracle() fills the device buffer on the host before the timed step (not counted); pre_verify()
    turns it into the slot map after the draft. Restore is conf-K's, so a wrong guess still restores exactly."""
    name = "oracle"
    needs_confidence = False

    def __init__(self, engine, bs, block, trace=None):
        super().__init__(engine, bs, block, slots=1)
        self.trace_path = os.path.join(trace, "raw", f"dflash_bs{bs}_run1.jsonl")
        self.trace = {r["order"]: r["accept"] for r in map(json.loads, open(self.trace_path))}   # engine row id = order
        self.oracle_host = torch.ones(bs, dtype=torch.long).pin_memory()
        self.oracle_dev = torch.ones(bs, dtype=torch.long, device=engine.dev)

    def load_oracle(self, n):
        for r, row in enumerate(self.eng.rows[:n]):
            t, k = self.trace.get(row["qid"], ()), len(row["accept"])
            self.oracle_host[r] = t[k] if k < len(t) else 1
        self.oracle_dev[:n].copy_(self.oracle_host[:n], non_blocking=True)

    def pre_verify(self, n):
        row = (self.oracle_dev[:n] - 1).clamp(0, self.S - 1)[:, None]
        smap = torch.full((n, self.S), -1, device=row.device, dtype=torch.int32)
        smap.scatter_(1, row, torch.zeros_like(row, dtype=torch.int32))
        self.rows, self.slot_map = row, smap

    def pred_stats(self):
        return dict(super().pred_stats(), oracle_trace=self.trace_path)


class OracleRollback(_Oracle, ConfKRollback):
    pass


class OracleStatic(_Oracle, ConfKStatic):
    pass


class OracleCopyStatic(OracleStatic):
    """oracle with the lightest restore: the single snapshot slot always holds the committed row, so the restore is
    one copy of the recurrent state and one gather of the conv window per GDN layer -- no fallback replay, no slot
    search, no device statistics. Correctness relies on the oracle being right; check_oracle() compares the oracle
    with the produced counts on the host after the timed step and counts mismatches."""
    name = "oracle_copy"

    def reset_stats(self):
        super().reset_stats()
        self.n_rows = self.n_miss = 0

    def restore(self, produced, n, accepted=None):
        km1 = self.eng.conv[self.eng.gdn[0]].shape[-1]
        widx = (produced[:, None] + torch.arange(km1, device=produced.device)[None])[:, None, :]
        for i in self.snap_rec:
            self.eng.rec[i][:n].copy_(self.snap_rec[i][:n, 0])
            full = self.conv_full[i]
            torch.gather(full, 2, widx.expand(-1, full.shape[1], -1), out=self.eng.conv[i][:n])
        self.inputs, self.conv_full = {}, {}

    def check_oracle(self, prod):
        self.n_rows += len(prod)
        self.n_miss += sum(int(self.oracle_host[r]) != a for r, a in enumerate(prod))

    def pred_stats(self):
        n = max(self.n_rows, 1)
        return dict(select="oracle_copy", slots_per_seq=1, row_steps=self.n_rows, oracle_misses=self.n_miss,
                    hit_rate=1 - self.n_miss / n, from_s0_rate=0.0, replay_tokens_per_row_step=0.0,
                    oracle_trace=self.trace_path)



class SlotStoreStatic(ConfKStatic):
    """Base of the fixed-slot and oracle-slot strategies below (youngjoon, 10/24). The verify kernel writes the kept
    rows into `ns` slots per sequence -- fp32 (bits 32), or 8-bit / 4-bit (symmetric, one fp32 scale per q_group key
    rows of each value column; None = per value channel), the kept rows being read once per sequence before the token
    loop (gdn_verify_slots; no per-token slot map) -- and the restore is one fused kernel per GDN layer
    (gdn_replay_from): each row starts from its largest kept row <= A-1 (or from S_0) and replays the rest, so the
    snapshot is read once and the durable state written once (no separate copy pass)."""
    needs_confidence = False
    bits = 32
    q_group = None

    def __init__(self, engine, bs, block, slots=1):
        Rollback.__init__(self, engine, bs, block)
        e = engine
        hv, kd, vd = e.rec[e.gdn[0]].shape[1:]
        self.select, self.k, self.thr, self.ns = "slots", slots, None, slots
        if self.bits in (4, 8):
            self.group = self.q_group or kd
            dt = torch.uint8 if self.bits == 4 else torch.int16          # two key rows per element
            self.snap_rec = {i: torch.empty(bs, slots, hv, kd // 2, vd, device=e.dev, dtype=dt) for i in e.gdn}
            self.snap_scale = {i: torch.empty(bs, slots, hv, kd // self.group, vd, device=e.dev) for i in e.gdn}
        else:
            assert self.bits == 32
            self.group = None
            self.snap_rec = {i: torch.empty(bs, slots, hv, kd, vd, device=e.dev, dtype=e.rec[i].dtype) for i in e.gdn}
            self.snap_scale = None
        self.inputs, self.conv_full, self.events, self.capture = {}, {}, [], None
        self.slot_map = self.rows = None
        self.reset_stats()

    def buffer_bytes(self):
        b = super().buffer_bytes()
        sc = sum(x.numel() * x.element_size() for x in (self.snap_scale or {}).values())
        return dict(b, rec_bytes=b["rec_bytes"] + sc, total_bytes=b["total_bytes"] + sc, bits=self.bits, q_group=self.group)

    def verify_gdn(self, i, q, k, v, g, beta, conv_full, n):
        e0 = self._ev()
        if i == self.eng.gdn[0]:                                     # one int32 copy of the kept rows per step
            self.rows32 = self.rows.to(torch.int32).contiguous()
        o = gdn_verify_slots(q, k, v, g, beta, self.eng.rec[i][:n], self.snap_rec[i][:n], self.rows32,
                             self.snap_scale[i][:n] if self.snap_scale else None)
        self._ev_end(e0)
        self.inputs[i] = (k, v, g, beta)
        self.conv_full[i] = conv_full
        if self.capture is not None:
            self.capture[i] = (q, k, v, g, beta)
        return o

    def restore(self, produced, n, accepted=None):
        dev = produced.device
        last = (produced - 1)[:, None]
        cand = torch.where((self.rows >= 0) & (self.rows <= last), self.rows, -1)
        best, slot = cand.max(1)
        have = best >= 0
        src = torch.where(have, torch.arange(n, device=dev) * self.ns + slot, -1)
        offs = torch.where(have, best + 1, 0)
        lens = produced - offs
        km1 = self.eng.conv[self.eng.gdn[0]].shape[-1]
        widx = (produced[:, None] + torch.arange(km1, device=dev)[None])[:, None, :]
        for i in self.snap_rec:
            k, v, g, beta = self.inputs[i]
            gdn_replay_from(k, v, g, beta, self.eng.rec[i][:n], offs, lens, src, self.snap_rec[i][:n].flatten(0, 1),
                            self.snap_scale[i][:n].flatten(0, 1) if self.snap_scale else None)
            full = self.conv_full[i]
            torch.gather(full, 2, widx.expand(-1, full.shape[1], -1), out=self.eng.conv[i][:n])
        self.inputs, self.conv_full = {}, {}
        valid = torch.arange(n, device=dev) < self.eng.n_dev
        vl = valid.long()
        self.stat["n"] += vl.sum()
        self.stat["hit"] += ((lens == 0) & valid).sum()
        self.stat["from_s0"] += ((~have) & valid).sum()
        self.stat["replay"] += (lens * vl).sum()
        self.stat["replay_hist"].index_add_(0, lens, vl)

    def pred_stats(self):
        return dict(super().pred_stats(), bits=self.bits, q_group=self.group)


class FixedSlotsStatic(SlotStoreStatic):
    """Snapshots at fixed acceptance lengths (als=(6,): the state after 6 committed tokens), the same for every row
    and step: no position prediction. A < smallest kept AL: replay from S_0."""
    name = "fixed"

    def __init__(self, engine, bs, block, als=(6,)):
        als = sorted({int(a) for a in als})
        assert all(1 <= a <= block for a in als), als
        super().__init__(engine, bs, block, slots=len(als))
        dev = engine.dev
        rows = torch.tensor([a - 1 for a in als], device=dev)
        smap = torch.full((block,), -1, dtype=torch.int32, device=dev)
        smap[rows] = torch.arange(self.ns, dtype=torch.int32, device=dev)
        self.rows_all = rows.expand(bs, -1).contiguous()
        self.smap_all = smap.expand(bs, -1).contiguous()
        self.als = als

    def pre_verify(self, n):
        self.rows, self.slot_map = self.rows_all[:n], self.smap_all[:n]

    def pred_stats(self):
        return dict(super().pred_stats(), als=self.als)


class FixedSlotsQ4(FixedSlotsStatic):
    name = "fixed_q4"
    bits = 4


class FixedSlotsQ8(FixedSlotsStatic):
    name = "fixed_q8"
    bits = 8


class OracleSlotStatic(_Oracle, SlotStoreStatic):
    """oracle with the fused restore; a wrong oracle (e.g. once 4-bit snapshots changed the output and the recorded
    trace no longer applies) still restores exactly from S_0 (counted in hit_rate). check_oracle() keeps, per step,
    whether each row's oracle was right (host side, after the timed step), so step times can be split by hit."""
    name = "oracle_fused"

    def reset_stats(self):
        super().reset_stats()
        self.hit_trace = []

    def check_oracle(self, prod):
        self.hit_trace.append([int(self.oracle_host[r]) == a for r, a in enumerate(prod)])

    def pred_stats(self):
        return dict(super().pred_stats(), oracle_hit_trace=self.hit_trace)


class OracleSlotQ4(OracleSlotStatic):
    name = "oracle_fused_q4"
    bits = 4


class OracleSlotQ8(OracleSlotStatic):
    name = "oracle_fused_q8"
    bits = 8



class ExpCostSlotStatic(SlotStoreStatic):
    """Our position-prediction methods on 8-bit snapshots (youngjoon, 10/24). From a predicted distribution of the
    acceptance length P(A), A = 1..S, each sequence keeps the one block row r that minimises the expected rollback
    cost, or none when recomputation (SpecLA) is expected to be cheaper:
        E[none] = sum_A P(A) rep(A)
        E[r]    = W + sum_{A <= r} P(A) rep(A) + sum_{A > r} P(A) fused(A - 1 - r)
    rep(n): replay n tokens from S_0; fused(m): read the 8-bit snapshot + replay m tokens; W: the snapshot write in
    verify. GPU kernel times per GDN layer from q4/test_q4.py --full --json (closest measured batch size)."""
    bits = 8

    def __init__(self, engine, bs, block, cost_table=None):
        super().__init__(engine, bs, block, slots=1)
        tabs = json.load(open(cost_table))
        t = tabs[min(tabs, key=lambda b: abs(int(b) - bs))]
        S = block
        rep = [t[f"replay {a} tok from S0"] for a in range(1, S + 1)]
        kind = "fp32" if self.bits == 32 else f"{self.bits}bit"
        fus = [t[f"fused {kind} snap + {m} tok"] for m in range(S)]
        w = sum(t[f"verify_slots {kind} (AL {a})"] for a in (2, 6, 12)) / 3 - t["verify, no store"]
        C = torch.empty(S, S + 1)                           # [A - 1, choice]; choice 0 = no snapshot, r + 1 = row r
        for A in range(1, S + 1):
            C[A - 1, 0] = rep[A - 1]
            for r in range(S):
                C[A - 1, r + 1] = w + (rep[A - 1] if A <= r else fus[A - 1 - r])
        self.cost = C.to(engine.dev)
        self.cost_table, self.cost_batch = cost_table, int(min(tabs, key=lambda b: abs(int(b) - bs)))
        z = lambda: torch.zeros((), dtype=torch.long, device=engine.dev)  # noqa: E731
        self.stat2 = dict(none=z(), rowsum=z(), n=z())

    def reset_stats(self):
        super().reset_stats()
        for x in getattr(self, "stat2", {}).values():
            x.zero_()

    def choose(self, P):
        """P [n, S] over A = 1..S -> self.rows [n, 1] (kept block row, -1 = none)."""
        ch = (P @ self.cost).argmin(1)
        self.rows = (ch - 1)[:, None]
        self.slot_map = None
        n = P.shape[0]
        valid = (torch.arange(n, device=P.device) < self.eng.n_dev).long()
        self.stat2["n"] += valid.sum()
        self.stat2["none"] += ((ch == 0).long() * valid).sum()
        self.stat2["rowsum"] += ((ch - 1).clamp(min=0) * valid).sum()

    @staticmethod
    def al_dist(conf):
        """Draft confidences c_1..c_{S-1} [n, S-1] -> P(A) [n, S]: P(A >= a) = c_1 ... c_{a-1}."""
        surv = torch.cat([torch.ones_like(conf[:, :1]), torch.cumprod(conf, 1)], 1)       # P(A >= 1..S)
        return surv - torch.cat([surv[:, 1:], torch.zeros_like(surv[:, :1])], 1)

    def pred_stats(self):
        st = {k: int(v) for k, v in self.stat2.items()}
        m = max(st["n"], 1)
        return dict(super().pred_stats(), cost_table=self.cost_table, cost_batch=self.cost_batch,
                    no_snapshot_rate=st["none"] / m, mean_kept_row=st["rowsum"] / max(st["n"] - st["none"], 1))


class HistCostStatic(ExpCostSlotStatic):
    """Statistics method (most-frequent-AL family): P(A) = online histogram of the observed acceptance lengths
    (device, updated inside the captured step; uniform prior of one count per A), the same row for every sequence.
    (Named al_counts: run_all treats an attribute "hist" as modeal's.)"""
    name = "hist_q8"
    needs_confidence = False

    def __init__(self, engine, bs, block, cost_table=None):
        super().__init__(engine, bs, block, cost_table)
        self.al_counts = torch.ones(block, device=engine.dev)

    def reset_stats(self):
        super().reset_stats()
        if hasattr(self, "al_counts"):
            self.al_counts.fill_(1.0)

    def pre_verify(self, n):
        self.choose((self.al_counts / self.al_counts.sum())[None].expand(n, -1))

    def restore(self, produced, n, accepted=None):
        super().restore(produced, n, accepted)
        valid = (torch.arange(n, device=produced.device) < self.eng.n_dev).float()
        a = accepted if accepted is not None else produced
        self.al_counts.index_add_(0, (a - 1).clamp(0, self.S - 1), valid)


class SoftmaxCostStatic(ExpCostSlotStatic):
    """softmax method: c_k = draft softmax probability of its own token, computed with the argmax in one pass over
    the logits (engine need_conf "fused"), P(A) from c_1..c_{S-1}."""
    name = "confsm_q8"
    needs_confidence = "fused"

    def predict(self, conf):
        self.choose(self.al_dist(conf.float()))


class HeadCostStatic(ExpCostSlotStatic):
    """learned-head method: c_k from the conf-K method B head on the draft hidden state, P(A) from c_1..c_{S-1}."""
    name = "confhead_q8"
    needs_confidence = "hidden"
    confidence = _ConfHead.confidence

    def __init__(self, engine, bs, block, cost_table=None, head=None):
        super().__init__(engine, bs, block, cost_table)
        self.head_path = head
        self.head = load_conf_head(head, engine.dev)

    def predict(self, conf):
        self.choose(self.al_dist(conf.float()))


class _Deferred(SlotStoreStatic):
    """Deferred restore (youngjoon, 10/24): restore() only records, per row, where the committed state starts (its
    kept snapshot slot, or S_0 = engine.rec) and which block tokens remain to be replayed; the next step's verify kernel
    (gdn_verify_deferred) does that replay in registers, writes the committed state to engine.rec once and goes on
    with the new block -- no separate restore pass over the state. The block's raw k, v, g, beta are kept by the kernel
    in a two-half buffer (half par this step, read next step). The per-row buffers join the engine's row tensors, so a
    retired row's slot is refilled together with them; rows inserted since the last step (ctx_cnt == 0) skip the
    pending replay. One snapshot slot per row. committed_bits 8 / 4: the committed state itself is kept quantised (per
    channel) instead of fp32 -- smaller writes and reads every step, a quantisation in every round."""
    committed_bits = 0

    def __init__(self, engine, bs, block, slots=1):
        super().__init__(engine, bs, block, slots=slots)
        assert slots == 1 and self.bits in (32, 8, 4)
        e, dev = engine, engine.dev
        hv, kd, vd = e.rec[e.gdn[0]].shape[1:]
        L, bf = len(e.gdn), torch.bfloat16
        # one buffer per kind for all GDN layers (row-major in the batch row), so retiring a row moves 9 tensors
        snap = torch.empty(bs, L, *self.snap_rec[e.gdn[0]].shape[1:], device=dev, dtype=self.snap_rec[e.gdn[0]].dtype)
        self.snap_rec = {i: snap[:, li] for li, i in enumerate(e.gdn)}
        self.bufs = [snap]
        if self.snap_scale:
            sc = torch.empty(bs, L, *self.snap_scale[e.gdn[0]].shape[1:], device=dev)
            self.snap_scale = {i: sc[:, li] for li, i in enumerate(e.gdn)}
            self.bufs.append(sc)
        pk = torch.zeros(bs, L, 2, block, hv, kd, device=dev, dtype=bf)
        pv = torch.zeros(bs, L, 2, block, hv, vd, device=dev, dtype=bf)
        pg = torch.zeros(bs, L, 2, block, hv, device=dev)
        pb = torch.zeros(bs, L, 2, block, hv, device=dev, dtype=bf)
        self.pk, self.pv, self.pg, self.pb = ({i: x[:, li] for li, i in enumerate(e.gdn)} for x in (pk, pv, pg, pb))
        self.pslot = torch.full((bs,), -1, dtype=torch.int32, device=dev)
        self.poffs = torch.zeros(bs, dtype=torch.int32, device=dev)
        self.plens = torch.zeros(bs, dtype=torch.int32, device=dev)
        self.par = torch.zeros((), dtype=torch.int32, device=dev)
        self.bufs += [pk, pv, pg, pb, self.pslot, self.poffs, self.plens]
        self.cq = self.cqs = None
        if self.committed_bits:                                      # quantised committed state instead of engine.rec
            dt = torch.int16 if self.committed_bits == 8 else torch.uint8
            cq = torch.empty(bs, L, hv, kd // 2, vd, device=dev, dtype=dt)
            cqs = torch.empty(bs, L, hv, 1, vd, device=dev)
            self.cq, self.cqs = ({i: x[:, li] for li, i in enumerate(e.gdn)} for x in (cq, cqs))
            self.bufs += [cq, cqs]
        rows_of = e._row_tensors
        e._row_tensors = lambda: list(rows_of()) + self.bufs

    def verify_gdn(self, i, q, k, v, g, beta, conv_full, n):
        if i == self.eng.gdn[0]:                                     # once per step
            self.rows32 = self.rows[:, 0].to(torch.int32).contiguous()
            self.fresh32 = (self.eng.ctx_cnt[:n] == 0).to(torch.int32)
        o = gdn_verify_deferred(q, k, v, g, beta, self.eng.rec[i][:n], self.snap_rec[i][:n], self.rows32,
                                self.pk[i][:n], self.pv[i][:n], self.pg[i][:n], self.pb[i][:n], self.par,
                                self.pslot[:n], self.poffs[:n], self.plens[:n], self.fresh32,
                                self.snap_scale[i][:n] if self.snap_scale else None,
                                self.cq[i][:n] if self.cq else None, self.cqs[i][:n] if self.cq else None)
        self.conv_full[i] = conv_full
        if self.capture is not None:
            self.capture[i] = (q, k, v, g, beta)
        return o

    def restore(self, produced, n, accepted=None):
        dev = produced.device
        last = (produced - 1)[:, None]
        cand = torch.where((self.rows >= 0) & (self.rows <= last), self.rows, -1)
        best, slot = cand.max(1)
        have = best >= 0
        offs = torch.where(have, best + 1, 0)
        lens = produced - offs
        self.pslot[:n].copy_(torch.where(have, slot, -1))
        self.poffs[:n].copy_(offs)
        self.plens[:n].copy_(lens)
        km1 = self.eng.conv[self.eng.gdn[0]].shape[-1]
        widx = (produced[:, None] + torch.arange(km1, device=dev)[None])[:, None, :]
        for i, full in self.conv_full.items():
            torch.gather(full, 2, widx.expand(-1, full.shape[1], -1), out=self.eng.conv[i][:n])
        self.conv_full = {}
        self.par.copy_(1 - self.par)
        valid = torch.arange(n, device=dev) < self.eng.n_dev
        vl = valid.long()
        self.stat["n"] += vl.sum()
        self.stat["hit"] += ((lens == 0) & valid).sum()
        self.stat["from_s0"] += ((~have) & valid).sum()
        self.stat["replay"] += (lens * vl).sum()
        self.stat["replay_hist"].index_add_(0, lens, vl)

    def pred_stats(self):
        return dict(super().pred_stats(), deferred_restore=True, committed_bits=self.committed_bits)


class HistCostDeferred(HistCostStatic, _Deferred):
    name = "hist_dq8"


class SoftmaxCostDeferred(SoftmaxCostStatic, _Deferred):
    name = "confsm_dq8"


class HeadCostDeferred(HeadCostStatic, _Deferred):
    name = "confhead_dq8"


class OracleDeferredQ8(OracleSlotStatic, _Deferred):
    name = "oracle_dq8"
    bits = 8


class HistCostDeferred4(HistCostDeferred):
    name = "hist_dq4"
    bits = 4


class SoftmaxCostDeferred4(SoftmaxCostDeferred):
    name = "confsm_dq4"
    bits = 4


class HeadCostDeferred4(HeadCostDeferred):
    name = "confhead_dq4"
    bits = 4


class OracleDeferredQ4(OracleSlotStatic, _Deferred):
    name = "oracle_dq4"
    bits = 4


class OracleDeferredQ8C(OracleDeferredQ8):
    name = "oracle_dq8c"
    committed_bits = 8


class SoftmaxCostDeferredC(SoftmaxCostDeferred):
    name = "confsm_dq8c"
    committed_bits = 8


class HistCostDeferredC(HistCostDeferred):
    name = "hist_dq8c"
    committed_bits = 8


class OracleDeferredQ4C(OracleDeferredQ4):
    name = "oracle_dq4c"
    committed_bits = 4


class SoftmaxCostDeferred4C(SoftmaxCostDeferred4):
    name = "confsm_dq4c"
    committed_bits = 4


class HistCostDeferred4C(HistCostDeferred4):
    name = "hist_dq4c"
    committed_bits = 4


class SoftmaxCostDeferred32(SoftmaxCostDeferred):
    name = "confsm_d32"
    bits = 32


class OracleDeferred32(OracleSlotStatic, _Deferred):
    """fp32 snapshots: outputs must equal SpecLA's token for token (correctness check of the deferral)."""
    name = "oracle_d32"

ROLLBACKS = {"fullstate": FullStateRollback, "modeal": ModeALRollback, "specla": SpecLARollback,
             "confk": ConfKRollback, "confkb": ConfKBRollback, "oracle": OracleRollback}
STATIC_ROLLBACKS = {"fullstate": FullStateRollback, "modeal": ModeALStatic, "specla": SpecLARollback,
                    "confk": ConfKStatic, "confkb": ConfKBStatic, "oracle": OracleStatic,
                    "oracle_copy": OracleCopyStatic, "fixed": FixedSlotsStatic, "fixed_q4": FixedSlotsQ4,
                    "oracle_fused": OracleSlotStatic, "oracle_fused_q4": OracleSlotQ4, "fixed_q8": FixedSlotsQ8,
                    "oracle_fused_q8": OracleSlotQ8, "hist_q8": HistCostStatic, "confsm_q8": SoftmaxCostStatic,
                    "confhead_q8": HeadCostStatic, "hist_dq8": HistCostDeferred, "confsm_dq8": SoftmaxCostDeferred,
                    "confhead_dq8": HeadCostDeferred, "oracle_dq8": OracleDeferredQ8,
                    "oracle_d32": OracleDeferred32, "hist_dq4": HistCostDeferred4, "confsm_dq4": SoftmaxCostDeferred4,
                    "confhead_dq4": HeadCostDeferred4, "oracle_dq4": OracleDeferredQ4, "oracle_dq8c": OracleDeferredQ8C,
                    "confsm_dq8c": SoftmaxCostDeferredC, "hist_dq8c": HistCostDeferredC,
                    "confsm_d32": SoftmaxCostDeferred32, "oracle_dq4c": OracleDeferredQ4C,
                    "confsm_dq4c": SoftmaxCostDeferred4C, "hist_dq4c": HistCostDeferred4C}  # CUDA-graph capturable


def make_rollback(name, engine, bs, block, static=False, **kwargs):
    table = STATIC_ROLLBACKS if static else ROLLBACKS
    if name not in table:
        raise ValueError(f"unknown rollback {name!r}; available: {sorted(table)}")
    return table[name](engine, bs, block, **kwargs)
