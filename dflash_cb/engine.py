"""Continuous-batching greedy decoding engine for hybrid Qwen3.5 targets: autoregressive or DFlash.

Rows 0..n-1 of every buffer are the active sequences. A finished sequence is removed by moving the last active row
into its slot; a new sequence is prefilled (stock HF forward, timed separately) and appended at row n.
  attention layers : KV buffers [bs, Hkv, Lmax, D] written at absolute positions; `start[r]` = tokens committed.
                     Entries at >= start (rejected draft positions) are masked and overwritten later, so KV
                     truncation after acceptance is a length update.
  GDN layers       : durable conv state [bs, C, k-1] and recurrent state [bs, HV, K, V] (fp32). AR decoding updates
                     them in place; DFlash verification only reads them and the rollback module restores them.
  draft            : context-KV buffers indexed by absolute position, same `start` as the target.
"""
import time

import torch
import torch.nn.functional as F
from dflash.model import extract_context_feature
from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
from transformers import DynamicCache
from transformers.models.qwen3_5 import modeling_qwen3_5 as M

from .kernels import argmax_maxprob
from .rollback import make_rollback


def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()


def _rope(x, cos, sin):
    """x [B, H, L, D]; cos/sin [B, L, rot]. Rotates the first `rot` dims (as HF apply_rotary_pos_emb)."""
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    r = cos.shape[-1]
    xr, xp = x[..., :r], x[..., r:]
    return torch.cat([xr * cos + M.rotate_half(xr) * sin, xp], dim=-1)


class Engine:
    def __init__(self, target, draft, method, bs, max_prompt, max_new, block=16, rollback="fullstate",
                 rb_kwargs=None, static_rollback=False, ctx_cols=None):
        self.t, self.d = target, draft
        self.method, self.dflash = method, method == "dflash"
        self.bs, self.max_new = bs, max_new
        self.S = block if self.dflash else 1
        c = self.cfg = target.config
        self.dev = target.device
        self.layers = target.model.layers
        self.gdn = [i for i, l in enumerate(self.layers) if l.block_type == "linear_attention"]
        self.att = [i for i, l in enumerate(self.layers) if l.block_type == "full_attention"]
        self.feat_ids = set(draft.target_layer_ids)
        self.max_len = max_prompt + max_new + 2 * block + 1
        L, dev, bf = self.max_len, self.dev, torch.bfloat16
        self.K = {i: torch.zeros(bs, c.num_key_value_heads, L, c.head_dim, device=dev, dtype=bf) for i in self.att}
        self.V = {i: torch.zeros_like(self.K[i]) for i in self.att}
        conv_dim = c.linear_key_head_dim * c.linear_num_key_heads * 2 + c.linear_value_head_dim * c.linear_num_value_heads
        self.conv = {i: torch.zeros(bs, conv_dim, c.linear_conv_kernel_dim - 1, device=dev, dtype=bf) for i in self.gdn}
        self.rec = {i: torch.zeros(bs, c.linear_num_value_heads, c.linear_key_head_dim, c.linear_value_head_dim,
                                   device=dev, dtype=torch.float32) for i in self.gdn}
        self.out = torch.zeros(bs, L, dtype=torch.long, device=dev)
        self.start = torch.zeros(bs, dtype=torch.long, device=dev)
        self.n_prompt = torch.zeros_like(self.start)
        self.max_total = torch.zeros_like(self.start)
        self.DK, self.DV, self.rb = {}, {}, None
        if self.dflash:
            dc = draft.config
            self.DK = {j: torch.zeros(bs, dc.num_key_value_heads, L, dc.head_dim, device=dev, dtype=bf)
                       for j in range(len(draft.layers))}
            self.DV = {j: torch.zeros_like(self.DK[j]) for j in self.DK}
            feat_dim = len(draft.target_layer_ids) * c.hidden_size
            self.ctx_feat = torch.zeros(bs, ctx_cols or max(max_prompt, block), feat_dim, device=dev, dtype=bf)
            self.ctx_cnt = torch.zeros_like(self.start)
            self.rb = make_rollback(rollback, self, bs, block, static=static_rollback, **(rb_kwargs or {}))
        self.n = 0
        self.rows = []

    # ------------------------------------------------------------ slots
    def _row_tensors(self):
        ts = [self.out, self.start, self.n_prompt, self.max_total]
        for dct in (self.K, self.V, self.conv, self.rec, self.DK, self.DV):
            ts += list(dct.values())
        if self.dflash:
            ts += [self.ctx_feat, self.ctx_cnt]
        return ts

    @torch.inference_mode()
    def insert(self, qid, ids):
        """Prefill one prompt (1-D LongTensor) into row n. Returns prefill seconds."""
        r, n = self.n, ids.shape[0]
        t0 = sync_time()
        cache = DynamicCache(config=self.cfg)
        out = self.t(ids[None].to(self.dev), past_key_values=cache, use_cache=True, logits_to_keep=1,
                     output_hidden_states=self.dflash)
        for i in self.att:
            self.K[i][r, :, :n] = cache.layers[i].keys[0]
            self.V[i][r, :, :n] = cache.layers[i].values[0]
        for i in self.gdn:
            self.conv[i][r] = cache.layers[i].conv_states[0][0, :, -self.conv[i].shape[-1]:]
            self.rec[i][r] = cache.layers[i].recurrent_states[0][0]
        self.out[r, :n] = ids.to(self.dev)
        self.out[r, n] = out.logits[0, -1].argmax()
        self.start[r] = n
        self.n_prompt[r] = n
        self.max_total[r] = n + self.max_new
        if self.dflash:
            self.ctx_feat[r, :n] = extract_context_feature(out.hidden_states, self.d.target_layer_ids)[0]
            self.ctx_cnt[r] = n
        del cache, out
        t1 = sync_time()
        self.rows.append(dict(qid=qid, prompt_len=n, t_decode=0.0, accept=[]))
        self.n += 1
        return t1 - t0

    def remove(self, r):
        """Retire row r (move the last active row into it). Returns its record with the generated ids."""
        last = self.n - 1
        rec = self.rows[r]
        rec["output"] = self.out[r, self.n_prompt[r].item(): self.start[r].item() + 1].tolist()
        if r != last:
            for x in self._row_tensors():
                x[r] = x[last]
            self.rows[r] = self.rows[last]
        self.rows.pop()
        self.n -= 1
        return rec

    def reset(self):
        self.n, self.rows = 0, []

    # ------------------------------------------------------------ target forward
    def _gdn(self, i, m, x, verify, n):
        B, T, _ = x.shape
        mixed = m.in_proj_qkv(x).transpose(1, 2)
        z = m.in_proj_z(x).reshape(B, T, -1, m.head_v_dim)
        beta = m.in_proj_b(x).sigmoid()
        g = -m.A_log.float().exp() * F.softplus(m.in_proj_a(x).float() + m.dt_bias)
        full = torch.cat([self.conv[i][:n], mixed], dim=-1)                # [n, C, k-1+T] conv inputs
        co = F.silu(F.conv1d(full.to(m.conv1d.weight.dtype), m.conv1d.weight, None, groups=full.shape[1]))
        q, k, v = torch.split(co.transpose(1, 2), [m.key_dim, m.key_dim, m.value_dim], dim=-1)
        q = q.reshape(B, T, -1, m.head_k_dim)
        k = k.reshape(B, T, -1, m.head_k_dim)
        v = v.reshape(B, T, -1, m.head_v_dim)
        rep = m.num_v_heads // m.num_k_heads
        if rep > 1:
            q, k = q.repeat_interleave(rep, dim=2), k.repeat_interleave(rep, dim=2)
        if verify:
            o = self.rb.verify_gdn(i, q, k, v, g, beta, full, n)
        else:
            o, h_new = fused_recurrent_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=self.rec[i][:n],
                                                        output_final_state=True, use_qk_l2norm_in_kernel=True)
            self.rec[i][:n].copy_(h_new)
            self.conv[i][:n].copy_(full[..., -self.conv[i].shape[-1]:])
        o = m.norm(o.reshape(-1, m.head_v_dim), z.reshape(-1, m.head_v_dim)).reshape(B, T, -1)
        return m.out_proj(o)

    def _attn(self, i, a, x, pos, cos, sin, L, n):
        B, T, _ = x.shape
        hd = a.head_dim
        q, gate = torch.chunk(a.q_proj(x).view(B, T, -1, hd * 2), 2, dim=-1)
        gate = gate.reshape(B, T, -1)
        q = a.q_norm(q).transpose(1, 2)
        k = a.k_norm(a.k_proj(x).view(B, T, -1, hd)).transpose(1, 2)
        v = a.v_proj(x).view(B, T, -1, hd).transpose(1, 2)
        q, k = _rope(q, cos, sin), _rope(k, cos, sin)
        bi = torch.arange(B, device=x.device)[:, None]
        self.K[i][bi, :, pos] = k.transpose(1, 2)
        self.V[i][bi, :, pos] = v.transpose(1, 2)
        mask = torch.arange(L, device=x.device)[None, None, :] <= pos[:, :, None]          # [B, T, L]
        o = F.scaled_dot_product_attention(q, self.K[i][:n, :, :L], self.V[i][:n, :, :L], attn_mask=mask[:, None],
                                           scale=a.scaling, enable_gqa=True)
        o = o.transpose(1, 2).reshape(B, T, -1) * torch.sigmoid(gate)
        return a.o_proj(o)

    def target_forward(self, ids, pos, verify, L=None):
        """ids, pos: [n, T]. Returns greedy tokens [n, T] and (verify) draft features [n, T, F].
        L: attention length (default: max position + 1; a fixed L keeps shapes static for CUDA graphs)."""
        n = ids.shape[0]
        h = self.t.model.embed_tokens(ids)
        cos, sin = self.t.model.rotary_emb(h, pos[None].expand(3, -1, -1))
        L = L or int(pos.max()) + 1
        feats = []
        for i, layer in enumerate(self.layers):
            r = h
            x = layer.input_layernorm(h)
            if layer.block_type == "linear_attention":
                x = self._gdn(i, layer.linear_attn, x, verify, n)
            else:
                x = self._attn(i, layer.self_attn, x, pos, cos, sin, L, n)
            h = r + x
            h = h + layer.mlp(layer.post_attention_layernorm(h))
            if verify and i in self.feat_ids:
                feats.append(h)
        tokens = self.t.lm_head(self.t.model.norm(h)).argmax(-1)
        return tokens, (torch.cat(feats, -1) if verify else None)

    # ------------------------------------------------------------ draft forward
    def draft_forward(self, ctx_feat, ctx_cnt, start, block_ids, need_conf=False, Lc=None):
        """ctx_feat [n, C, F]: target features of the newest context tokens (first ctx_cnt valid) at positions
        [start - ctx_cnt, start). block_ids [n, S] at positions [start, start + S). Returns draft tokens [n, S-1] and,
        if need_conf, the draft probability of each of them [n, S-1] (softmax over the vocabulary)."""
        d = self.d
        B, C, _ = ctx_feat.shape
        S = block_ids.shape[1]
        dev = self.dev
        th = d.hidden_norm(d.fc(ctx_feat))
        x = F.embedding(block_ids, self.t.get_input_embeddings().weight)
        pos_c = (start - ctx_cnt)[:, None] + torch.arange(C, device=dev)[None]
        pos_b = start[:, None] + torch.arange(S, device=dev)[None]
        cos_c, sin_c = d.rotary_emb(x, pos_c)
        cos_b, sin_b = d.rotary_emb(x, pos_b)
        Lc = Lc or int(start.max())
        key_pos = torch.arange(Lc, device=dev)[None, None, :]
        valid_c = (key_pos < start[:, None, None]).expand(B, S, Lc)               # committed context only
        near = pos_b[:, :, None] - key_pos
        tri = torch.ones(S, S, dtype=torch.bool, device=dev).tril()[None].expand(B, S, S)
        full_blk = torch.ones(B, S, S, dtype=torch.bool, device=dev)
        bi = torch.arange(B, device=dev)[:, None]
        pos_w = pos_c.clamp(max=self.max_len - 1)  # padded entries land at >= start: masked now, overwritten later
        for li, layer in enumerate(d.layers):
            a = layer.self_attn
            hd = a.head_dim
            hs = layer.input_layernorm(x)
            q = a.q_norm(a.q_proj(hs).view(B, S, -1, hd)).transpose(1, 2)
            kc = a.k_norm(a.k_proj(th).view(B, C, -1, hd)).transpose(1, 2)
            vc = a.v_proj(th).view(B, C, -1, hd).transpose(1, 2)
            kb = a.k_norm(a.k_proj(hs).view(B, S, -1, hd)).transpose(1, 2)
            vb = a.v_proj(hs).view(B, S, -1, hd).transpose(1, 2)
            q, kb, kc = _rope(q, cos_b, sin_b), _rope(kb, cos_b, sin_b), _rope(kc, cos_c, sin_c)
            self.DK[li][bi, :, pos_w] = kc.transpose(1, 2)
            self.DV[li][bi, :, pos_w] = vc.transpose(1, 2)
            mc = valid_c if a.sliding_window is None else valid_c & (near < a.sliding_window)
            mask = torch.cat([mc, tri if a.is_causal else full_blk], dim=-1)[:, None]
            keys = torch.cat([self.DK[li][:B, :, :Lc], kb], dim=2)
            vals = torch.cat([self.DV[li][:B, :, :Lc], vb], dim=2)
            o = F.scaled_dot_product_attention(q, keys, vals, attn_mask=mask, scale=a.scaling, enable_gqa=True)
            x = x + a.o_proj(o.transpose(1, 2).reshape(B, S, -1))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        hn = d.norm(x)[:, 1:]
        logits = d.compute_logits(hn, self.t.lm_head)
        if need_conf == "fused":                        # argmax and its softmax probability in one pass (youngjoon)
            tok, conf = argmax_maxprob(logits.flatten(0, 1))
            return tok.view(B, S - 1), conf.view(B, S - 1)
        tok = logits.argmax(-1)
        if not need_conf:
            return tok, None
        if need_conf == "hidden":                       # conf-K method B: the head reads the lm_head input
            return tok, hn
        conf = torch.empty(B, S - 1, device=dev)
        for a in range(0, B, 8):                       # chunked fp32 softmax keeps the extra memory small
            lg = logits[a:a + 8].float()
            conf[a:a + 8] = (lg.amax(-1) - lg.logsumexp(-1)).exp()
        return tok, conf

    # ------------------------------------------------------------ one decode step over all active rows
    @torch.inference_mode()
    def step(self):
        """Returns (section times dict, produced per row list, finished row indices, fused-kernel GPU ms)."""
        n, S, dev = self.n, self.S, self.dev
        jS = torch.arange(S, device=dev)[None]
        start = self.start[:n]
        pos = start[:, None] + jS
        block = self.out[:n].gather(1, pos)                     # [pending token, draft slots...]
        if self.dflash:
            block[:, 1:] = self.d.mask_token_id                 # draft input: pending token + mask tokens
        T = {}
        if self.dflash and hasattr(self.rb, "load_oracle"):
            self.rb.load_oracle(n)                      # oracle AL from a recorded run, outside the timed step
        t0 = sync_time()
        if self.dflash:
            C = int(self.ctx_cnt[:n].max())
            need_conf = getattr(self.rb, "needs_confidence", False)
            block[:, 1:], conf = self.draft_forward(self.ctx_feat[:n, :C], self.ctx_cnt[:n], start, block, need_conf)
            if need_conf == "hidden":
                conf = self.rb.confidence(conf, block[:, :-1])
            if need_conf:
                self.rb.predict(conf)                   # part of the draft section
            if hasattr(self.rb, "pre_verify"):
                self.rb.pre_verify(n)
        t1 = sync_time()
        post, f_new = self.target_forward(block, pos, verify=self.dflash)
        t2 = sync_time()
        if self.dflash:
            self.rb.save(n)
        t3 = sync_time()
        bi = torch.arange(n, device=dev)
        acc = (block[:, 1:] == post[:, :-1]).long().cumprod(1).sum(1) if self.dflash else torch.zeros_like(start)
        bonus = post[bi, acc]
        # committed this step: pending token + acc accepted drafts; new tokens start+1..: drafts then bonus
        produced = torch.minimum(acc + 1, self.max_total[:n] - start - 1)
        new = torch.where(jS < acc[:, None], F.pad(block[:, 1:], (0, 1)), bonus[:, None])
        wmask = jS < produced[:, None]
        self.out[bi[:, None].expand(-1, S)[wmask], (start[:, None] + 1 + jS)[wmask]] = new[wmask]
        t4 = sync_time()
        if self.dflash:
            self.rb.restore(produced, n, acc + 1)
        t5 = sync_time()
        self.start[:n] += produced                              # KV truncate (target + draft context)
        if self.dflash:
            self.ctx_feat[:n, :S] = f_new
            self.ctx_cnt[:n] = produced
        t6 = sync_time()
        T.update(draft=t1 - t0, verify=t2 - t1, save=t3 - t2, accept=t4 - t3, restore=t5 - t4, kv_truncate=t6 - t5)
        kern = self.rb.kernel_ms() if self.dflash else 0.0
        prod = produced.tolist()
        step_s = t6 - t0
        for r in range(n):
            self.rows[r]["t_decode"] += step_s
            self.rows[r]["accept"].append(prod[r])
        finished = (self.start[:n] + 1 >= self.max_total[:n]).nonzero().flatten().tolist()
        return T, prod, finished, kern
