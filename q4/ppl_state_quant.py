"""Perplexity of Qwen3.5-4B when its GDN recurrent state is quantised and dequantised every `every` tokens (youngjoon,
10/25): symmetric, one scale per value channel (absmax over the key rows), 8 bit (-127..127) or 4 bit (-7..7) -- the
same quantiser as the snapshot kernels. The snapshot methods put a quantised state into the recurrence about once
per DFlash round (mean acceptance length about 6), so every = 6 approximates them and every = 2 is a stress case.
Teacher-forced over WikiText-2 test (GSM8K test text when offline), sequences of L tokens in batches, fed `every`
tokens at a time through the engine's AR target forward (the HF forward does not carry the GDN state across multi-token
chunks); reference = the same chunking without quantisation. Usage:
  python q4/ppl_state_quant.py --every 6,2 --bits 8,4 --seqs 128 --len 1024 --out ppl.json"""
import argparse
import json
import math
import os

import sys
import types

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dflash_cb.engine import Engine  # noqa: E402


def qdq(x, bits):
    """quantise + dequantise a [..., K, V] state per value channel (fp32 math, as the kernels)."""
    qmax = 127.0 if bits == 8 else 7.0
    xf = x.float()
    s = xf.abs().amax(-2, keepdim=True).clamp_min(1e-30) / qmax
    return (torch.floor(xf / s + 0.5).clamp(-qmax, qmax) * s).to(x.dtype)


@torch.inference_mode()
def nll(eng, ids, every, bits):
    """Sum of next-token NLL and count over a batch [B, L], fed `every` tokens at a time through the engine's AR target
    forward (GDN layers run fused_recurrent from engine.rec, which carries the state across chunks); after each chunk
    the GDN states are quantised and dequantised (bits None: left in fp32)."""
    B, L = ids.shape
    for i in eng.gdn:
        eng.rec[i][:B].zero_()
        eng.conv[i][:B].zero_()
    tot, cnt = 0.0, 0
    t = eng.t
    for a in range(0, L - 1, every):
        chunk = ids[:, a:min(a + every, L - 1)]
        T = chunk.shape[1]
        pos = (a + torch.arange(T, device=ids.device))[None].expand(B, -1)
        h = t.model.embed_tokens(chunk)
        cos, sin = t.model.rotary_emb(h, pos[None].expand(3, -1, -1))
        for li, layer in enumerate(eng.layers):            # = Engine.target_forward(verify=False), keeping the logits
            r = h
            x = layer.input_layernorm(h)
            if layer.block_type == "linear_attention":
                x = eng._gdn(li, layer.linear_attn, x, False, B)
            else:
                x = eng._attn(li, layer.self_attn, x, pos, cos, sin, a + T, B)
            h = r + x
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        logits = t.lm_head(t.model.norm(h)).float()
        tgt = ids[:, a + 1:a + 1 + T]
        tot -= torch.log_softmax(logits, -1).gather(-1, tgt[..., None])[..., 0].sum().item()
        cnt += tgt.numel()
        if bits:
            for i in eng.gdn:
                eng.rec[i][:B].copy_(qdq(eng.rec[i][:B], bits))
    return tot, cnt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", default="6,2")
    ap.add_argument("--bits", default="8,4")
    ap.add_argument("--seqs", type=int, default=128)
    ap.add_argument("--len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default=os.path.join(ROOT, "results_ppl/ppl.json"))
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, "models/Qwen3.5-4B"))
    model = AutoModelForCausalLM.from_pretrained(os.path.join(ROOT, "models/Qwen3.5-4B"), attn_implementation="sdpa",
                                                 dtype=torch.bfloat16).cuda().eval()
    try:
        text = "\n\n".join(t for t in load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"] if t.strip())
        src = "wikitext-2-raw-v1 test"
    except Exception as e:                                   # offline fallback: GSM8K test questions + answers
        print("[ppl] wikitext unavailable:", str(e)[:120], flush=True)
        ds = load_dataset("openai/gsm8k", "main", split="test")
        text = "\n\n".join(q + "\n" + an for q, an in zip(ds["question"], ds["answer"]))
        src = "openai/gsm8k main test (question + answer)"
    allids = torch.tensor(tok.encode(text, add_special_tokens=False))
    n = min(a.seqs, len(allids) // a.len)
    data = allids[: n * a.len].view(n, a.len).cuda()
    eng = Engine(model, types.SimpleNamespace(target_layer_ids=[]), "ar", a.batch, a.len, 0, 16)
    res = dict(dataset=src, seqs=n, len=a.len, tokens=n * (a.len - 1), runs={})
    for every in map(int, a.every.split(",")):
        for bits in [None] + [int(b) for b in a.bits.split(",")]:
            tot = cnt = 0
            for i in range(0, n, a.batch):
                t, c = nll(eng, data[i:i + a.batch], every, bits)
                tot, cnt = tot + t, cnt + c
            ppl = math.exp(tot / cnt)
            key = f"every{every}_" + (f"q{bits}" if bits else "fp32")
            res["runs"][key] = dict(every=every, bits=bits or 32, ppl=ppl, nll=tot / cnt)
            print(f"[ppl] {key}: {ppl:.4f}", flush=True)
        base = res["runs"][f"every{every}_fp32"]["ppl"]
        for k, v in res["runs"].items():
            if v["every"] == every:
                v["rel_change_pct"] = 100 * (v["ppl"] / base - 1)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("written", a.out, flush=True)
