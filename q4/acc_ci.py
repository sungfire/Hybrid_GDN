"""GSM8K accuracy of runs vs a reference run with the 95% interval of the paired difference (youngjoon, 10/25).
Usage: python q4/acc_ci.py ACC_DIR REF RUN [RUN ...] [--bs 32]"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_outputs import last_number  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("ref")
ap.add_argument("runs", nargs="+")
ap.add_argument("--bs", type=int, default=32)
a = ap.parse_args()
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tok = AutoTokenizer.from_pretrained(os.path.join(R, "models/Qwen3.5-4B"))
gold = load_dataset("openai/gsm8k", "main", split="test")
stops = {i for i in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")) if isinstance(i, int) and i >= 0}


def answers(name):
    out = {}
    for r in map(json.loads, open(os.path.join(a.dir, name, "raw", f"dflash_bs{a.bs}_run1.jsonl"))):
        ids = r["output"]
        ids = next((ids[:i] for i, t in enumerate(ids) if t in stops), ids)
        x = last_number(tok.decode(ids))
        g = gold[r["qid"]]["answer"].split("####")[-1].strip().replace(",", "")
        out[r["qid"]] = (x is not None and (x.split(".")[0] == g if g.lstrip("-").isdigit() else x == g), ids)
    return out


ref = answers(a.ref)
n = len(ref)
res = {a.ref: dict(n=n, acc=100 * sum(v[0] for v in ref.values()) / n)}
print(f"{a.ref:14s} n={n} acc {res[a.ref]['acc']:.2f}%")
for name in a.runs:
    run = answers(name)
    lost = sum(ref[q][0] and not run[q][0] for q in ref)
    gain = sum(run[q][0] and not ref[q][0] for q in ref)
    se = 100 * ((lost + gain - (gain - lost) ** 2 / n) ** 0.5) / n
    d = dict(n=n, acc=100 * sum(v[0] for v in run.values()) / n, diff=100 * (gain - lost) / n, ci95=1.96 * se,
             lost=lost, gained=gain, identical=100 * sum(run[q][1] == ref[q][1] for q in ref) / n)
    res[name] = d
    print(f"{name:14s} acc {d['acc']:.2f}%  diff {d['diff']:+.2f}%p (95% ±{d['ci95']:.2f})  lost {lost} gained {gain}  "
          f"identical {d['identical']:.0f}%")
json.dump(res, open(os.path.join(a.dir, "acc_ci.json"), "w"), indent=1)
