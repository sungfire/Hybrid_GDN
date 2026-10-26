"""Output agreement of a run against an exact reference run (same prompts, greedy): identical sequences, common
prefix, and GSM8K answer agreement (last number in the decoded text). youngjoon, 10/24.
Usage: python q4/compare_outputs.py REF_DIR RUN_DIR [--bs 1] [--model models/Qwen3.5-4B]"""
import argparse
import json
import os
import re


def load(d, bs):
    return {r["order"]: r for r in map(json.loads, open(os.path.join(d, "raw", f"dflash_bs{bs}_run1.jsonl")))}


def last_number(text):
    m = re.findall(r"-?\d[\d,]*\.?\d*", text.replace("$", ""))
    return m[-1].replace(",", "").rstrip(".") if m else None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("run")
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--model", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                    "models/Qwen3.5-4B"))
    a = ap.parse_args()
    ref, run = load(a.ref, a.bs), load(a.run, a.bs)
    from transformers import AutoTokenizer
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained(a.model)
    gold = load_dataset("openai/gsm8k", "main", split="test")
    stops = {i for i in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>"))
             if isinstance(i, int) and i >= 0}

    def cut(ids):                     # the engine decodes max_new tokens without stopping: keep the first answer turn
        return next((ids[:i] for i, t in enumerate(ids) if t in stops), ids)
    n = same = ans_same = acc_ref = acc_run = 0
    prefix, first_div = [], []
    for o, r in ref.items():
        if o not in run:
            continue
        x, y = cut(r["output"]), cut(run[o]["output"])
        n += 1
        p = next((i for i, (u, w) in enumerate(zip(x, y)) if u != w), min(len(x), len(y)))
        prefix.append(p / max(len(x), 1))
        if x == y:
            same += 1
        else:
            first_div.append(p)
        ax, ay = last_number(tok.decode(x)), last_number(tok.decode(y))
        ans_same += ax == ay
        g = gold[r["qid"]]["answer"].split("####")[-1].strip().replace(",", "")
        acc_ref += ax is not None and ax.split(".")[0] == g if g.lstrip("-").isdigit() else ax == g
        acc_run += ay is not None and ay.split(".")[0] == g if g.lstrip("-").isdigit() else ay == g
    tau = lambda d: sum(sum(r["accept"]) for r in d.values()) / sum(len(r["accept"]) for r in d.values())  # noqa: E731
    out = dict(n=n, identical=same, identical_rate=same / n, mean_common_prefix=sum(prefix) / n,
               median_first_divergence=sorted(first_div)[len(first_div) // 2] if first_div else None,
               final_number_agree=ans_same / n, gsm8k_acc_ref=acc_ref / n, gsm8k_acc_run=acc_run / n,
               tau_ref=tau(ref), tau_run=tau(run),
               mean_answer_len_ref=sum(len(cut(r["output"])) for r in ref.values()) / n,
               answers_cut_at_stop=sum(len(cut(r["output"])) < len(r["output"]) for r in ref.values()))
    print(json.dumps(out, indent=1))
