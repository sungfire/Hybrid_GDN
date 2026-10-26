"""GSM8K test subset: 256 questions sampled without replacement (seed 42), zero-shot chat prompts, thinking off."""
import json
import os
import random

from datasets import load_dataset


def load_gsm8k_256(root, tok, n=256, seed=42):
    ds = load_dataset("openai/gsm8k", "main", split="test")
    path = os.path.join(root, "data/gsm8k_256_indices.json")
    if n is None or n >= len(ds):                   # the whole test set, in dataset order (youngjoon, 10/25)
        idx = list(range(len(ds)))
    elif os.path.exists(path):
        idx = json.load(open(path))["indices"]
    else:
        idx = random.Random(seed).sample(range(len(ds)), n)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(dict(dataset="openai/gsm8k main test", n_total=len(ds), n=n, seed=seed,
                       method="random.Random(seed).sample(range(n_total), n)", indices=idx), open(path, "w"), indent=1)
    prompts = []
    for i in idx:
        text = tok.apply_chat_template([{"role": "user", "content": ds[i]["question"]}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        prompts.append(tok.encode(text, add_special_tokens=False))
    return idx, prompts


def load_specbench(root, tok, category):
    """Spec-Bench (data/spec_bench_question.jsonl) questions of one category, first turn only, zero-shot chat prompts,
    thinking off (youngjoon, 10/25). Returns question ids and prompts."""
    rows = [json.loads(line) for line in open(os.path.join(root, "data/spec_bench_question.jsonl"))]
    rows = [r for r in rows if r["category"] == category]
    prompts = []
    for r in rows:
        text = tok.apply_chat_template([{"role": "user", "content": r["turns"][0]}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        prompts.append(tok.encode(text, add_special_tokens=False))
    return [r["question_id"] for r in rows], prompts


def load_nq(root, tok, n=256, seed=42):
    """NQ-open dev (data/nq_open_dev.jsonl, the source of Spec-Bench QA): n questions sampled with seed, asked as
    Spec-Bench QA does (the question alone, capitalised, with '?'), zero-shot chat prompts, thinking off (youngjoon,
    10/25). Returns dev indices and prompts."""
    rows = [json.loads(line) for line in open(os.path.join(root, "data/nq_open_dev.jsonl"))]
    idx = random.Random(seed).sample(range(len(rows)), n)
    prompts = []
    for i in idx:
        q = rows[i]["question"].strip()
        q = q[:1].upper() + q[1:] + ("" if q.endswith("?") else "?")
        text = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        prompts.append(tok.encode(text, add_special_tokens=False))
    return idx, prompts
