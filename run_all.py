"""DFlash (Qwen3.5-4B + z-lab/Qwen3.5-4B-DFlash) vs autoregressive on GSM8K-256 with continuous batching.

    python run_all.py --rollback fullstate

Phase 0: env.json. Phase 3: correctness gate (state_check.json, output_check.json); on failure the run stops and
report.md explains why. Phase 4: {ar, dflash} x batch {1, 8, 64, 128} x 3 runs, resumable per cell (cells/*.json,
raw/*.jsonl). Then summary.csv, plots/ and report.md (dflash_cb/report.py).
"""
import argparse
import collections
import gc
import json
import math
import os
import shutil
import sys

import torch
from dflash.model import DFlashDraftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from dflash_cb import checks
from dflash_cb import env as envmod
from dflash_cb.data import load_gsm8k_256, load_nq, load_specbench
from dflash_cb.engine import Engine, sync_time
from dflash_cb.graph_engine import GraphEngine
from dflash_cb.kernels import gdn_verify

ROOT = os.path.dirname(os.path.abspath(__file__))


def rb_kwargs(args):
    if args.rollback in ("oracle", "oracle_copy", "oracle_fused", "oracle_fused_q4", "oracle_fused_q8", "oracle_dq8",
                         "oracle_d32", "oracle_dq4", "oracle_dq8c", "oracle_dq4c"):
        return dict(trace=args.oracle_from)
    if args.rollback in ("fixed", "fixed_q4", "fixed_q8"):
        return dict(als=[int(x) for x in args.slots.split(",")])
    if args.rollback in ("hist_q8", "confsm_q8", "hist_dq8", "confsm_dq8", "hist_dq4", "confsm_dq4", "confsm_dq8c",
                         "hist_dq8c", "confsm_d32", "confsm_dq4c", "hist_dq4c"):
        return dict(cost_table=args.cost_table)
    if args.rollback in ("confhead_q8", "confhead_dq8", "confhead_dq4"):
        return dict(cost_table=args.cost_table, head=args.head)
    if args.rollback == "confkb":
        return dict(select=args.select, k=args.k, thr=args.thr, head=args.head)
    return dict(select=args.select, k=args.k, thr=args.thr) if args.rollback == "confk" else {}


def out_dir(args):
    if args.out:
        return args.out
    tag = ("graph_" if args.engine == "graph" else "") + args.rollback
    if args.rollback in ("confk", "confkb"):
        tag += f"_top{args.k}" if args.select == "topk" else f"_thr{args.thr:g}"
    if args.rollback in ("fixed", "fixed_q4", "fixed_q8"):
        tag += "_s" + args.slots.replace(",", "-")
    return os.path.join(ROOT, f"results_{tag}")


def fullstate_bytes(cfg, n_gdn, bs, S):
    """Snapshot buffer size = GDN layers x positions x (recurrent fp32 + conv bf16 state) x batch."""
    rec = cfg.linear_num_value_heads * cfg.linear_key_head_dim * cfg.linear_value_head_dim * 4
    conv_dim = cfg.linear_key_head_dim * cfg.linear_num_key_heads * 2 + cfg.linear_value_head_dim * cfg.linear_num_value_heads
    conv = conv_dim * (cfg.linear_conv_kernel_dim - 1) * 2
    return dict(layers=n_gdn, positions=S, batch=bs, rec_state_bytes=rec, conv_state_bytes=conv,
                total_bytes=n_gdn * S * (rec + conv) * bs)


def make_engine(args, target, draft, method, bs, max_prompt, debug=False):
    if args.engine == "graph":
        eng = GraphEngine(target, draft, method, bs, max_prompt, args.max_new, args.block, args.rollback,
                          rb_kwargs(args), debug=debug)
        eng.capture()
        return eng
    return Engine(target, draft, method, bs, max_prompt, args.max_new, args.block, args.rollback, rb_kwargs(args))


def run_once(eng, prompts, qids):
    """Continuous batching over all prompts. Returns per-question records, per-step records and a summary."""
    eng.reset()
    if eng.dflash:
        eng.rb.reset_stats()
    queue = collections.deque(range(len(prompts)))
    prefill_s = decode_s = kern_ms = 0.0
    sec = collections.defaultdict(float)
    steps, recs = [], []

    def fill():
        nonlocal prefill_s, decode_s
        while eng.n < eng.bs and queue:
            k = queue.popleft()
            prefill_s += eng.insert(k, torch.tensor(prompts[k]))
        if hasattr(eng, "pop_ingest"):          # graph engine: draft context of new prompts = draft (decode) work
            t = eng.pop_ingest()
            decode_s += t
            sec["draft"] += t
            for r in eng.rows:
                r["t_decode"] += t

    fill()
    while eng.n > 0:
        T, prod, finished, kern = eng.step()
        wall = T.pop("_wall", None)
        decode_s += wall if wall is not None else sum(T.values())
        for k, v in T.items():
            sec[k] += v
        if wall is not None:                     # graph engine: GPU section times + host gap (launch, sync, idle)
            sec["host_gap"] += wall - sum(T.values())
        kern_ms += kern
        steps.append(dict(n=len(prod), **T, kern_ms=kern, produced=prod if eng.dflash else None))
        t0 = sync_time()
        for r in sorted(finished, reverse=True):
            recs.append(eng.remove(r))
        t1 = sync_time()
        decode_s += t1 - t0
        sec["remove"] += t1 - t0
        fill()
    q = []
    for r in sorted(recs, key=lambda x: x["qid"]):
        dec = sum(r["accept"])                       # tokens produced by decode steps (first token comes from prefill)
        q.append(dict(order=r["qid"], qid=qids[r["qid"]], prompt_len=r["prompt_len"], output=r["output"],
                      n_output=len(r["output"]), decode_tokens=dec, t_decode=r["t_decode"],
                      tpot_ms=1e3 * r["t_decode"] / dec, steps=len(r["accept"]),
                      accept=r["accept"] if eng.dflash else None, tau=dec / len(r["accept"])))
    dec_tok = sum(x["decode_tokens"] for x in q)
    row_steps = sum(x["steps"] for x in q)
    summary = dict(n_seq=len(q), decode_tokens=dec_tok, decode_s=decode_s, prefill_s=prefill_s,
                   throughput=dec_tok / decode_s, tau=dec_tok / row_steps,
                   tpot_ms=sum(x["tpot_ms"] for x in q) / len(q), steps=len(steps), sections=dict(sec),
                   gdn_verify_kernel_s=kern_ms / 1e3)
    if eng.dflash and hasattr(eng.rb, "pred_stats"):    # confk: snapshot hit rate / replay lengths
        summary["prediction"] = eng.rb.pred_stats()
    if eng.dflash and hasattr(eng.rb, "hist"):          # modeal: online acceptance-length histogram and chosen m
        summary["al_hist"] = eng.rb.observed_hist() if hasattr(eng.rb, "observed_hist") else eng.rb.hist
        summary["m_counts"] = dict(collections.Counter(eng.rb.m_trace))
    return q, steps, summary


def warmup(eng, prompts, iters):
    eng.reset()
    for k in range(min(eng.bs, len(prompts))):
        eng.insert(k, torch.tensor(prompts[k]))
    for _ in range(iters):
        eng.step()
    eng.reset()


@torch.inference_mode()
def calibrate_store(eng, prompts, reps=20):
    """GPU time of the fused verify kernel with vs without writing the per-position states, on real inputs of a
    full batch (one GDN layer; all GDN layers have the same shape). Used to split 'state save' out of 'verify'."""
    eng.reset()
    for k in range(min(eng.bs, len(prompts))):
        eng.insert(k, torch.tensor(prompts[k]))
    n = eng.n
    eng.rb.capture = {}
    eng.step()
    cap, eng.rb.capture = eng.rb.capture, None
    i = eng.gdn[0]
    q, k, v, g, beta = cap[i]
    skw = eng.rb.store_kwargs()                      # the strategy's own snapshot writes for this step
    res = {}
    for store in (True, False):
        kw = dict(store=store, **skw)
        for _ in range(3):
            gdn_verify(q, k, v, g, beta, eng.rec[i][:n], eng.rb.snap_rec[i][:n], **kw)
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(reps):
            gdn_verify(q, k, v, g, beta, eng.rec[i][:n], eng.rb.snap_rec[i][:n], **kw)
        e1.record()
        torch.cuda.synchronize()
        res["store" if store else "nostore"] = e0.elapsed_time(e1) / reps
    eng.reset()
    eng.rb.reset_stats()
    L = len(eng.gdn)
    return dict(batch=n, store_position=("all" if not skw else "per-row slots" if "slot_map" in skw
                                         else skw["store_t"] + 1), layer_ms_store=res["store"], layer_ms_nostore=res["nostore"],
                step_ms_store=res["store"] * L, step_ms_nostore=res["nostore"] * L,
                step_ms_save_overhead=(res["store"] - res["nostore"]) * L)


def run_cell(method, bs, args, models, prompts, qids, out, max_prompt):
    name = f"{method}_bs{bs}"
    path = os.path.join(out, "cells", name + ".json")
    if os.path.exists(path) and json.load(open(path)).get("status") in ("done", "oom"):
        print(f"[skip] {name} already recorded", flush=True)
        return
    src = args.reuse_ar_from
    if method == "ar" and src:
        src_cell = os.path.join(src, "cells", name + ".json")
        same_env = json.load(open(os.path.join(src, "env.json"))) == json.load(open(os.path.join(out, "env.json")))
        if os.path.exists(src_cell) and json.load(open(src_cell)).get("status") == "done" and same_env:
            info = json.load(open(src_cell))
            info["reused_from"] = os.path.abspath(src)
            for f in os.listdir(os.path.join(src, "raw")):
                if f.startswith(name + "_run"):
                    shutil.copy(os.path.join(src, "raw", f), os.path.join(out, "raw", f))
            json.dump(info, open(path, "w"), indent=1)
            print(f"[reuse] {name} from {src} (identical env.json)", flush=True)
            return
        print(f"[reuse] {name}: not reusable from {src} (missing cell or env differs); measuring", flush=True)
    target, draft = models()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    info = dict(method=method, batch=bs, block=args.block if method == "dflash" else 1,
                rollback=args.rollback if method == "dflash" else None,
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                gpu_uuid=str(torch.cuda.get_device_properties(0).uuid))
    if method == "dflash":                  # per-state sizes; report.py derives each strategy's requirement
        info["snapshot_formula"] = fullstate_bytes(target.config, sum(
            1 for t in target.config.layer_types if t == "linear_attention"), bs, args.block)
    stage, runs = "alloc", []
    try:
        stage = "alloc+capture" if args.engine == "graph" else "alloc"
        eng = make_engine(args, target, draft, method, bs, max_prompt)
        info["engine"] = args.engine
        if args.engine == "graph":
            info["buckets"] = eng.buckets
        if method == "dflash":
            info["snapshot_allocated"] = eng.rb.buffer_bytes()
        stage = "warmup"
        warmup(eng, prompts, args.warmup)
        if method == "dflash" and eng.rb.stores_snapshots and args.engine != "graph":
            stage = "calibrate"
            info["store_calibration"] = calibrate_store(eng, prompts)
        for r in range(1, args.runs + 1):
            stage = f"run{r}"
            q, steps, s = run_once(eng, prompts, qids)
            with open(os.path.join(out, "raw", f"{name}_run{r}.jsonl"), "w") as f:
                for x in q:
                    f.write(json.dumps(x) + "\n")
            with open(os.path.join(out, "raw", f"{name}_run{r}_steps.jsonl"), "w") as f:
                for x in steps:
                    f.write(json.dumps(x) + "\n")
            s["run"] = r
            runs.append(s)
            print(f"[{name} run{r}] {s['throughput']:.1f} tok/s tau={s['tau']:.3f} tpot={s['tpot_ms']:.1f}ms "
                  f"decode={s['decode_s']:.1f}s prefill={s['prefill_s']:.1f}s", flush=True)
        info.update(status="done")
    except torch.OutOfMemoryError as e:
        info.update(status="oom", stage=stage, error=str(e).splitlines()[0][:400])
        print(f"[{name}] OOM at {stage}: {info['error']}", flush=True)
    info.update(runs=runs, peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9)
    eng = None
    gc.collect()
    torch.cuda.empty_cache()
    json.dump(info, open(path, "w"), indent=1)


def phase3(args, models, prompts, qids, out, max_prompt):
    """Correctness gate. Returns True when both checks pass."""
    path_s, path_o = os.path.join(out, "state_check.json"), os.path.join(out, "output_check.json")
    if not (os.path.exists(path_s) and os.path.exists(path_o)):
        target, draft = models()
    if not os.path.exists(path_s):
        if args.engine == "graph":
            eng = make_engine(args, target, draft, "dflash", 1, max_prompt, debug=True)
            res = checks.state_check_graph(eng, prompts, n_prompts=10, seed=0)
        else:
            eng = Engine(target, draft, "dflash", 1, max_prompt, args.max_new, args.block, args.rollback,
                         rb_kwargs(args))
            res = checks.state_check(eng, prompts, n_prompts=10, seed=0)
            res["conv_exact_max_abs"] = checks.conv_exact(eng, prompts[0])
            res["verdict"] = checks.state_pass(res, res["conv_exact_max_abs"])
        json.dump(res, open(path_s, "w"), indent=1)
        eng = None
        torch.cuda.empty_cache()
    res = json.load(open(path_s))
    print("[phase3] state check:", json.dumps(res["verdict"]), json.dumps(res["worst"]), flush=True)
    if not os.path.exists(path_o):
        outs, ar_from = {}, None
        src = args.reuse_ar_from and os.path.join(args.reuse_ar_from, "output_check.json")
        if src and os.path.exists(src) and json.load(open(os.path.join(args.reuse_ar_from, "env.json"))) == \
                json.load(open(os.path.join(out, "env.json"))):
            outs["ar"], ar_from = json.load(open(src))["outputs"]["ar"], os.path.abspath(src)
            print(f"[phase3] AR outputs of the output check reused from {src}", flush=True)
        for m in ("ar", "dflash"):
            if m in outs:
                continue
            eng = make_engine(args, target, draft, m, 1, max_prompt)
            q, _, _ = run_once(eng, prompts[:50], qids[:50])
            outs[m] = [x["output"] for x in q]
            eng = None
            torch.cuda.empty_cache()
        rows = []
        for k, (a, d) in enumerate(zip(outs["ar"], outs["dflash"])):
            diff = next((j for j in range(min(len(a), len(d))) if a[j] != d[j]), None)
            row = dict(order=k, qid=qids[k], identical=a == d, first_diff=diff)
            if diff is not None:   # target logits (stock HF forward) on the shared prefix at the divergence
                ids = torch.tensor([prompts[k] + a[:diff]], device=target.device)
                with torch.inference_mode():
                    lg = target(ids, use_cache=False, logits_to_keep=1).logits[0, -1].float()
                top = lg.topk(2)
                t1 = top.values[0].item()
                row.update(top1_logit=t1, bf16_ulp_at_top1=2.0 ** (math.floor(math.log2(abs(t1))) - 7))
                row.update(ar_token=a[diff], dflash_token=d[diff], top1_top2_gap=(top.values[0] - top.values[1]).item(),
                           logit_ar_minus_dflash=(lg[a[diff]] - lg[d[diff]]).item(),
                           hf_top1_is=("ar" if top.indices[0] == a[diff] else
                                       "dflash" if top.indices[0] == d[diff] else "other"))
            rows.append(row)
        n_id = sum(r["identical"] for r in rows)
        extra = {}
        eager = os.path.join(ROOT, f"results_{args.rollback}", "output_check.json")
        if args.engine == "graph" and os.path.exists(eager):   # same rollback, eager engine (information only)
            eo = json.load(open(eager))["outputs"]
            extra = {f"identical_to_eager_{m}": sum(a == b for a, b in zip(outs[m], eo[m])) for m in ("ar", "dflash")}
        json.dump(dict(n=len(rows), identical=n_id, rows=rows, outputs=outs, ar_outputs_from=ar_from, **extra),
                  open(path_o, "w"), indent=1)
    oc = json.load(open(path_o))
    print(f"[phase3] output check: {oc['identical']}/{oc['n']} identical", flush=True)
    return res["verdict"]["passed"] and oc["identical"] == oc["n"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollback", default="fullstate")
    ap.add_argument("--out", default=None, help="default: results_<rollback>")
    ap.add_argument("--model", default=os.path.join(ROOT, "models/Qwen3.5-4B"))
    ap.add_argument("--draft", default=os.path.join(ROOT, "models/Qwen3.5-4B-DFlash"))
    ap.add_argument("--batch-sizes", default="1,8,64,128")
    ap.add_argument("--methods", default="ar,dflash")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--phases", default="0,3,4,report")
    ap.add_argument("--reuse-ar-from", default=None, help="results dir whose AR cells to reuse if env.json matches")
    ap.add_argument("--engine", default="eager", choices=["eager", "graph"],
                    help="graph: CUDA-graph captured decode steps (results_graph_<rollback>)")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N of the 256 questions (smoke tests)")
    ap.add_argument("--gsm8k-all", action="store_true", help="all 1319 GSM8K test questions instead of the 256 subset")
    ap.add_argument("--dataset", default="gsm8k", help="gsm8k (default), nq[:n] (NQ-open dev sample) or specbench:<category>, e.g. specbench:qa")
    ap.add_argument("--select", default="topk", choices=["topk", "thr"], help="confk: how snapshot rows are chosen")
    ap.add_argument("--k", type=int, default=1, help="confk topk: number of most likely stop rows to keep")
    ap.add_argument("--thr", type=float, default=0.5, help="confk thr: per-token confidence threshold")
    ap.add_argument("--cost-table", default=os.path.join(ROOT, "q4/bench_cost_rtx3090.json"),
                    help="hist_* / confsm_* / confhead_*: kernel times per batch size (q4/test_q4.py --full --json)")
    ap.add_argument("--slots", default="6", help="fixed / fixed_q4 / fixed_q8: kept acceptance lengths, comma separated")
    ap.add_argument("--oracle-from", default=None,
                    help="oracle: results dir of the same engine whose raw/dflash_bs<B>_run1.jsonl AL traces to replay")
    ap.add_argument("--head", default=os.path.join(ROOT, "ext/confk_head_B/confk-head-B/heads/head_mlp.pt"),
                    help="confkb: confidence head checkpoint (conf-K method B)")
    ap.add_argument("--accept-tie-mismatch", action="store_true",
                    help="continue past Phase 3 if the only failures are batch-1 output mismatches on bf16 logit ties")
    args = ap.parse_args()
    out = out_dir(args)
    for d in ("raw", "cells", "plots"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    phases = args.phases.split(",")

    if "0" in phases:
        env = envmod.collect(os.path.join(ROOT, "third_party/dflash"))
        json.dump(env, open(os.path.join(out, "env.json"), "w"), indent=1)
        print("[phase0]", json.dumps(env), flush=True)

    cache = {}

    def models():
        """Target + draft on cuda:0, loaded on first use (skipped / reused cells need no GPU)."""
        if not cache:
            dev = torch.device("cuda:0")
            target = AutoModelForCausalLM.from_pretrained(args.model, attn_implementation="sdpa", dtype=torch.bfloat16)
            cache["t"] = target.to(dev).eval()
            cache["d"] = DFlashDraftModel.from_pretrained(args.draft, attn_implementation="sdpa",
                                                          dtype=torch.bfloat16).to(dev).eval()
            assert cache["d"].block_size == args.block, (cache["d"].block_size, args.block)
        return cache["t"], cache["d"]

    tok = AutoTokenizer.from_pretrained(args.model)
    if args.dataset == "gsm8k":
        qids, prompts = load_gsm8k_256(ROOT, tok, n=None if args.gsm8k_all else 256)
    elif args.dataset.startswith("nq"):                    # nq or nq:<n>: NQ-open dev sample
        qids, prompts = load_nq(ROOT, tok, n=int(args.dataset.split(":", 1)[1]) if ":" in args.dataset else 256)
    else:                                                 # specbench:<category>, e.g. specbench:qa
        qids, prompts = load_specbench(ROOT, tok, args.dataset.split(":", 1)[1])
    if args.limit:
        qids, prompts = qids[: args.limit], prompts[: args.limit]
    max_prompt = max(len(p) for p in prompts)

    if "3" in phases:
        if not phase3(args, models, prompts, qids, out, max_prompt):
            sc = json.load(open(os.path.join(out, "state_check.json")))
            oc = json.load(open(os.path.join(out, "output_check.json")))
            bad = [r for r in oc["rows"] if not r["identical"]]
            gaps = [r["top1_top2_gap"] for r in bad]
            # bf16 near-tie: top-1/top-2 logits within 2 ulps at the top-1 logit's magnitude (user-approved; older
            # records without the magnitude fall back to the earlier fixed 0.125 rule)
            tie_only = sc["verdict"]["passed"] and all(
                r["top1_top2_gap"] <= (2 * r["bf16_ulp_at_top1"] if "bf16_ulp_at_top1" in r else 0.125) for r in bad)
            if not (args.accept_tie_mismatch and tie_only):
                print("[phase3] FAILED: stopping before Phase 4 (see state_check.json / output_check.json)", flush=True)
                sys.exit(2)
            json.dump(dict(reason="user-approved: batch-1 output mismatches recorded as match rate when the state "
                                  "check passes and every mismatch sits on a bf16 near-tie (top1-top2 gap <= 2 ulps "
                                  "at the top-1 logit; earlier records: <= 0.125)",
                           identical=oc["identical"], n=oc["n"], mismatch_gaps=gaps,
                           mismatch_gaps_in_ulps=[r["top1_top2_gap"] / r["bf16_ulp_at_top1"] for r in bad
                                                  if "bf16_ulp_at_top1" in r]),
                      open(os.path.join(out, "phase3_override.json"), "w"), indent=1)
            print(f"[phase3] output check {oc['identical']}/{oc['n']}: all mismatches on bf16 near-ties (<= 2 ulp) -> "
                  "continuing (--accept-tie-mismatch)", flush=True)
    if "4" in phases:
        for m in args.methods.split(","):
            for bs in map(int, args.batch_sizes.split(",")):
                run_cell(m, bs, args, models, prompts, qids, out, max_prompt)
    if "report" in phases:
        from dflash_cb import report
        report.build(out)


if __name__ == "__main__":
    main()
