"""Render the README's result tables from results/*. Every number in the README's results
section comes from here, so the README cannot drift from the measurements."""

import json
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"


def offline() -> str:
    rows = {json.loads(l)["arm"]: json.loads(l) for l in open(R / "offline.jsonl")}
    base = rows["hf-seq"]["output_tok_per_s"]
    label = {
        "hf-seq": "HF `generate`, one request at a time",
        "hf-static16": "HF `generate`, static batches of 16 (left-padded)",
        "nanoserve-nokernel": "nanoserve, stock cuBLAS for every matmul",
        "nanoserve": "**nanoserve** (measured GEMM dispatch)",
        "vllm": "vLLM 0.34 (CUDA graphs + torch.compile, Triton attention)",
        "vllm-eager": "vLLM, `enforce_eager=True`",
    }
    out = ["| engine | output tok/s | vs HF sequential | wall time | memory for KV / activations |",
           "|---|---:|---:|---:|---|"]
    for arm in ("hf-seq", "hf-static16", "vllm-eager", "vllm", "nanoserve-nokernel", "nanoserve"):
        if arm not in rows:
            continue
        r = rows[arm]
        if arm.startswith("vllm"):
            mem = "paged KV pool (its own allocator)"
        elif arm.startswith("hf"):
            mem = f"{r['peak_mem_mib']:.0f} MiB peak above weights"
            if arm == "hf-static16":
                waste = 1 - r["output_tokens"] / r["computed_output_tokens"]
                mem += f"; {waste:.0%} of generated tokens discarded"
        else:
            mem = f"{r['peak_kv_mib_used']:.0f} MiB of KV at peak, {r['kv_slot_utilisation']:.0%} slot utilisation"
        tps = r["output_tok_per_s"]
        b = f"**{tps:.1f}**" if arm == "nanoserve" else f"{tps:.1f}"
        x = f"**{tps / base:.1f}x**" if arm == "nanoserve" else f"{tps / base:.1f}x"
        out.append(f"| {label[arm]} | {b} | {x} | {r['wall_s']:.0f} s | {mem} |")
    return "\n".join(out)


def online() -> str:
    rows = [json.loads(l) for l in open(R / "online.jsonl")]
    label = {"nanoserve": "nanoserve", "nanoserve-seq": "nanoserve, max 1 sequence",
             "hf-server": "FastAPI + HF `generate`"}
    out = ["| server | req/s offered | output tok/s | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | end-to-end p50 |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]

    def ms(v):
        return f"{v / 1000:.1f} s" if v >= 1000 else f"{v:.0f} ms"

    for arm in ("nanoserve", "nanoserve-seq", "hf-server"):
        for r in sorted((r for r in rows if r["arm"] == arm), key=lambda r: r["rate"]):
            out.append(f"| {label[arm]} | {r['rate']:g} | {r['output_tok_per_s']:.1f} | {ms(r['ttft_p50_ms'])} | "
                       f"{ms(r['ttft_p99_ms'])} | {ms(r['tpot_p50_ms'])} | {ms(r['tpot_p99_ms'])} | {ms(r['e2e_p50_ms'])} |")
    return "\n".join(out)


def spec() -> str:
    rows = json.load(open(R / "spec.json"))
    out = ["| workload | concurrent | k | tok/s | speed-up | draft acceptance | tokens per forward (batch 1) | output == batch-1 plain greedy |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        base = next(x for x in rows if x["workload"] == r["workload"] and x["concurrency"] == r["concurrency"] and x["k"] == 0)
        sp = r["tok_per_s"] / base["tok_per_s"]
        acc = "–" if r["k"] == 0 else f"{r['acceptance']:.0%}"
        tpf = f"{r['tokens_per_step']:.2f}" if r["concurrency"] == 1 else "–"
        out.append(f"| {r['workload']} | {r['concurrency']} | {r['k']} | {r['tok_per_s']:.1f} | {sp:.2f}x | {acc} | "
                   f"{tpf} | {r['identical_to_plain']}/{r['of']} |")
    return "\n".join(out)


def steps() -> str:
    d = json.load(open(R / "steps.json"))
    out = ["| | stock cuBLAS | measured dispatch |", "|---|---:|---:|"]
    for a, b in zip(d["stock_cublas"]["prefill"], d["dispatch"]["prefill"]):
        out.append(f"| prefill {a['seqs']} x {a['len']} tokens | {a['tok_per_s']} tok/s | **{b['tok_per_s']} tok/s** "
                   f"({b['tflops']:.2f} TFLOP/s, {b['mfu_vs_fp32_peak']:.0%} of fp32 peak) |")
    for a, b in zip(d["stock_cublas"]["decode"], d["dispatch"]["decode"]):
        out.append(f"| decode step, batch {a['batch']} | {a['ms_per_step']:.1f} ms ({a['tok_per_s']:.0f} tok/s) | "
                   f"**{b['ms_per_step']:.1f} ms ({b['tok_per_s']:.0f} tok/s)** |")
    return "\n".join(out)


def attention() -> str:
    rows = json.load(open(R / "attention.json"))
    out = ["| shape | SDPA (mem-efficient) | grouped-GQA matmul |", "|---|---:|---:|"]
    for r in rows:
        a, b = r["sdpa_ms"], r["grouped_ms"]
        fa, fb = (f"**{a:.2f} ms**", f"{b:.2f} ms") if a < b else (f"{a:.2f} ms", f"**{b:.2f} ms**")
        out.append(f"| {r['shape']} | {fa} | {fb} |")
    return "\n".join(out)


def write_readme(path: Path) -> None:
    import re

    text = path.read_text()
    for name, f in [("offline", offline), ("online", online), ("spec", spec), ("steps", steps), ("attention", attention)]:
        pat = re.compile(rf"(<!-- table:{name} -->\n).*?(<!-- /table:{name} -->)", re.S)
        text = pat.sub(lambda m: m.group(1) + f() + "\n" + m.group(2), text)
    path.write_text(text)


if __name__ == "__main__":
    import sys

    if "--write" in sys.argv:
        write_readme(R.parent / "README.md")
        sys.exit(0)
    for name, f in [("offline", offline), ("online", online), ("spec", spec), ("steps", steps), ("attention", attention)]:
        try:
            print(f"### {name}\n{f()}\n")
        except FileNotFoundError as e:
            print(f"### {name}: missing {e.filename}\n")
