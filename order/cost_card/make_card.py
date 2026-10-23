"""Assemble raw.json and COST_CARD.md from bench_judge.py outputs.

Usage: python make_card.py OUT_DIR group/key:run=path.json [...]
  group  "new" (current judges) or "baseline" (earlier card)
  run    "cold" or "warm"; the table uses the warm run when present.
"""
import glob
import json
import os
import platform
import shutil
import sys

CALLS = {"Set M (MT-Bench pairs, ~80 x 2 orders)": 160, "Set F (fresh pairs, ~40 x 2 orders)": 80}
EXCLUDE_FROM_TOTALS = {"Qwen/Qwen3-8B"}

DATASET = {
    "repo": "lmsys/mt_bench_human_judgments",
    "files": {
        "data/gpt4_pair-00000-of-00001-c0b431264a82ddc0.parquet": 649996,
        "data/human-00000-of-00001-25f4910818759289.parquet": 738612,
    },
    "total_bytes": 1393033,
    "how_checked": "HfApi().dataset_info(files_metadata=True); not downloaded",
}

# Hub metadata (HfApi().model_info, 2026-10-23) for every candidate considered. Nothing here is a measurement.
CANDIDATES = [
    ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca", False, 1.50, "measured"),
    ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", False, 4.06, "measured"),
    ("Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c", False, 8.04, "measured"),
    ("microsoft/Phi-4-mini-instruct", "cfbefacb99257ffa30c83adab238a50856ac3083", False, 7.67, "measured"),
    ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218", False, 16.38, "does not fit in RAM; measured anyway via mmap thrashing (see Fit note)"),
    ("Qwen/Qwen3-4B-Instruct-2507", "cdbee75f17c01a7cc42f958dc650907174af0554", False, 8.04, "not measured: optional; same size class as Qwen3-4B"),
    ("Qwen/Qwen3.5-0.8B / 2B / 4B", "main", False, None, "not measured: `qwen3_5` multimodal architecture needs transformers>=5; this repo pins <5"),
    ("HuggingFaceTB/SmolLM3-3B", "a07cc9a04f16550a088caea529712d1d335b0ac1", False, 6.15, "not measured: optional"),
    ("google/gemma-3-4b-it", "093f9f388b31de276ce2de164bdc2081324b9767", "manual", 8.60, "gated, skipped"),
    ("meta-llama/Llama-3.2-3B-Instruct", "0cb88a4f764b7a12671c53f0838cd831a0843b95", "manual", 6.43, "gated, not measured (401 without token)"),
]

GPU_ESTIMATE = {
    "label": "list-price estimate, nothing paid, not measured",
    "judge": "Qwen/Qwen3-32B (ungated, rev 9216db5781bf21249d130ec9da846c4624c16137, 65.52 GB bf16 safetensors)",
    "gpu": "1x NVIDIA H100 PCIe 80 GB (Lambda on-demand)",
    "price_usd_per_hr": 3.29,
    "price_source": "https://lambda.ai/service/gpu-cloud (1x table, 'NVIDIA H100 PCIe', $3.29/GPU/hr plus tax; fetched 2026-10-23)",
    "calls": 480,
    "assumed_s_per_call": 1.0,
    "throughput_assumption": (
        "Assumed 1.0 s/call as a conservative ceiling: 600-token prefill on a 32B dense bf16 model is ~3.8e13 FLOP, "
        "~0.1 s at 35% of H100 bf16 peak; 2 decode tokens are memory-bound, ~0.04 s at 3.35 TB/s over 65 GB. "
        "1 s/call leaves ~5x margin for an unoptimised HF transformers loop. Not benchmarked."
    ),
    "overhead_assumption": "Instance boot + 65 GB weight download + load budgeted at 45 min; billing is per minute.",
}


def machine():
    import torch

    mem_kb = int(next(l for l in open("/proc/meminfo") if l.startswith("MemTotal")).split()[1])
    has_gpu = torch.cuda.is_available() or shutil.which("nvidia-smi") is not None or glob.glob("/dev/nvidia*")
    return {
        "nproc": os.cpu_count(),
        "ram_total_gib": round(mem_kb / 1024**2, 2),
        "gpu": "present" if has_gpu else "none (torch.cuda.is_available()=False, no nvidia-smi, no /dev/nvidia*)",
        "cpu_model": next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "unknown"),
        "kernel": platform.release(),
        "torch": torch.__version__,
    }


def minutes(sec):
    return f"{sec/60:.1f} min"


def main():
    out_dir = sys.argv[1]
    runs = {"new": {}, "baseline": {}}
    for arg in sys.argv[2:]:
        label, path = arg.split("=", 1)
        group, rest = label.split("/")
        key, run = rest.split(":")
        runs[group].setdefault(key, {})[run] = json.load(open(path))

    gpu = dict(GPU_ESTIMATE)
    gpu["est_compute_min"] = gpu["calls"] * gpu["assumed_s_per_call"] / 60
    gpu["est_total_hr"] = round((gpu["est_compute_min"] + 45) / 60, 2)
    gpu["est_cost_usd"] = round(gpu["est_total_hr"] * gpu["price_usd_per_hr"], 2)

    raw = {
        "machine": machine(),
        "planned_calls_per_judge": CALLS,
        "mt_bench_dataset": DATASET,
        "candidates_hub_metadata": [dict(zip(("model", "revision", "gated", "safetensors_gb", "outcome"), c)) for c in CANDIDATES],
        "gpu_estimate": gpu,
        "judges": runs,
    }
    json.dump(raw, open(os.path.join(out_dir, "raw.json"), "w"), indent=2)

    m = raw["machine"]
    L = [
        "# Order Is All You Need: judge cost card (draft)",
        "",
        "CPU-only measurement of one pairwise LLM-judge call on this Cursor VM. Not the experiment.",
        "",
        "## Machine",
        "",
        f"- CPU: {m['cpu_model']}, nproc = {m['nproc']}",
        f"- RAM: {m['ram_total_gib']} GiB total (hypervisor memory balloon present; see notes)",
        f"- GPU: {m['gpu']}",
        f"- torch {m['torch']} (CPU wheel), transformers 4.57.6, bf16 weights, greedy decoding (temperature 0), {m['nproc']} threads",
        "",
        "## Protocol",
        "",
        "- 10 synthetic pairwise prompts (fixed system/user template, two ~150-250-word responses each), plus 1 untimed warmup call.",
        "- Per call: one forward pass to read log-probs of tokens `A`, `B`, `C` at the verdict position, then `generate(max_new_tokens=2, do_sample=False)`.",
        "- Qwen3 hybrid-thinking models are run with `enable_thinking=False` in the chat template, so the first generated token is the verdict.",
        "- Peak RSS from `resource.getrusage(RUSAGE_SELF).ru_maxrss`. Warm run = weights already in the local HF cache.",
        "- Every number in the tables is copied from `raw.json`; cells that say *not measured* were not run and say why.",
        "",
        "## Per-call cost (warm run)",
        "",
        "| judge | HF revision | thinking | params (B) | load s | peak RSS GB | prompt tokens (mean) | mean s/call | p95 s/call | parse rate |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    def row(pick, tag=""):
        th = pick.get("enable_thinking")
        th = "off" if th is False else ("n/a" if th is None or str(th).startswith("n/a") else str(th))
        return (
            f"| {pick['model']}{tag} | `{(pick.get('revision') or 'not recorded')[:12]}` | {th} | {pick['params_B']:.2f} | {pick['load_s']:.2f} | "
            f"{pick['peak_rss_gb']:.2f} | {pick['prompt_tokens_mean']:.0f} | {pick['mean_s_per_call']:.2f} | {pick['p95_s_per_call']:.2f} | "
            f"{pick['parse_rate']:.0%} ({sum(c['parsed'] is not None for c in pick['calls'])}/{pick['n_calls']}) |"
        )

    measured = []
    for group, tag in (("new", ""), ("baseline", " (baseline)")):
        for key, r in runs[group].items():
            pick = r.get("warm") or next(iter(r.values()))
            t = tag if "warm" in r else tag + " (cold run only; does not fit, disk-thrashing)"
            if pick["status"] != "ok":
                L.append(f"| {pick['model']}{tag} | n/a | n/a | not measured | {pick['status']} | not measured | not measured | not measured | not measured | not measured |")
                continue
            L.append(row(pick, t))
            measured.append(pick)

    L += [
        "",
        "Baseline rows are the earlier card's judges; their revisions were recovered post hoc from the HF cache snapshot directory used by those runs (`revision_source` in `raw.json`). Their cold runs are also in `raw.json`.",
        "",
        "## Extrapolation to the planned run",
        "",
        "Two pair sets, each judged in both orders: Set M = 160 calls, Set F = 80 calls, 240 per judge. "
        "Wall time = calls x measured per-call seconds; excludes model load and any prompt-length difference between the synthetic prompts and the real pairs.",
        "",
        "| judge | line item | calls | est. wall @ mean | est. wall @ p95 |",
        "|---|---|---|---|---|",
    ]
    for p in measured:
        tot = 0
        for name, n in CALLS.items():
            L.append(f"| {p['model']} | {name} | {n} | {minutes(n*p['mean_s_per_call'])} | {minutes(n*p['p95_s_per_call'])} |")
            tot += n
        L.append(f"| {p['model']} | **combined** | **{tot}** | **{minutes(tot*p['mean_s_per_call'])}** | **{minutes(tot*p['p95_s_per_call'])}** |")
    fits = [p for p in measured if p["model"] not in EXCLUDE_FROM_TOTALS]
    new = [p for p in fits if p["model"] in {r[k]["model"] for r in runs["new"].values() for k in r}]
    tot_calls = sum(CALLS.values())
    for label, ps in ((f"all {len(new)} new judges that fit", new), (f"all {len(fits)} fitting judges (incl. baseline)", fits)):
        L.append(
            f"| **{label}** | all sets | **{tot_calls*len(ps)}** | "
            f"**{minutes(sum(tot_calls*p['mean_s_per_call'] for p in ps))}** | **{minutes(sum(tot_calls*p['p95_s_per_call'] for p in ps))}** |"
        )
    L.append("")
    L.append(f"Totals exclude {', '.join(sorted(EXCLUDE_FROM_TOTALS))}: it does not fit in RAM and its disk-thrashing time is not a usable judge speed.")

    g = gpu
    L += [
        "",
        "## Rented-GPU estimate for a 14B-32B judge",
        "",
        f"**{g['label'].upper()}.**",
        "",
        "| judge | GPU | list price | calls | assumed s/call | est. compute | est. billed | est. cost |",
        "|---|---|---|---|---|---|---|---|",
        f"| {g['judge']} | {g['gpu']} | ${g['price_usd_per_hr']:.2f}/hr ([source]({g['price_source'].split(' ')[0]})) | {g['calls']} | "
        f"{g['assumed_s_per_call']:.1f} (assumed) | {g['est_compute_min']:.0f} min | ~{g['est_total_hr']:.2f} h incl. 45 min setup | ~${g['est_cost_usd']:.2f} + tax |",
        "",
        f"- Price source: {g['price_source']}.",
        f"- Throughput assumption: {g['throughput_assumption']}",
        f"- Overhead assumption: {g['overhead_assumption']}",
        "- Qwen3-14B (29.54 GB bf16) would also fit on the same card; the 32B line is the upper bound of the requested range.",
        "",
        "## Candidate screening (Hub metadata, not measurements)",
        "",
        "| model | revision | gated | bf16 safetensors GB | outcome |",
        "|---|---|---|---|---|",
    ]
    for model, rev, gated, gb, outcome in CANDIDATES:
        L.append(f"| {model} | `{rev[:12]}` | {gated} | {'n/a' if gb is None else f'{gb:.2f}'} | {outcome} |")

    d = DATASET
    L += [
        "",
        "## Notes",
        "",
        "- **Gating.** No HF token is present in this VM and none was used. Llama-3.2-3B returned 401 when tried in the first card; Gemma-3 is `gated: manual` and was skipped without an attempt.",
        "- **Fit: Qwen3-8B does not fit.** Its bf16 safetensors are 16.38 GB against 15.63 GiB total RAM. The load did not OOM because `low_cpu_mem_usage` keeps the weights mmap-backed; instead every forward pass re-streamed weights from disk (`vmstat`: ~570 MB/s reads, ~50% iowait; peak RSS 14.46 GB). The row above is that thrashing regime, measured in a single cold run (download + load = 14.69 s, no separate warm run; per-call times drifted from 46 s to 78 s as the page cache filled). It is reported for completeness, not as a usable judge speed. No 4-bit/8-bit CPU path was tried: bitsandbytes/torchao CPU quantisation is not a one-line change in this stack and was out of scope for a card.",
        "- **Thinking.** Qwen3-0.6B/1.7B/4B were run with `enable_thinking=False`; `raw.json` records the flag per run. Phi-4-mini has no thinking switch (`n/a`).",
        "- **Load time.** `from_pretrained(..., low_cpu_mem_usage=True)` memory-maps safetensors lazily, so warm load is mostly tokenizer + mmap setup and weights page in during the warmup call. Load is a one-off per judge and not the dominant cost.",
        "- **Cold-run outliers (baseline).** In the Phi-3.5 cold run the first two timed calls took ~11 s (page-in plus memory-balloon deflation), giving p95 = 11.1 s; the warm run has none. One warmup is not enough on a cold cache.",
        "- **Memory.** ~10 GiB of the 16 GiB was held by a hypervisor balloon at boot; it deflated under pressure and the ~4B bf16 judges (peak RSS 8.1-8.4 GB) loaded without OOM. Run judges sequentially.",
        "- **Parsing.** Verdict parsed from the first generated character. Phi-4-mini emitted `C (` and `A\\n\\n` within its 2 tokens; Phi-3.5 emitted `B\\n` and `A or`. These parse as the leading letter. The A/B/C log-prob readout is independent of this and is in `raw.json`.",
        "- **Prompt length.** Synthetic prompts averaged ~510-545 tokens, the low end of the 500-700 target. Real MT-Bench answers are often longer; CPU per-call time grows roughly linearly with prompt length, so treat the estimates as a floor.",
        f"- **MT-Bench human-judgment dataset.** `{d['repo']}` is {d['total_bytes']/1e6:.2f} MB total: `human` parquet ({d['files']['data/human-00000-of-00001-25f4910818759289.parquet']/1e3:.0f} KB) and `gpt4_pair` parquet ({d['files']['data/gpt4_pair-00000-of-00001-c0b431264a82ddc0.parquet']/1e3:.0f} KB), ungated. Checked via Hub metadata, not downloaded.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "pip install --index-url https://download.pytorch.org/whl/cpu torch && pip install 'transformers<5' accelerate psutil",
        "for m in Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B microsoft/Phi-4-mini-instruct; do",
        "  python order/cost_card/bench_judge.py $m ${m//\\//_}.json   # run twice for a warm run",
        "done",
        "python order/cost_card/make_card.py order/cost_card new/qwen3-0.6b:warm=Qwen_Qwen3-0.6B.json ...",
        "```",
        "",
    ]
    open(os.path.join(out_dir, "COST_CARD.md"), "w").write("\n".join(L))


if __name__ == "__main__":
    main()
