"""Assemble raw.json and COST_CARD.md from bench_judge.py outputs.

Usage: python make_card.py OUT_DIR run_label=path.json [...]
Labels are "<judge-key>:<run-name>", e.g. qwen:cold=/tmp/qwen.json qwen:warm=/tmp/qwen2.json.
The table uses the run named "warm" when present, otherwise the only run.
"""
import glob
import json
import os
import platform
import shutil
import sys

CALLS = {"Set M (MT-Bench pairs, ~80 x 2 orders)": 160, "Set F (fresh pairs, ~40 x 2 orders)": 80}
DATASET = {
    "repo": "lmsys/mt_bench_human_judgments",
    "files": {
        "data/gpt4_pair-00000-of-00001-c0b431264a82ddc0.parquet": 649996,
        "data/human-00000-of-00001-25f4910818759289.parquet": 738612,
    },
    "total_bytes": 1393033,
    "how_checked": "HfApi().dataset_info(files_metadata=True); not downloaded",
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


def fmt(x, nd=2):
    return "not measured" if x is None else f"{x:.{nd}f}"


def hhmm(sec):
    return f"{sec/60:.1f} min"


def main():
    out_dir = sys.argv[1]
    runs = {}
    for arg in sys.argv[2:]:
        label, path = arg.split("=", 1)
        key, run = label.split(":")
        runs.setdefault(key, {})[run] = json.load(open(path))

    raw = {"machine": machine(), "planned_calls_per_judge": CALLS, "mt_bench_dataset": DATASET, "judges": runs}
    json.dump(raw, open(os.path.join(out_dir, "raw.json"), "w"), indent=2)

    m = raw["machine"]
    lines = [
        "# Order Is All You Need: judge cost card (draft)",
        "",
        "CPU-only measurement of one pairwise LLM-judge call on this Cursor VM. Not the experiment.",
        "",
        "## Machine",
        "",
        f"- CPU: {m['cpu_model']}, nproc = {m['nproc']}",
        f"- RAM: {m['ram_total_gib']} GiB total (hypervisor memory balloon present; see note below)",
        f"- GPU: {m['gpu']}",
        f"- torch {m['torch']} (CPU wheel), bf16 weights, greedy decoding, {m['nproc']} threads",
        "",
        "## Protocol",
        "",
        "- 10 synthetic pairwise prompts (fixed system/user template, two ~150-250-word responses each), plus 1 untimed warmup call.",
        "- Per call: one forward pass to read log-probs of tokens `A`, `B`, `C` at the verdict position, then `generate(max_new_tokens=2, do_sample=False)`.",
        "- Peak RSS from `resource.getrusage(RUSAGE_SELF).ru_maxrss`.",
        "- Each judge was run twice in a fresh process: `cold` (weights downloaded + loaded) and `warm` (weights already in the HF cache). The table reports the `warm` run; both are in `raw.json`.",
        "",
        "## Per-call cost",
        "",
        "| judge | params (B) | load s (cold incl. download / warm) | peak RSS GB | prompt tokens (mean) | mean s/call | p95 s/call | parse rate |",
        "|---|---|---|---|---|---|---|---|",
    ]
    measured = {}
    for key, r in runs.items():
        pick = r.get("warm") or next(iter(r.values()))
        if pick["status"] != "ok":
            lines.append(f"| {pick['model']} | not measured | {pick['status']} | not measured | not measured | not measured | not measured | not measured |")
            continue
        cold = r.get("cold", {}).get("load_s")
        lines.append(
            f"| {pick['model']} | {pick['params_B']:.2f} | {fmt(cold)} / {fmt(pick['load_s'])} | {pick['peak_rss_gb']:.2f} | "
            f"{pick['prompt_tokens_mean']:.0f} | {pick['mean_s_per_call']:.2f} | {pick['p95_s_per_call']:.2f} | {pick['parse_rate']:.0%} ({pick['n_calls']}/{pick['n_calls']}) |"
        )
        measured[pick["model"]] = pick

    lines += [
        "",
        "## Extrapolation to the planned run",
        "",
        "Two pair sets, each judged in both orders. Wall time = calls x measured per-call seconds; excludes model load and any prompt-length difference between the synthetic prompts and the real pairs.",
        "",
        "| judge | line item | calls | est. wall @ mean | est. wall @ p95 |",
        "|---|---|---|---|---|",
    ]
    grand_mean = grand_p95 = 0.0
    for model, p in measured.items():
        tot = 0
        for name, n in CALLS.items():
            lines.append(f"| {model} | {name} | {n} | {hhmm(n*p['mean_s_per_call'])} | {hhmm(n*p['p95_s_per_call'])} |")
            tot += n
        lines.append(f"| {model} | **combined** | **{tot}** | **{hhmm(tot*p['mean_s_per_call'])}** | **{hhmm(tot*p['p95_s_per_call'])}** |")
        grand_mean += tot * p["mean_s_per_call"]
        grand_p95 += tot * p["p95_s_per_call"]
    tot_calls = sum(CALLS.values())
    lines.append(f"| **both measured judges** | all sets | **{tot_calls*len(measured)}** | **{hhmm(grand_mean)}** | **{hhmm(grand_p95)}** |")

    gated = [r for r in runs.values() for x in r.values() if x["status"] != "ok"]
    d = DATASET
    lines += [
        "",
        "## Notes",
        "",
        f"- **meta-llama/Llama-3.2-3B-Instruct: gated, not measured.** The Hub returned 401 (`gated: manual`) with no HF token in this VM; per instructions no workaround was attempted. Its safetensors total 6.43 GB per Hub metadata. microsoft/Phi-3.5-mini-instruct (ungated) was used as the second judge instead."
        if gated else "- All judges measured.",
        "- **Load time.** `from_pretrained(..., low_cpu_mem_usage=True)` memory-maps safetensors lazily, so the warm load number is mostly tokenizer + mmap setup; weights are paged in during the warmup call. The cold number includes the HF download on this VM's network. Neither is the dominant cost: model load is a one-off per judge.",
        "- **Cold-run outliers.** In the Phi-3.5 cold run the first two timed calls took ~11 s (page-in plus memory-balloon deflation, see `raw.json`), giving p95 = 11.1 s. The warm run has no such outliers (max 2.87 s). One warmup call is not enough on a cold cache; budget 2-3 warmups or use the warm numbers.",
        "- **Memory.** ~10 GiB of the 16 GiB was held by a hypervisor balloon at boot; it deflated under pressure and the 3.8B bf16 model (peak RSS 8.3 GB) loaded without OOM. Only one judge fits comfortably at a time; run them sequentially.",
        "- **Parsing.** Verdict parsed from the first generated character (`A`/`B`/`C`). Phi-3.5 emitted `B\\n` and once `A or` within its 2 tokens; these parse as B/A. The log-prob readout is independent of this and is in `raw.json`.",
        "- **Prompt length.** Synthetic prompts averaged ~520-545 tokens, at the low end of the 500-700 target. Real MT-Bench answers are often longer; per-call time grows roughly linearly with prompt length on CPU, so treat the estimates as a floor.",
        f"- **MT-Bench human-judgment dataset.** `{d['repo']}` is {d['total_bytes']/1e6:.2f} MB total: two parquet files, `human` ({d['files']['data/human-00000-of-00001-25f4910818759289.parquet']/1e3:.0f} KB) and `gpt4_pair` ({d['files']['data/gpt4_pair-00000-of-00001-c0b431264a82ddc0.parquet']/1e3:.0f} KB), ungated. Checked via Hub metadata, not downloaded.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "pip install --index-url https://download.pytorch.org/whl/cpu torch && pip install 'transformers<5' accelerate psutil",
        "python order/cost_card/bench_judge.py Qwen/Qwen2.5-1.5B-Instruct qwen.json",
        "python order/cost_card/bench_judge.py microsoft/Phi-3.5-mini-instruct phi.json",
        "python order/cost_card/make_card.py order/cost_card qwen:warm=qwen.json phi:warm=phi.json",
        "```",
        "",
    ]
    open(os.path.join(out_dir, "COST_CARD.md"), "w").write("\n".join(lines))


if __name__ == "__main__":
    main()
