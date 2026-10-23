# Order Is All You Need: judge cost card (draft)

CPU-only measurement of one pairwise LLM-judge call on this Cursor VM. Not the experiment.

## Machine

- CPU: Intel(R) Xeon(R) Processor, nproc = 4
- RAM: 15.63 GiB total (hypervisor memory balloon present; see notes)
- GPU: none (torch.cuda.is_available()=False, no nvidia-smi, no /dev/nvidia*)
- torch 2.14.1+cpu (CPU wheel), transformers 4.57.6, bf16 weights, greedy decoding (temperature 0), 4 threads

## Protocol

- 10 synthetic pairwise prompts (fixed system/user template, two ~150-250-word responses each), plus 1 untimed warmup call.
- Per call: one forward pass to read log-probs of tokens `A`, `B`, `C` at the verdict position, then `generate(max_new_tokens=2, do_sample=False)`.
- Qwen3 hybrid-thinking models are run with `enable_thinking=False` in the chat template, so the first generated token is the verdict.
- Peak RSS from `resource.getrusage(RUSAGE_SELF).ru_maxrss`. Warm run = weights already in the local HF cache.
- Every number in the tables is copied from `raw.json`; cells that say *not measured* were not run and say why.

## Per-call cost (warm run)

| judge | HF revision | thinking | params (B) | load s | peak RSS GB | prompt tokens (mean) | mean s/call | p95 s/call | parse rate |
|---|---|---|---|---|---|---|---|---|---|
| Qwen/Qwen3-0.6B | `c1899de289a0` | off | 0.60 | 0.83 | 1.97 | 524 | 0.62 | 0.70 | 100% (10/10) |
| Qwen/Qwen3-1.7B | `70d244cc86cc` | off | 1.72 | 0.73 | 4.06 | 524 | 1.27 | 1.58 | 100% (10/10) |
| Qwen/Qwen3-4B | `1cfa9a720891` | off | 4.02 | 0.62 | 8.45 | 524 | 3.08 | 3.61 | 100% (10/10) |
| microsoft/Phi-4-mini-instruct | `cfbefacb9925` | n/a | 3.84 | 0.70 | 8.12 | 510 | 2.40 | 2.58 | 100% (10/10) |
| Qwen/Qwen3-8B (cold run only; does not fit, disk-thrashing) | `b968826d9c46` | off | 8.19 | 14.69 | 14.46 | 524 | 65.96 | 77.81 | 100% (10/10) |
| Qwen/Qwen2.5-1.5B-Instruct (baseline) | `989aa7980e4c` | n/a | 1.54 | 0.48 | 3.58 | 520 | 1.17 | 1.35 | 100% (10/10) |
| meta-llama/Llama-3.2-3B-Instruct (baseline) | n/a | n/a | not measured | gated, not measured | not measured | not measured | not measured | not measured | not measured |
| microsoft/Phi-3.5-mini-instruct (baseline) | `2fe192450127` | n/a | 3.82 | 0.34 | 8.27 | 545 | 2.62 | 2.87 | 100% (10/10) |

Baseline rows are the earlier card's judges; their revisions were recovered post hoc from the HF cache snapshot directory used by those runs (`revision_source` in `raw.json`). Their cold runs are also in `raw.json`.

## Extrapolation to the planned run

Two pair sets, each judged in both orders: Set M = 160 calls, Set F = 80 calls, 240 per judge. Wall time = calls x measured per-call seconds; excludes model load and any prompt-length difference between the synthetic prompts and the real pairs.

| judge | line item | calls | est. wall @ mean | est. wall @ p95 |
|---|---|---|---|---|
| Qwen/Qwen3-0.6B | Set M (MT-Bench pairs, ~80 x 2 orders) | 160 | 1.6 min | 1.9 min |
| Qwen/Qwen3-0.6B | Set F (fresh pairs, ~40 x 2 orders) | 80 | 0.8 min | 0.9 min |
| Qwen/Qwen3-0.6B | **combined** | **240** | **2.5 min** | **2.8 min** |
| Qwen/Qwen3-1.7B | Set M (MT-Bench pairs, ~80 x 2 orders) | 160 | 3.4 min | 4.2 min |
| Qwen/Qwen3-1.7B | Set F (fresh pairs, ~40 x 2 orders) | 80 | 1.7 min | 2.1 min |
| Qwen/Qwen3-1.7B | **combined** | **240** | **5.1 min** | **6.3 min** |
| Qwen/Qwen3-4B | Set M (MT-Bench pairs, ~80 x 2 orders) | 160 | 8.2 min | 9.6 min |
| Qwen/Qwen3-4B | Set F (fresh pairs, ~40 x 2 orders) | 80 | 4.1 min | 4.8 min |
| Qwen/Qwen3-4B | **combined** | **240** | **12.3 min** | **14.5 min** |
| microsoft/Phi-4-mini-instruct | Set M (MT-Bench pairs, ~80 x 2 orders) | 160 | 6.4 min | 6.9 min |
| microsoft/Phi-4-mini-instruct | Set F (fresh pairs, ~40 x 2 orders) | 80 | 3.2 min | 3.4 min |
| microsoft/Phi-4-mini-instruct | **combined** | **240** | **9.6 min** | **10.3 min** |
| Qwen/Qwen3-8B | Set M (MT-Bench pairs, ~80 x 2 orders) | 160 | 175.9 min | 207.5 min |
| Qwen/Qwen3-8B | Set F (fresh pairs, ~40 x 2 orders) | 80 | 87.9 min | 103.7 min |
| Qwen/Qwen3-8B | **combined** | **240** | **263.8 min** | **311.2 min** |
| Qwen/Qwen2.5-1.5B-Instruct | Set M (MT-Bench pairs, ~80 x 2 orders) | 160 | 3.1 min | 3.6 min |
| Qwen/Qwen2.5-1.5B-Instruct | Set F (fresh pairs, ~40 x 2 orders) | 80 | 1.6 min | 1.8 min |
| Qwen/Qwen2.5-1.5B-Instruct | **combined** | **240** | **4.7 min** | **5.4 min** |
| microsoft/Phi-3.5-mini-instruct | Set M (MT-Bench pairs, ~80 x 2 orders) | 160 | 7.0 min | 7.7 min |
| microsoft/Phi-3.5-mini-instruct | Set F (fresh pairs, ~40 x 2 orders) | 80 | 3.5 min | 3.8 min |
| microsoft/Phi-3.5-mini-instruct | **combined** | **240** | **10.5 min** | **11.5 min** |
| **all 4 new judges that fit** | all sets | **960** | **29.5 min** | **33.9 min** |
| **all 6 fitting judges (incl. baseline)** | all sets | **1440** | **44.6 min** | **50.8 min** |

Totals exclude Qwen/Qwen3-8B: it does not fit in RAM and its disk-thrashing time is not a usable judge speed.

## Rented-GPU estimate for a 14B-32B judge

**LIST-PRICE ESTIMATE, NOTHING PAID, NOT MEASURED.**

| judge | GPU | list price | calls | assumed s/call | est. compute | est. billed | est. cost |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen3-32B (ungated, rev 9216db5781bf21249d130ec9da846c4624c16137, 65.52 GB bf16 safetensors) | 1x NVIDIA H100 PCIe 80 GB (Lambda on-demand) | $3.29/hr ([source](https://lambda.ai/service/gpu-cloud)) | 480 | 1.0 (assumed) | 8 min | ~0.88 h incl. 45 min setup | ~$2.90 + tax |

- Price source: https://lambda.ai/service/gpu-cloud (1x table, 'NVIDIA H100 PCIe', $3.29/GPU/hr plus tax; fetched 2026-10-23).
- Throughput assumption: Assumed 1.0 s/call as a conservative ceiling: 600-token prefill on a 32B dense bf16 model is ~3.8e13 FLOP, ~0.1 s at 35% of H100 bf16 peak; 2 decode tokens are memory-bound, ~0.04 s at 3.35 TB/s over 65 GB. 1 s/call leaves ~5x margin for an unoptimised HF transformers loop. Not benchmarked.
- Overhead assumption: Instance boot + 65 GB weight download + load budgeted at 45 min; billing is per minute.
- Qwen3-14B (29.54 GB bf16) would also fit on the same card; the 32B line is the upper bound of the requested range.

## Candidate screening (Hub metadata, not measurements)

| model | revision | gated | bf16 safetensors GB | outcome |
|---|---|---|---|---|
| Qwen/Qwen3-0.6B | `c1899de289a0` | False | 1.50 | measured |
| Qwen/Qwen3-1.7B | `70d244cc86cc` | False | 4.06 | measured |
| Qwen/Qwen3-4B | `1cfa9a720891` | False | 8.04 | measured |
| microsoft/Phi-4-mini-instruct | `cfbefacb9925` | False | 7.67 | measured |
| Qwen/Qwen3-8B | `b968826d9c46` | False | 16.38 | does not fit in RAM; measured anyway via mmap thrashing (see Fit note) |
| Qwen/Qwen3-4B-Instruct-2507 | `cdbee75f17c0` | False | 8.04 | not measured: optional; same size class as Qwen3-4B |
| Qwen/Qwen3.5-0.8B / 2B / 4B | `main` | False | n/a | not measured: `qwen3_5` multimodal architecture needs transformers>=5; this repo pins <5 |
| HuggingFaceTB/SmolLM3-3B | `a07cc9a04f16` | False | 6.15 | not measured: optional |
| google/gemma-3-4b-it | `093f9f388b31` | manual | 8.60 | gated, skipped |
| meta-llama/Llama-3.2-3B-Instruct | `0cb88a4f764b` | manual | 6.43 | gated, not measured (401 without token) |

## Notes

- **Gating.** No HF token is present in this VM and none was used. Llama-3.2-3B returned 401 when tried in the first card; Gemma-3 is `gated: manual` and was skipped without an attempt.
- **Fit: Qwen3-8B does not fit.** Its bf16 safetensors are 16.38 GB against 15.63 GiB total RAM. The load did not OOM because `low_cpu_mem_usage` keeps the weights mmap-backed; instead every forward pass re-streamed weights from disk (`vmstat`: ~570 MB/s reads, ~50% iowait; peak RSS 14.46 GB). The row above is that thrashing regime, measured in a single cold run (download + load = 14.69 s, no separate warm run; per-call times drifted from 46 s to 78 s as the page cache filled). It is reported for completeness, not as a usable judge speed. No 4-bit/8-bit CPU path was tried: bitsandbytes/torchao CPU quantisation is not a one-line change in this stack and was out of scope for a card.
- **Thinking.** Qwen3-0.6B/1.7B/4B were run with `enable_thinking=False`; `raw.json` records the flag per run. Phi-4-mini has no thinking switch (`n/a`).
- **Load time.** `from_pretrained(..., low_cpu_mem_usage=True)` memory-maps safetensors lazily, so warm load is mostly tokenizer + mmap setup and weights page in during the warmup call. Load is a one-off per judge and not the dominant cost.
- **Cold-run outliers (baseline).** In the Phi-3.5 cold run the first two timed calls took ~11 s (page-in plus memory-balloon deflation), giving p95 = 11.1 s; the warm run has none. One warmup is not enough on a cold cache.
- **Memory.** ~10 GiB of the 16 GiB was held by a hypervisor balloon at boot; it deflated under pressure and the ~4B bf16 judges (peak RSS 8.1-8.4 GB) loaded without OOM. Run judges sequentially.
- **Parsing.** Verdict parsed from the first generated character. Phi-4-mini emitted `C (` and `A\n\n` within its 2 tokens; Phi-3.5 emitted `B\n` and `A or`. These parse as the leading letter. The A/B/C log-prob readout is independent of this and is in `raw.json`.
- **Prompt length.** Synthetic prompts averaged ~510-545 tokens, the low end of the 500-700 target. Real MT-Bench answers are often longer; CPU per-call time grows roughly linearly with prompt length, so treat the estimates as a floor.
- **MT-Bench human-judgment dataset.** `lmsys/mt_bench_human_judgments` is 1.39 MB total: `human` parquet (739 KB) and `gpt4_pair` parquet (650 KB), ungated. Checked via Hub metadata, not downloaded.

## Reproduce

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch && pip install 'transformers<5' accelerate psutil
for m in Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B microsoft/Phi-4-mini-instruct; do
  python order/cost_card/bench_judge.py $m ${m//\//_}.json   # run twice for a warm run
done
python order/cost_card/make_card.py order/cost_card new/qwen3-0.6b:warm=Qwen_Qwen3-0.6B.json ...
```
