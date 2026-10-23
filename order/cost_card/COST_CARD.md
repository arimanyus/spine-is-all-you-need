# Order Is All You Need: judge cost card (draft)

CPU-only measurement of one pairwise LLM-judge call on this Cursor VM. Not the experiment.

## Machine

- CPU: Intel(R) Xeon(R) Processor, nproc = 4
- RAM: 15.63 GiB total (hypervisor memory balloon present; see note below)
- GPU: none (torch.cuda.is_available()=False, no nvidia-smi, no /dev/nvidia*)
- torch 2.14.1+cpu (CPU wheel), bf16 weights, greedy decoding, 4 threads

## Protocol

- 10 synthetic pairwise prompts (fixed system/user template, two ~150-250-word responses each), plus 1 untimed warmup call.
- Per call: one forward pass to read log-probs of tokens `A`, `B`, `C` at the verdict position, then `generate(max_new_tokens=2, do_sample=False)`.
- Peak RSS from `resource.getrusage(RUSAGE_SELF).ru_maxrss`.
- Each judge was run twice in a fresh process: `cold` (weights downloaded + loaded) and `warm` (weights already in the HF cache). The table reports the `warm` run; both are in `raw.json`.

## Per-call cost

| judge | params (B) | load s (cold incl. download / warm) | peak RSS GB | prompt tokens (mean) | mean s/call | p95 s/call | parse rate |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen2.5-1.5B-Instruct | 1.54 | 4.31 / 0.48 | 3.58 | 520 | 1.17 | 1.35 | 100% (10/10) |
| meta-llama/Llama-3.2-3B-Instruct | not measured | gated, not measured | not measured | not measured | not measured | not measured | not measured |
| microsoft/Phi-3.5-mini-instruct | 3.82 | 7.34 / 0.34 | 8.27 | 545 | 2.62 | 2.87 | 100% (10/10) |

## Extrapolation to the planned run

Two pair sets, each judged in both orders. Wall time = calls x measured per-call seconds; excludes model load and any prompt-length difference between the synthetic prompts and the real pairs.

| judge | line item | calls | est. wall @ mean | est. wall @ p95 |
|---|---|---|---|---|
| Qwen/Qwen2.5-1.5B-Instruct | Set M (MT-Bench pairs, ~80 x 2 orders) | 160 | 3.1 min | 3.6 min |
| Qwen/Qwen2.5-1.5B-Instruct | Set F (fresh pairs, ~40 x 2 orders) | 80 | 1.6 min | 1.8 min |
| Qwen/Qwen2.5-1.5B-Instruct | **combined** | **240** | **4.7 min** | **5.4 min** |
| microsoft/Phi-3.5-mini-instruct | Set M (MT-Bench pairs, ~80 x 2 orders) | 160 | 7.0 min | 7.7 min |
| microsoft/Phi-3.5-mini-instruct | Set F (fresh pairs, ~40 x 2 orders) | 80 | 3.5 min | 3.8 min |
| microsoft/Phi-3.5-mini-instruct | **combined** | **240** | **10.5 min** | **11.5 min** |
| **both measured judges** | all sets | **480** | **15.2 min** | **16.9 min** |

## Notes

- **meta-llama/Llama-3.2-3B-Instruct: gated, not measured.** The Hub returned 401 (`gated: manual`) with no HF token in this VM; per instructions no workaround was attempted. Its safetensors total 6.43 GB per Hub metadata. microsoft/Phi-3.5-mini-instruct (ungated) was used as the second judge instead.
- **Load time.** `from_pretrained(..., low_cpu_mem_usage=True)` memory-maps safetensors lazily, so the warm load number is mostly tokenizer + mmap setup; weights are paged in during the warmup call. The cold number includes the HF download on this VM's network. Neither is the dominant cost: model load is a one-off per judge.
- **Cold-run outliers.** In the Phi-3.5 cold run the first two timed calls took ~11 s (page-in plus memory-balloon deflation, see `raw.json`), giving p95 = 11.1 s. The warm run has no such outliers (max 2.87 s). One warmup call is not enough on a cold cache; budget 2-3 warmups or use the warm numbers.
- **Memory.** ~10 GiB of the 16 GiB was held by a hypervisor balloon at boot; it deflated under pressure and the 3.8B bf16 model (peak RSS 8.3 GB) loaded without OOM. Only one judge fits comfortably at a time; run them sequentially.
- **Parsing.** Verdict parsed from the first generated character (`A`/`B`/`C`). Phi-3.5 emitted `B\n` and once `A or` within its 2 tokens; these parse as B/A. The log-prob readout is independent of this and is in `raw.json`.
- **Prompt length.** Synthetic prompts averaged ~520-545 tokens, at the low end of the 500-700 target. Real MT-Bench answers are often longer; per-call time grows roughly linearly with prompt length on CPU, so treat the estimates as a floor.
- **MT-Bench human-judgment dataset.** `lmsys/mt_bench_human_judgments` is 1.39 MB total: two parquet files, `human` (739 KB) and `gpt4_pair` (650 KB), ungated. Checked via Hub metadata, not downloaded.

## Reproduce

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch && pip install 'transformers<5' accelerate psutil
python order/cost_card/bench_judge.py Qwen/Qwen2.5-1.5B-Instruct qwen.json
python order/cost_card/bench_judge.py microsoft/Phi-3.5-mini-instruct phi.json
python order/cost_card/make_card.py order/cost_card qwen:warm=qwen.json phi:warm=phi.json
```
