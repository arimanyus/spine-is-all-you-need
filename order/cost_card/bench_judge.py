"""Measure the CPU cost of one pairwise LLM-judge call for a given HF model.

Usage: python bench_judge.py MODEL_ID OUT_JSON [--n 10] [--mock]

Per call: one forward pass over the prompt to read logprobs for A/B/C at the
verdict position, then greedy generation of up to 2 tokens. Writes a JSON
record; a load failure (e.g. gated repo) is recorded instead of raised.
"""
import argparse
import json
import os
import platform
import resource
import statistics
import sys
import time

SYSTEM = (
    "You are an impartial judge. Compare two responses to the user's question. "
    "Judge correctness first, then helpfulness. Do not let response order or "
    "length influence you."
)
USER_TMPL = (
    "[Question]\n{q}\n\n[Response A]\n{a}\n\n[Response B]\n{b}\n\n"
    "Output only one token: A, B, or C (tie)."
)

QUESTIONS = [
    "Why does ice float on liquid water?",
    "What is the difference between a process and a thread?",
    "How does compound interest work?",
    "Why is the sky blue during the day but red at sunset?",
    "What causes the seasons on Earth?",
    "How does a hash table achieve O(1) average lookup?",
    "Why do we need vaccines if the immune system already fights infections?",
    "What is the purpose of a database index?",
    "How does public-key cryptography let two strangers communicate securely?",
    "Why does bread rise when you add yeast?",
]

# Filler sentences used to pad each synthetic response to ~150-250 words.
FILLER = [
    "This matters in practice because the underlying mechanism shows up in many everyday situations, "
    "and understanding it helps you predict what will happen in related cases.",
    "A common misconception is to attribute the effect to a single cause, when in fact several "
    "factors interact and the dominant one depends on the specific conditions.",
    "To make this concrete, consider a simple example where only one variable changes while "
    "everything else is held fixed; the outcome then follows directly from the principle above.",
    "Textbooks usually introduce the idea with an idealised model first and only later add the "
    "complications that appear in real systems, which is a reasonable way to learn it.",
    "If you want to verify this yourself, there are straightforward experiments or measurements "
    "that demonstrate the effect clearly without specialised equipment.",
    "Historically, the explanation was worked out over several decades as better measurements "
    "ruled out competing hypotheses one by one.",
    "In summary, the key point is the relationship between the quantities involved, and once "
    "that relationship is clear the rest of the behaviour follows naturally.",
]


def synthetic_response(idx: int, variant: int, target_words: int) -> str:
    q = QUESTIONS[idx]
    opener = (
        f"The short answer to '{q}' is that it comes down to a well-understood physical or "
        f"logical principle, which I will explain step by step."
        if variant == 0
        else f"Good question. '{q}' is best answered by first stating the principle and then "
        f"walking through why it leads to the observed behaviour."
    )
    words = opener.split()
    i = (idx + variant) % len(FILLER)
    while len(words) < target_words:
        words += FILLER[i % len(FILLER)].split()
        i += 1
    return " ".join(words[:target_words])


def build_prompts(n: int):
    prompts = []
    for k in range(n):
        idx = k % len(QUESTIONS)
        a = synthetic_response(idx, 0, 150 + (k * 37) % 101)  # 150-250 words
        b = synthetic_response(idx, 1, 150 + (k * 53) % 101)
        prompts.append(
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER_TMPL.format(q=QUESTIONS[idx], a=a, b=b)},
            ]
        )
    return prompts


def peak_rss_gb() -> float:
    # ru_maxrss is KiB on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def parse_verdict(text: str):
    t = text.strip().upper()
    for tok in ("A", "B", "C"):
        if t.startswith(tok) and (len(t) == 1 or not t[1].isalpha()):
            return tok
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    args = ap.parse_args()

    rec = {
        "model": args.model,
        "n_calls": args.n,
        "threads": args.threads,
        "dtype": "bfloat16",
        "device": "cpu",
        "python": platform.python_version(),
        "status": "ok",
    }

    import torch

    torch.set_num_threads(args.threads)
    rec["torch"] = torch.__version__
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers

    rec["transformers"] = transformers.__version__

    t0 = time.perf_counter()
    try:
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
        model.eval()
    except Exception as e:  # noqa: BLE001 - we record, not raise
        msg = f"{type(e).__name__}: {e}"
        rec["status"] = "gated, not measured" if ("gated" in msg.lower() or "401" in msg or "403" in msg) else "load_failed"
        rec["error"] = msg[:2000]
        json.dump(rec, open(args.out, "w"), indent=2)
        print(rec["status"], "-", msg[:300])
        return
    rec["load_s"] = round(time.perf_counter() - t0, 2)
    rec["revision"] = getattr(model.config, "_commit_hash", None)
    rec["params_B"] = round(sum(p.numel() for p in model.parameters()) / 1e9, 3)
    rec["rss_after_load_gb"] = round(peak_rss_gb(), 3)

    # Qwen3-style hybrid models: disable thinking so the verdict is the first generated token.
    template_kwargs = {}
    if tok.chat_template and "enable_thinking" in tok.chat_template:
        template_kwargs["enable_thinking"] = False
    rec["enable_thinking"] = template_kwargs.get("enable_thinking", "n/a (template has no thinking switch)")

    # Token ids for the verdict letters (first token of " A"/"A" etc.).
    letter_ids = {}
    for L in ("A", "B", "C"):
        ids = tok.encode(L, add_special_tokens=False)
        letter_ids[L] = ids[0]
    rec["letter_token_ids"] = letter_ids

    prompts = build_prompts(args.n + 1)  # +1 for the warmup prompt
    enc = [
        tok.apply_chat_template(p, add_generation_prompt=True, return_tensors="pt", **template_kwargs)
        for p in prompts
    ]
    rec["prompt_tokens_mean"] = round(statistics.mean(e.shape[1] for e in enc[1:]), 1)
    rec["prompt_tokens_min"] = min(e.shape[1] for e in enc[1:])
    rec["prompt_tokens_max"] = max(e.shape[1] for e in enc[1:])

    @torch.inference_mode()
    def judge(ids):
        out = model(ids)
        logits = out.logits[0, -1].float()
        logprobs = torch.log_softmax(logits, dim=-1)
        lp = {L: round(logprobs[i].item(), 4) for L, i in letter_ids.items()}
        gen = model.generate(
            ids, max_new_tokens=2, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id
        )
        text = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
        return lp, text

    judge(enc[0])  # warmup, not timed
    times, calls = [], []
    for ids in enc[1:]:
        t = time.perf_counter()
        lp, text = judge(ids)
        dt = time.perf_counter() - t
        times.append(dt)
        v = parse_verdict(text)
        calls.append({"s": round(dt, 3), "prompt_tokens": ids.shape[1], "logprobs": lp, "output": text, "parsed": v})

    times_sorted = sorted(times)
    rec["mean_s_per_call"] = round(statistics.mean(times), 3)
    rec["p95_s_per_call"] = round(times_sorted[min(len(times) - 1, int(round(0.95 * (len(times) - 1))))], 3)
    rec["min_s_per_call"] = round(times_sorted[0], 3)
    rec["max_s_per_call"] = round(times_sorted[-1], 3)
    rec["parse_rate"] = sum(c["parsed"] is not None for c in calls) / len(calls)
    rec["peak_rss_gb"] = round(peak_rss_gb(), 3)
    rec["calls"] = calls
    json.dump(rec, open(args.out, "w"), indent=2)
    print(json.dumps({k: v for k, v in rec.items() if k != "calls"}, indent=2))


if __name__ == "__main__":
    main()
