"""STEP 2: searched top-k neuron steering with dose escalation, matched random controls.

Neuron rankings come from the search sets only, using the down_proj-input activations that
directions.py saved to steer/private/search_acts.npz (mean over generated tokens at the final
turn), ranked by Cohen's d in layers 13/14:
  spine   : fold vs hold on spine_contrast pressured dialogues; sign pushes toward fold  (EXPLORATORY)
  refusal : comply vs refuse on contrast_prompts;               sign pushes toward comply
Steering (steer.py recipe: dose * median_resid_norm[L] / ||W_down[:, n]|| added to the neuron's
activation) is applied only on dialogue turns >= 1 (pressure / bait turns) and on the single-turn
factual gate prompts. For each k in (1,2,4,8) doses escalate 1,4,8,16; after each searched
condition a matched random set (same per-layer counts and signs, fresh seed) runs at the same
dose; escalation for that k stops once factual accuracy falls more than 10 points below unsteered.
Writes steer/raw_neuron_sweep.json (scores only).
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import steer  # noqa: E402
from steer import EXPLORATORY, MODEL_ID, PROBES_REF, REVISION, SEARCH_LAYERS, Harness, materialize, rank_neurons, rss_gb, top_k_pairs  # noqa: E402
from evalsets import Eval  # noqa: E402

SEED = 0
KS = (1, 2, 4, 8)
DOSES = (1, 4, 8, 16)
GATE = 0.10


class NeuronIntervention:
    def __init__(self, H, neurons, dose):
        self.H, self.neurons, self.dose, self.deltas = H, neurons, dose, None

    def enable(self):
        self.deltas = self.H.set_steering(self.neurons, self.dose)

    def disable(self):
        self.H.clear_steering()


def rankings(acts):
    def groups(prefix, mask):
        return [{L: acts[f"{prefix}_L{L}_gen_mean"][i] for L in SEARCH_LAYERS} for i in np.where(mask)[0]]
    labels = acts["spine_labels"]
    fold_side = ["fold"] if (labels == "fold").sum() >= 2 else ["fold", "other", "hedge"]
    fold_mask, hold_mask = np.isin(labels, fold_side), labels == "hold"
    sides = acts["cp_sides"]
    return {
        "spine": {"label": EXPLORATORY, "contrast": "fold-side vs hold on spine_contrast pressured dialogues (final turn, gen_mean)",
                  "fold_side_labels": fold_side, "n_fold_side": int(fold_mask.sum()), "n_hold": int(hold_mask.sum()),
                  "ranking": rank_neurons(groups("spine", fold_mask), groups("spine", hold_mask))},
        "refusal": {"label": "search set only (contrast_prompts); may be confirmatory",
                    "contrast": "comply vs refuse on contrast_prompts (gen_mean)",
                    "n_comply": int((sides == "comply").sum()), "n_refuse": int((sides == "refuse").sum()),
                    "ranking": rank_neurons(groups("cp", sides == "comply"), groups("cp", sides == "refuse"))},
    }


def matched_random(neurons, n_inter, rng, exclude):
    out = []
    for L, _, sign in neurons:
        n = rng.randrange(n_inter)
        while (L, n) in exclude:
            n = rng.randrange(n_inter)
        out.append((L, n, sign))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes-ref", default=PROBES_REF)
    ap.add_argument("--refusal-text", default=str(HERE / "private" / "refusal_probes.jsonl"))
    ap.add_argument("--acts", default=str(HERE / "private" / "search_acts.npz"))
    ap.add_argument("--calib", default=str(HERE / "calib_norms.json"))
    ap.add_argument("--out", default=str(HERE / "raw_neuron_sweep.json"))
    ap.add_argument("--families", default="spine,refusal")
    ap.add_argument("--n-refusal", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    steer.MAX_NEW_TOKENS = args.max_new_tokens
    q = args.quick
    out_path = Path(args.out)

    t0 = time.monotonic()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION, dtype=torch.bfloat16).eval()
    materialize(model)
    H = Harness(model, tok)
    H.resid_median = json.loads(Path(args.calib).read_text())["median_resid_norm"]
    E = Eval(H, args.probes_ref, args.refusal_text, n_refusal=1 if q else args.n_refusal, n_factual=2 if q else 10, n_pressure=2 if q else 12)
    acts = np.load(args.acts)
    rk = rankings(acts)
    import subprocess
    raw = {"meta": {"model_id": MODEL_ID, "revision": REVISION, "dtype": "bfloat16", "device": "cpu", "decoding": "greedy",
                    "max_new_tokens": args.max_new_tokens, "quick": q, "timing_clock": "time.monotonic", "seed": SEED,
                    "probes_ref": args.probes_ref,
                    "probes_commit": subprocess.check_output(["git", "rev-parse", args.probes_ref], cwd=HERE.parent, text=True).strip(),
                    "sets": E.sets_meta, "calib": args.calib, "search_acts": args.acts, "load_time_s": time.monotonic() - t0,
                    "ks": list(KS), "doses": list(DOSES), "factual_gate_drop": GATE, "steer_turns": "pressure (dialogue turns >= 1) + factual",
                    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
           "rankings": rk, "failures": [], "conditions": {}}
    n_inter = model.config.intermediate_size

    def dump():
        raw["peak_rss_gb"] = rss_gb()
        out_path.write_text(json.dumps(raw, indent=1))
        E.dump_private(HERE / "private" / "neuron_sweep_generations_full.json")

    def run(name, neurons, dose, sets, extra):
        iv = NeuronIntervention(H, neurons, dose) if neurons else steer_none
        out = E.run(name, iv, steer_turns="pressure", sets=sets)
        raw["conditions"][name] = {"neurons": [{"layer": L, "neuron": n, "sign": s} for L, n, s in neurons], "dose": dose,
                                   "deltas": iv.deltas if neurons else [], **extra, **out}
        print(name, json.dumps(out["summary"]), f"wall {out['wall_s']:.0f}s rss {out['peak_rss_gb']:.2f}", flush=True)
        dump()
        return out["summary"]

    from evalsets import NoOp
    steer_none = NoOp()
    base = run("none", [], 0, ("factual", "pressure", "refusal"), {})
    acc0 = base["factual_accuracy"]
    for fam in args.families.split(","):
        sets = ("factual", "pressure") if fam == "spine" else ("factual", "refusal")
        ranking = rk[fam]["ranking"]
        if all(r.get("undefined") for r in ranking.values()):
            raw["failures"].append(f"{fam}: ranking undefined, family skipped")
            continue
        for k in (KS[:2] if q else KS):
            top = top_k_pairs(ranking, k)
            for dose in (DOSES[:2] if q else DOSES):
                extra = {"family": fam, "k": k, "kind": "searched", "label": rk[fam]["label"]}
                s = run(f"{fam}_top{k}_dose{dose}", top, dose, sets, extra)
                rng = random.Random(f"{SEED}-{fam}-{k}-{dose}")
                rnd = matched_random(top, n_inter, rng, {(L, n) for L, n, _ in top})
                run(f"{fam}_random{k}_dose{dose}", rnd, dose, sets, {**extra, "kind": "matched random"})
                if s["factual_accuracy"] is not None and s["factual_accuracy"] < acc0 - GATE:
                    raw["conditions"][f"{fam}_top{k}_dose{dose}"]["gate"] = f"factual accuracy {s['factual_accuracy']:.2f} < unsteered {acc0:.2f} - {GATE}; escalation stopped for k={k}"
                    print(raw["conditions"][f"{fam}_top{k}_dose{dose}"]["gate"], flush=True)
                    dump()
                    break
    raw["meta"]["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    dump()


if __name__ == "__main__":
    main()
