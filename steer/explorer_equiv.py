"""HF-only check that our down_proj hook and dose units match lyramakesmusic/neuron-explorer (MIT).
Adapts the *steering* equivalence of their sweep/check_equivalence.py; the vLLM-vs-HF rollout and the
capture round-trip are skipped (no vLLM, no GPU, no captures here).

    steer/.venv/bin/python steer/explorer_equiv.py      # writes steer/raw_explorer_equiv.json

Confirms, on the HF path (Qwen3.5-4B, bf16):
  1. adding v to neuron n's activation (our forward-pre-hook on mlp.down_proj input) gives the same greedy
     rollout and the same MLP output as adding v * W_down[:, n] at the layer's MLP output (their `mlp_out` push).
  2. dose units: our steer.py `dose` equals their `--dose-unit rel` dose and their recorded `rel_push`:
       add_to_activation = dose * resid_p50[L] / ||W_down[L][:, n]||,
     so the added MLP-output vector has norm dose * resid_p50[L] (a fixed fraction `dose` of the median residual
     norm leaving layer L, the same for every neuron). Their absmax dose for the same push is
       dose_absmax = add / abs_max[L, n] = dose * resid_p50[L] / (||W_down[L][:, n]|| * abs_max[L, n]).
     abs_max is taken from their explorer/cache/act_scale.npz (max(pos, -neg)); steer_index/connections were
     built at rel dose 0.8.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from steer import MODEL_ID, REVISION, Harness, materialize  # noqa: E402

CACHE = Path("/tmp/neuron-explorer/explorer/cache")


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(CACHE))
    ap.add_argument("--calib", default=str(HERE / "calib_norms.json"))
    ap.add_argument("--out", default=str(HERE / "raw_explorer_equiv.json"))
    ap.add_argument("--neurons", default="10:3551,13:7956,14:8949", help="layer:neuron pairs to check")
    ap.add_argument("--doses", default="0.5,0.8,1.0")
    ap.add_argument("--new", type=int, default=32)
    ap.add_argument("--prompt", default="Who are you?")
    args = ap.parse_args()
    cache = Path(args.cache)
    resid = json.loads(Path(args.calib).read_text())["median_resid_norm"]
    asc = np.load(cache / "act_scale.npz")
    abs_max = np.maximum(asc["pos"], -asc["neg"])
    pairs = [(int(l), int(n)) for l, n in (p.split(":") for p in args.neurons.split(","))]
    doses = [float(x) for x in args.doses.split(",")]

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION, dtype=torch.bfloat16).eval()
    materialize(model)
    H = Harness(model, tok)
    ids = H.encode([{"role": "user", "content": args.prompt}])
    raw = {"meta": {"model_id": MODEL_ID, "revision": REVISION, "dtype": "bfloat16", "device": "cpu",
                    "explorer_repo": "https://github.com/lyramakesmusic/neuron-explorer (MIT)",
                    "adapted_from": "sweep/check_equivalence.py (HF path only; vLLM and capture checks skipped)",
                    "act_scale_sha256": hashlib.sha256((cache / "act_scale.npz").read_bytes()).hexdigest(),
                    "calib": args.calib, "prompt": args.prompt, "new_tokens": args.new,
                    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
           "hook_equivalence": [], "dose_conversion": [],
           "note": "our steer.py dose == explorer rel dose (--dose-unit rel) == explorer rel_push; "
                   "steer_index/connections were built at rel dose 0.8"}

    @torch.no_grad()
    def rollout(hook_factory):
        h = hook_factory()
        out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=args.new, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
        h.remove()
        return out[0, ids.shape[1]:].tolist()

    base = rollout(lambda: model.model.layers[0].register_forward_pre_hook(lambda m, a: None))
    for L, n in pairs:
        down = model.model.layers[L].mlp.down_proj
        col = down.weight[:, n].float()
        dnorm = float(col.norm())
        v = 1.0 * resid[L] / dnorm  # dose 1 add in activation units
        # input-side (our hook): add v to activation channel n
        in_ids = rollout(lambda: down.register_forward_pre_hook(
            lambda m, a: ((a[0].clone().index_add_(-1, torch.tensor([n]), torch.full_like(a[0][..., :1], v))),) + tuple(a[1:])))
        # output-side (their push): add v * W_down[:, n] at mlp out
        out_ids = rollout(lambda: down.register_forward_hook(lambda m, a, o: o + v * col.to(o.dtype)))
        # one-forward numeric check at the last token
        cap = {}
        hh = down.register_forward_hook(lambda m, a, o: cap.update(base_in=a[0][0, -1].clone(), base_out=o[0, -1].clone()))
        model(ids); hh.remove()
        added_in = torch.zeros_like(cap["base_in"].float()); added_in[n] = v
        expected = down.weight.float() @ (cap["base_in"].float() + added_in)
        got = down.weight.float() @ cap["base_in"].float() + v * col
        raw["hook_equivalence"].append({
            "layer": L, "neuron": n, "dose": 1.0, "add_to_activation": v, "down_col_norm": dnorm,
            "greedy_agree_input_vs_output_hook": common_prefix(in_ids, out_ids), "n_new": len(in_ids),
            "greedy_agree_input_vs_unsteered": common_prefix(in_ids, base),
            "added_vec_norm": float((v * col).norm()), "target_norm_dose1_x_resid": resid[L],
            "rel_err_inputformula_vs_outputformula_fp32": float((expected - got).norm() / got.norm())})
        print("hook_eq", raw["hook_equivalence"][-1], flush=True)
        for d in doses:
            add = d * resid[L] / dnorm
            raw["dose_conversion"].append({
                "layer": L, "neuron": n, "our_dose_rel": d, "add_to_activation": add,
                "added_mlp_out_norm": d * resid[L], "fraction_of_resid_p50": d,
                "explorer_rel_push": abs(add) * dnorm / resid[L], "explorer_dose_absmax": add / float(abs_max[L, n]),
                "abs_max": float(abs_max[L, n]), "resid_p50_mine": resid[L]})
    raw["meta"]["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    Path(args.out).write_text(json.dumps(raw, indent=1))
    print(json.dumps({"hook_equivalence": raw["hook_equivalence"], "dose_conversion_sample": raw["dose_conversion"][:3]}, indent=1))


if __name__ == "__main__":
    main()
