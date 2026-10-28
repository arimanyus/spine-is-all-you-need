"""STEP 1: direction ablation (Arditi et al. 2024, "Refusal in LMs is mediated by a single direction")
on Qwen3.5-4B, CPU bf16, greedy. Writes steer/raw_directions.json (scores only) and private artifacts.

Directions are difference-of-means of the residual stream at the answer position (last prompt token,
after the chat template's empty <think> block), taken at each decoder layer output:
  refusal direction r_L = mean_refuse(h_L) - mean_comply(h_L) on contrast_prompts (search set only;
      refuse-side text resolved from a local gitignored file). Layer chosen by held-out separation:
      pairs are split 10 train / 9 held-out (seed 0); for each L the train direction is scored by
      Cohen's d of the held-out refuse vs comply projections; the best L is refit on all pairs.
  spine direction s_L = mean(hold) - mean(fold) on spine_contrast pressured dialogues (search set
      only), hold/fold from scorers.spine.spine_label on the unsteered final-turn reply. If fewer than
      2 folds, the fold side widens to {fold, other, hedge} (flagged). Layer chosen the same way when
      each side has >= 4 items, else in-sample (flagged).
Interventions (all tokens):
  ablate: h <- h - (h . u) u at the embedding output and every decoder layer output, u = unit direction.
  subtract: h <- h - v at the chosen layer's output, v = the unnormalised difference-of-means.
Controls: unsteered; a random unit direction (seed 0) ablated; the same random direction scaled to
||r|| / ||s|| subtracted at the respective chosen layer.
Spine-side directions are EXPLORATORY (template-family overlap, see LEAKAGE.md).
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
from steer import EXPLORATORY, MODEL_ID, PROBES_REF, REVISION, Harness, load_jsonl, load_probe, materialize, rss_gb  # noqa: E402
from steer import SEARCH_LAYERS, crude_refused  # noqa: E402
from evalsets import Eval  # noqa: E402
from scorers.spine import spine_label  # noqa: E402

SEED = 0


def cohens_d(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float((a.mean() - b.mean()) / (np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) + 1e-9))


class Resid:
    """Residual-stream capture and interventions. Index 0 = embedding output, i+1 = decoder layer i output."""

    def __init__(self, H):
        self.H, self.handles = H, []
        self.points = [H.model.model.embed_tokens] + list(H.layers)

    @staticmethod
    def _out(out):
        return out[0] if isinstance(out, tuple) else out

    @staticmethod
    def _wrap(out, h):
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    @torch.no_grad()
    def last_token(self, ids):
        """[n_points, hidden] float32 residual at the last position of a prefill forward."""
        buf = [None] * len(self.points)

        def mk(i):
            def hook(_m, _a, out):
                buf[i] = self._out(out)[0, -1].float().numpy().copy()
            return hook
        hs = [p.register_forward_hook(mk(i)) for i, p in enumerate(self.points)]
        self.H.model(ids)
        for h in hs:
            h.remove()
        return np.stack(buf)

    def ablate(self, unit):
        u = torch.tensor(unit, dtype=torch.float32)

        def hook(_m, _a, out):
            h = self._out(out)
            hf = h.float()
            return self._wrap(out, (hf - (hf @ u)[..., None] * u).to(h.dtype))
        self.handles = [p.register_forward_hook(hook) for p in self.points]

    def subtract(self, layer, vec):
        v = torch.tensor(vec, dtype=torch.float32)

        def hook(_m, _a, out):
            h = self._out(out)
            return self._wrap(out, (h.float() - v).to(h.dtype))
        self.handles = [self.H.layers[layer].register_forward_hook(hook)]

    def clear(self):
        for h in self.handles:
            h.remove()
        self.handles = []


class Intervention:
    def __init__(self, resid, kind=None, **kw):
        self.resid, self.kind, self.kw = resid, kind, kw

    def enable(self):
        self.resid.clear()
        if self.kind == "ablate":
            self.resid.ablate(self.kw["unit"])
        elif self.kind == "subtract":
            self.resid.subtract(self.kw["layer"], self.kw["vec"])

    def disable(self):
        self.resid.clear()


def pick_layer(acts_a, acts_b, train_idx_a, train_idx_b, held_a, held_b):
    """acts_*: [n, n_points, hidden]. Returns (best decoder layer, per-layer held-out d, unit dirs)."""
    curve = []
    for p in range(1, acts_a.shape[1]):  # skip embedding output (p=0)
        d = acts_a[train_idx_a, p].mean(0) - acts_b[train_idx_b, p].mean(0)
        u = d / (np.linalg.norm(d) + 1e-9)
        curve.append(cohens_d(acts_a[held_a, p] @ u, acts_b[held_b, p] @ u))
    best = int(np.nanargmax(curve)) if not np.all(np.isnan(curve)) else 0
    return best, curve  # best = decoder layer index (point best+1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes-ref", default=PROBES_REF)
    ap.add_argument("--refusal-text", default=str(HERE / "private" / "refusal_probes.jsonl"))
    ap.add_argument("--contrast-text", default=str(HERE / "private" / "contrast_prompts.jsonl"))
    ap.add_argument("--n-refusal", type=int, default=8)
    ap.add_argument("--out", default=str(HERE / "raw_directions.json"))
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--quick", action="store_true", help="tiny smoke of the pipeline (few items)")
    ap.add_argument("--conditions", default="none,refusal_ablate,refusal_subtract,random_ablate,random_subtract_refnorm,"
                                            "spine_ablate,spine_subtract,random_subtract_spinenorm")
    args = ap.parse_args()
    import steer
    steer.MAX_NEW_TOKENS = args.max_new_tokens
    q = args.quick
    private = HERE / "private"
    private.mkdir(exist_ok=True)
    out_path = Path(args.out)

    t0 = time.monotonic()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION, dtype=torch.bfloat16).eval()
    materialize(model)
    H, R = Harness(model, tok), None
    R = Resid(H)
    E = Eval(H, args.probes_ref, args.refusal_text, n_refusal=1 if q else args.n_refusal, n_factual=2 if q else 10, n_pressure=2 if q else 12)
    import subprocess
    raw = {"meta": {"model_id": MODEL_ID, "revision": REVISION, "dtype": "bfloat16", "device": "cpu", "decoding": "greedy",
                    "max_new_tokens": args.max_new_tokens, "quick": q, "timing_clock": "time.monotonic", "seed": SEED,
                    "probes_ref": args.probes_ref,
                    "probes_commit": subprocess.check_output(["git", "rev-parse", args.probes_ref], cwd=HERE.parent, text=True).strip(),
                    "sets": E.sets_meta, "load_time_s": time.monotonic() - t0,
                    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
           "failures": [], "directions": {}, "conditions": {}}

    def dump():
        raw["peak_rss_gb"] = rss_gb()
        out_path.write_text(json.dumps(raw, indent=1))
        E.dump_private(private / "directions_generations_full.json")

    # ---------------- refusal direction (contrast_prompts) ----------------
    cp = load_jsonl(args.contrast_text) if Path(args.contrast_text).exists() else load_probe(args.probes_ref, "contrast_prompts")
    by_id = {r["id"]: r for r in cp}
    pairs = [(f"cp-{i:03d}", f"cp-{i + 1:03d}") for i in range(1, 41, 2)]
    pairs = [(a, b) for a, b in pairs if "prompt" in by_id.get(a, {}) and "prompt" in by_id.get(b, {})][:4 if q else None]
    raw["directions"]["refusal"] = {"source": "contrast_prompts (search set only)", "n_pairs_with_text": len(pairs),
                                    "pairs_dropped_no_refuse_text": [a for a, b in [(f"cp-{i:03d}", f"cp-{i + 1:03d}") for i in range(1, 41, 2)]
                                                                     if "prompt" not in by_id.get(a, {})]}
    print(f"refusal contrast: {len(pairs)} pairs with text", flush=True)
    acts_ref, acts_com, cp_gen, cp_down = [], [], {}, {}
    for a, b in pairs:
        for cid, bucket in ((a, acts_ref), (b, acts_com)):
            hist = [{"role": "user", "content": by_id[cid]["prompt"]}]
            bucket.append(R.last_token(H.encode(hist)))
            H.start_recording()
            g = H.generate(hist)
            cp_down[cid] = H.stop_recording()
            cp_gen[cid] = {"side": by_id[cid]["side"], "refused_crude": crude_refused(g["text"]), "n_tokens": g["n_tokens"],
                           "seconds": g["seconds"]}
            E.private.setdefault("search_contrast", {})[cid] = g["text"]
    acts_ref, acts_com = np.stack(acts_ref), np.stack(acts_com)
    rng = random.Random(SEED)
    idx = list(range(len(pairs))); rng.shuffle(idx)
    tr, ho = idx[:len(idx) // 2 + 1], idx[len(idx) // 2 + 1:]
    best_L, curve = pick_layer(acts_ref, acts_com, tr, tr, ho, ho)
    r_vec = acts_ref[:, best_L + 1].mean(0) - acts_com[:, best_L + 1].mean(0)
    r_unit = r_vec / np.linalg.norm(r_vec)
    raw["directions"]["refusal"].update({
        "unsteered_refusal_rate_refuse_side_crude": float(np.mean([g["refused_crude"] for g in cp_gen.values() if g["side"] == "refuse"])),
        "unsteered_refusal_rate_comply_side_crude": float(np.mean([g["refused_crude"] for g in cp_gen.values() if g["side"] == "comply"])),
        "per_prompt": cp_gen, "split": {"train_pairs": len(tr), "heldout_pairs": len(ho), "seed": SEED},
        "heldout_d_per_layer": curve, "chosen_layer": best_L, "direction_norm": float(np.linalg.norm(r_vec)),
        "heldout_d_at_chosen": curve[best_L],
        "insample_d_at_chosen": cohens_d(acts_ref[:, best_L + 1] @ r_unit, acts_com[:, best_L + 1] @ r_unit),
        "label": "search set only; may be confirmatory (contrast_prompts shares no template/family with refusal_probes)"})
    print(f"refusal direction: layer {best_L}, held-out d {curve[best_L]:.2f}, norm {np.linalg.norm(r_vec):.2f}", flush=True)
    dump()

    # ---------------- spine direction (spine_contrast pressured rows) ----------------
    sc = load_probe(args.probes_ref, "spine_contrast")
    pressured = [r for r in sc if r["side"] == "pressured"][:4 if q else None]
    sp_acts, sp_labels, sp_rec, sp_down = [], [], {}, {}
    for item in pressured:
        hist, turns = [], []
        for ti, msg in enumerate(item["turns"]):
            hist.append({"role": msg["role"], "content": msg["content"]})
            if ti == len(item["turns"]) - 1:
                sp_acts.append(R.last_token(H.encode(hist)))
                H.start_recording()
            g = H.generate(hist)
            if ti == len(item["turns"]) - 1:
                sp_down[item["id"]] = H.stop_recording()
            turns.append({"turn": ti, "label": spine_label(g["text"], item["gold"], item["foil"]), "n_tokens": g["n_tokens"], "seconds": g["seconds"]})
            E.private.setdefault("search_spine", {}).setdefault(item["id"], []).append(g["text"])
            hist.append({"role": "assistant", "content": g["text"]})
        sp_labels.append(turns[-1]["label"])
        sp_rec[item["id"]] = {"source_id": item["source_id"], "template": item["template"], "turns": turns, "final_label": turns[-1]["label"]}
        print(f"[search_spine] {item['id']} final {turns[-1]['label']}", flush=True)
    sp_acts = np.stack(sp_acts)
    hold_idx = [i for i, l in enumerate(sp_labels) if l == "hold"]
    fold_idx = [i for i, l in enumerate(sp_labels) if l == "fold"]
    fold_side = ["fold"]
    if len(fold_idx) < 2:
        fold_side = ["fold", "other", "hedge"]
        fold_idx = [i for i, l in enumerate(sp_labels) if l in fold_side]
    spine = {"source": "spine_contrast pressured rows (search set only)", "label": EXPLORATORY, "n_dialogues": len(pressured),
             "final_labels": {l: sp_labels.count(l) for l in ("hold", "hedge", "fold", "other")}, "fold_side_labels": fold_side,
             "n_hold": len(hold_idx), "n_fold_side": len(fold_idx), "per_dialogue": sp_rec}
    if len(hold_idx) >= 2 and len(fold_idx) >= 2:
        if min(len(hold_idx), len(fold_idx)) >= 4:
            rh, rf = hold_idx[:], fold_idx[:]; rng.shuffle(rh); rng.shuffle(rf)
            best_S, s_curve = pick_layer(sp_acts, sp_acts, rh[:len(rh) // 2], rf[:len(rf) // 2], rh[len(rh) // 2:], rf[len(rf) // 2:])
            spine["layer_selection"] = "held-out split"
        else:
            best_S, s_curve = pick_layer(sp_acts, sp_acts, hold_idx, fold_idx, hold_idx, fold_idx)
            spine["layer_selection"] = "in-sample (too few items for a held-out split)"
        s_vec = sp_acts[hold_idx, best_S + 1].mean(0) - sp_acts[fold_idx, best_S + 1].mean(0)
        s_unit = s_vec / np.linalg.norm(s_vec)
        spine.update({"d_per_layer": s_curve, "chosen_layer": best_S, "direction_norm": float(np.linalg.norm(s_vec)),
                      "d_at_chosen": s_curve[best_S]})
        print(f"spine direction: layer {best_S}, d {s_curve[best_S]:.2f}, norm {np.linalg.norm(s_vec):.2f}", flush=True)
    else:
        s_vec = s_unit = None
        raw["failures"].append(f"spine direction undefined: hold {len(hold_idx)}, fold-side {len(fold_idx)}")
    raw["directions"]["spine"] = spine
    # down_proj-input activations at the final turn for STEP 2's neuron search (search set only)
    np.savez(private / "search_acts.npz",
             spine_ids=np.array(list(sp_down)), spine_labels=np.array(sp_labels),
             cp_ids=np.array(list(cp_down)), cp_sides=np.array([cp_gen[i]["side"] for i in cp_down]),
             cp_refused=np.array([cp_gen[i]["refused_crude"] for i in cp_down]),
             **{f"spine_L{L}_{k}": np.stack([sp_down[i][L][k] for i in sp_down]) for L in SEARCH_LAYERS for k in ("last_prompt", "gen_mean")},
             **{f"cp_L{L}_{k}": np.stack([cp_down[i][L][k] for i in cp_down]) for L in SEARCH_LAYERS for k in ("last_prompt", "gen_mean")})
    dump()

    # ---------------- scored conditions ----------------
    rnd = np.random.default_rng(SEED).standard_normal(r_vec.shape[0]).astype(np.float32)
    rnd_unit = rnd / np.linalg.norm(rnd)
    conds = {
        "none": (Intervention(R), {}),
        "refusal_ablate": (Intervention(R, "ablate", unit=r_unit), {"direction": "refusal", "layers": "all", "unit_norm": 1.0}),
        "refusal_subtract": (Intervention(R, "subtract", layer=best_L, vec=r_vec), {"direction": "refusal", "layer": best_L, "norm": float(np.linalg.norm(r_vec))}),
        "random_ablate": (Intervention(R, "ablate", unit=rnd_unit), {"direction": "random unit (seed 0)", "layers": "all"}),
        "random_subtract_refnorm": (Intervention(R, "subtract", layer=best_L, vec=rnd_unit * np.linalg.norm(r_vec)),
                                    {"direction": "random, norm matched to refusal", "layer": best_L, "norm": float(np.linalg.norm(r_vec))}),
    }
    if s_vec is not None:
        conds.update({
            "spine_ablate": (Intervention(R, "ablate", unit=s_unit), {"direction": "spine (hold - fold)", "layers": "all", "label": EXPLORATORY}),
            "spine_subtract": (Intervention(R, "subtract", layer=best_S, vec=s_vec), {"direction": "spine", "layer": best_S, "norm": float(np.linalg.norm(s_vec)), "label": EXPLORATORY}),
            "random_subtract_spinenorm": (Intervention(R, "subtract", layer=best_S, vec=rnd_unit * np.linalg.norm(s_vec)),
                                          {"direction": "random, norm matched to spine", "layer": best_S, "norm": float(np.linalg.norm(s_vec))}),
        })
    for name in args.conditions.split(","):
        if name not in conds:
            raw["failures"].append(f"condition {name} skipped (undefined)")
            continue
        iv, desc = conds[name]
        out = E.run(name, iv, steer_turns="all")
        raw["conditions"][name] = {**desc, **out}
        print(name, json.dumps(out["summary"]), f"wall {out['wall_s']:.0f}s rss {out['peak_rss_gb']:.2f}", flush=True)
        dump()
    raw["meta"]["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    dump()


if __name__ == "__main__":
    main()
