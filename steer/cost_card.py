"""Write steer/COST_CARD.md from steer/raw.json (+ judge_timing.json, leakage.json). No hand-typed numbers.

    steer/.venv/bin/python steer/cost_card.py

Projection unit costs are the measured per-generation seconds (mean, p95) pooled over every
generation of the smoke run (max_new_tokens 96, greedy, bf16, this VM). Multi-turn items cost
one generation per turn; turn counts are read from the probe files on Kenji's branch.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "steer"))
HERE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "steer"  # dir holding raw.json etc.
PROBES_REF = "research/cursor/steering-probes-clean-5853"  # set sizes / turn counts for the projection


def load_probe(ref, name):
    text = subprocess.check_output(["git", "show", f"{ref}:steer/probes/{name}.jsonl"], cwd=ROOT, text=True)
    return [json.loads(l) for l in text.splitlines() if l.strip()]


def turns(rows):
    return sum(len(r["turns"]) if "turns" in r else 1 for r in rows)


def hms(s):
    return f"{s / 3600:.1f} h" if s >= 3600 else f"{s / 60:.1f} min"


def main():
    raw = json.loads((HERE / "raw.json").read_text())
    judge = json.loads((HERE / "judge_timing.json").read_text()) if (HERE / "judge_timing.json").exists() else None
    leak = json.loads((HERE / "leakage.json").read_text()) if (HERE / "leakage.json").exists() else None
    meta, conds, summ = raw["meta"], raw["conditions"], raw["summary"]
    ref = PROBES_REF
    ref_sha = subprocess.check_output(["git", "rev-parse", ref], cwd=ROOT, text=True).strip()

    # pooled unit costs over all generations
    secs = []
    for set_name, items in raw["results"].items():
        for r in items.values():
            for rec in r["by_condition"].values():
                secs += [t["seconds"] for t in rec["turns"]] if "turns" in rec else [rec["seconds"]]
    secs.sort()
    n_gen = len(secs)
    mean_s = sum(secs) / n_gen
    p95_s = secs[min(n_gen - 1, round(0.95 * (n_gen - 1)))]
    ppl_s = raw["ppl_pass_seconds"] / n_gen
    judge_s = judge["mean_s_per_call_excluding_first"] if judge else 2.40
    judge_src = (f"measured here: {judge['judge_id']}, {judge['n_calls']} calls, mean {judge['mean_s_per_call']:.2f} s "
                 f"(first call {judge['first_call_s_includes_warmup']:.2f} s incl. warm-up; {judge_s:.2f} s excluding it), "
                 f"mean prompt {judge['mean_prompt_tokens']:.0f} tokens (shorter than the ~520-token prompts behind the earlier 2.40 s figure), "
                 f"max {judge['max_new_tokens']} new tokens"
                 if judge else "NOT measured here (time); using the earlier cited ~2.40 s/call for ~520-token prompts on Phi-4-mini-instruct")

    # scored / search set sizes from the probe files
    P = {n: load_probe(ref, n) for n in ("persona_probes", "factual_probes", "pressure_probes", "contrast_prompts", "spine_contrast")}
    P["refusal_probes"] = load_probe(ref, "refusal_probes.ids")  # ids + markers only; turn counts come from the local text
    T = {n: turns(rows) for n, rows in P.items()}
    N = {n: len(rows) for n, rows in P.items()}
    tc = meta["refusal_text_source"].get("turn_counts", {})
    T["refusal_probes"] = sum(tc.values()) if len(tc) == N["refusal_probes"] else N["refusal_probes"]
    refusal_turn_note = (f"{T['refusal_probes'] // N['refusal_probes']}-turn dialogues (turn counts from the local withheld text, "
                         f"sha256 {meta['refusal_text_source'].get('sha256', '')[:12]})" if tc else
                         "counted as 1 generation each (withheld text unavailable; real items are multi-turn, so this under-counts)")

    def design(name, conditions, cond_note, sets, extra_gens=0, extra_note=""):
        gens_per_cond = sum(T[s] for s in sets)
        prompts = sum(N[s] for s in sets)
        gens = conditions * gens_per_cond + extra_gens
        return {"name": name, "conditions": conditions, "condition_note": cond_note, "sets": sets,
                "prompts_per_condition": prompts, "generations_per_condition": gens_per_cond,
                "extra_generations": extra_gens, "extra_note": extra_note, "total_generations": gens,
                "gen_wall_mean_s": gens * mean_s, "gen_wall_p95_s": gens * p95_s,
                "ppl_pass_s": gens * ppl_s, "judge_calls": gens, "judge_s": gens * judge_s,
                "total_mean_s": gens * (mean_s + ppl_s + judge_s), "total_p95_s": gens * (p95_s + ppl_s + judge_s)}

    asha_sets = ["persona_probes", "refusal_probes", "factual_probes"]
    designs = [
        design("Asha 53 condition-runs", 1 + 4 + 40 + 8,
               "none + lyra pair x4 doses + 10 random pairs x4 doses + 2 single neurons x4 doses",
               asha_sets),
        design("Asha rough '18 conditions' variant", 18,
               "none + lyra x4 doses + 10 random pairs at dose 1 + 2 singles at dose 1 + 1 spare", asha_sets),
        design("Vivek set (adds 12 pressure items), 53 conditions", 53,
               "same 53 condition-runs as above", asha_sets + ["pressure_probes"]),
        design("Full grid with separate neuron search (leakage note)", 1 + 4 + 40 + 8 + 16 + 16,
               "none + lyra x4 + 10 random pairs x4 + 2 singles x4 + top-k (k=1,2,4,8) x4 + random matched-k x4",
               ["refusal_probes", "pressure_probes", "factual_probes", "persona_probes"],
               extra_gens=T["contrast_prompts"] + T["spine_contrast"],
               extra_note="unsteered search pass over contrast_prompts + spine_contrast (one generation per turn, activations recorded)"),
    ]
    # Asha counted 12 tool_bait as 12 prompts; the repo items are 4-turn dialogues, so note the 72-prompt figure too.
    asha_prompt_count = N["persona_probes"] + 12 + N["factual_probes"]

    proj = {"unit_costs": {"n_generations_measured": n_gen, "mean_s_per_generation": mean_s, "p95_s_per_generation": p95_s,
                           "ppl_check_s_per_generation": ppl_s, "judge_s_per_call": judge_s, "judge_source": judge_src,
                           "max_new_tokens": meta["max_new_tokens"]},
            "set_sizes": {"items": N, "turns": T}, "designs": designs}
    (HERE / "projection.json").write_text(json.dumps(proj, indent=1))

    L = []
    L += [f"# Cost card: neuron steering smoke on CPU", "",
          f"Model `{meta['model_id']}` @ `{meta['revision']}` ({meta['model_class']}, text-only path, {meta['dtype']}, CPU, "
          f"{meta['torch_threads']} torch threads). torch {meta['torch']}, transformers {meta['transformers']}, python {meta['python']}. "
          f"Greedy, max_new_tokens {meta['max_new_tokens']}, chat template with `enable_thinking=False`. "
          f"Probes from `{ref}` @ `{ref_sha[:12]}`" + ("" if ref == meta["probes_ref"] else f" (smoke prompts were read from `{meta['probes_ref']}` @ `{meta['probes_commit'][:12]}`)")
          + ". All numbers below are read from `raw.json` / "
          f"`judge_timing.json` / `leakage.json` by `cost_card.py`.", "",
          f"- load time: {raw['load_time_s']:.1f} s (includes copying the mmapped safetensors into anonymous memory, see `materialize`)",
          f"- peak RSS after load: {raw['peak_rss_gb_after_load']:.2f} GB; end of run: {raw['peak_rss_gb']:.2f} GB (ru_maxrss; it counts the "
          f"mmapped file pages and the anonymous copy together at the moment of copying, so the steady-state footprint is lower, "
          f"about the bf16 weights plus activations; the Phi-4-mini judge peaked at "
          + (f"{judge['peak_rss_gb']:.2f} GB" if judge else "n/a") + ")",
          f"- timing clock: {meta.get('timing_clock', 'time.time')}",
          f"- calibration: {raw['calibration']['n_prompts']} prompts, {raw['calibration']['n_positions']} positions "
          f"(position 0 excluded), {raw['calibration']['seconds']:.1f} s; median residual norm L13 = "
          f"{raw['calibration']['median_resid_norm'][13]:.2f}, L14 = {raw['calibration']['median_resid_norm'][14]:.2f}",
          f"- equivalence check (L{raw['equivalence_check']['layer']} n{raw['equivalence_check']['neuron']}, dose 1, one token): "
          f"added-vector norm {raw['equivalence_check']['added_vec_norm']:.2f} vs target {raw['equivalence_check']['target_norm_dose_x_median']:.2f}, "
          f"cosine to down_proj column {raw['equivalence_check']['cos_added_vs_column']:.4f}, "
          f"rel. err vs fp32 output-side formula {raw['equivalence_check']['rel_err_vs_output_side_fp32']:.2e}",
          f"- smoke sets (lowest ids): neutral {len(meta['sets']['neutral'])}, factual {len(meta["sets"]["factual"])} (Kenji's factual_probes), "
          f"pressure {len(meta['sets']['pressure'])}, refusal {len(meta['sets']['refusal'])} of {N['refusal_probes']} "
          f"(prompt text resolved from local gitignored file: {meta['refusal_text_source']['available']})", ""]
    if raw["failures"]:
        L += ["Failures / skips:", *[f"- {f}" for f in raw["failures"]], ""]

    L += ["## Measured smoke (dose 1)", "",
          "| condition | neurons | gens | mean s/gen | p95 s/gen | tok/s | fold rate (final) | mean delta_logprob (final) | refusal rate crude / repo-lexical (final) | factual acc | mean PPL (unsteered) | villain kw (neutral, crude) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c, s in summ.items():
        neurons = ", ".join(f"L{d['layer']} n{d['neuron']} {d['sign']:+.0f}" for d in conds[c]["deltas"]) or "-"
        f = lambda k, fmt="{:.2f}": fmt.format(s[k]) if s.get(k) is not None else "-"
        L.append(f"| {c}{' (' + conds[c]['label'] + ')' if conds[c].get('label') else ''} | {neurons} | {s['n']} | {s['mean_s']:.1f} | {s['p95_s']:.1f} | {s['tok_per_s']:.2f} "
                 f"| {f('fold_rate_final')} | {f('mean_final_delta_logprob', '{:+.2f}')} | {f('refusal_rate_final_crude')} / {f('refusal_rate_final_repo_lexical')} "
                 f"| {f('factual_accuracy')} | {f('mean_ppl_unsteered')} | {f('villain_keyword_total_neutral_crude', '{}')} |")
    L += ["", f"Pooled over all {n_gen} generations: mean {mean_s:.1f} s, p95 {p95_s:.1f} s per generation; "
          f"unsteered-PPL check {ppl_s:.2f} s per generation (one forward pass).", "",
          "Refusal rate = fraction of refusal (tool_bait) items whose final-turn reply trips the crude marker list in "
          "`steer.py` / the item's own `refusal_markers` via `scorers.spine.marker_hits` (lexical; the NLI scorer needs a "
          "download and was not run). Fold rate = `scorers.spine.spine_label == 'fold'` on the final turn. "
          "delta_logprob = log P(gold) - log P(foil) teacher-forced at the answer position under the steered model. "
          "Villain keyword count is a crude substring count and is secondary.", ""]

    ns = raw["neuron_search"]
    L += [f"## Neuron search (smoke) - {ns['label']}", "",
          "Down_proj-input activations in layers 13/14 on the unsteered run, ranked by Cohen's d; top-20 per layer and "
          "feature are in `raw.json['neuron_search']`. The search used the same items that were scored, so nothing here "
          "supports a claim. Steered quick-check conditions `top2_fold` / `top2_comply` (if present above) carry the same label.", ""]
    for key in ("pressure_fold_vs_hold", "refusal_refused_vs_comply_benign"):
        c = ns[key]
        L.append(f"- `{key}`: a = {c['group_a']}, b = {c['group_b']}; "
                 + (f"n_a = {len(c['ids_a'])}, n_b = {len(c['ids_b'])}" if "ids_a" in c else f"n_a = {c['n_a']}, n_b = {c['n_b']}"))
        for Lyr, r in c["gen_mean"].items():
            if r.get("undefined"):
                L.append(f"  - L{Lyr} (gen_mean): undefined (n_a {r['n_a']}, n_b {r['n_b']})")
            else:
                L.append(f"  - L{Lyr} (gen_mean) top-3: " + ", ".join(f"n{t['neuron']} d={t['d']:+.2f}" for t in r["top"][:3]))
    L += ["", "## Projection for the full falsifier", "",
          f"Unit costs: {mean_s:.1f} s (mean) / {p95_s:.1f} s (p95) per generation at max_new_tokens {meta['max_new_tokens']}; "
          f"PPL check {ppl_s:.2f} s/gen; judge {judge_s:.2f} s/call ({judge_src}). One-time: model load + calibration "
          f"{raw['load_time_s'] + raw['calibration']['seconds']:.0f} s per process, not included.", "",
          f"Set sizes (items / generations per unsteered pass, one generation per dialogue turn): "
          + ", ".join(f"{n} {N[n]}/{T[n]}" for n in N) + f". Asha's '72 prompts' = {asha_prompt_count} items "
          f"(30 neutral + 12 tool_bait + 30 factual); the tool_bait items are {refusal_turn_note} "
          f"and all {N['refusal_probes']} are in `refusal_probes.ids`, so the rows below use every turn of all {N['refusal_probes']} refusal items.", "",
          "| design | conditions | prompts/cond | gens/cond | extra gens | total gens | gen wall @mean | gen wall @p95 | + PPL pass | judge calls @ s/call | total @mean | total @p95 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in designs:
        L.append(f"| {d['name']} | {d['conditions']} | {d['prompts_per_condition']} | {d['generations_per_condition']} | {d['extra_generations']} "
                 f"| {d['total_generations']} | {hms(d['gen_wall_mean_s'])} | {hms(d['gen_wall_p95_s'])} | {hms(d['ppl_pass_s'])} "
                 f"| {d['judge_calls']} x {judge_s:.2f} s = {hms(d['judge_s'])} | {hms(d['total_mean_s'])} | {hms(d['total_p95_s'])} |")
    L += ["", "Condition definitions:", *[f"- {d['name']}: {d['conditions']} = {d['condition_note']}"
                                          + (f"; extra = {d['extra_note']}" if d["extra_note"] else "") for d in designs], "",
          "Judging: one persona-rubric call per generation (upper bound; judging only the neutral set would divide the "
          "judge column by the share of neutral generations). Doses: 0.25 / 0.5 / 1 / 2. Everything greedy, one generation per prompt.", ""]
    if leak:
        a, b = leak["spine_contrast_vs_pressure_probes"], leak["contrast_prompts_vs_refusal_probes"]
        L += ["## Leakage (see `LEAKAGE.md`)", "",
              f"- spine_contrast vs pressure_probes: shared ids {len(a['shared_ids'])}; scored items with `template` in search set "
              f"{a['scored_with_template_in_search']}/{a['n_scored']}; with template *family* (fp/mr prefix) in search set "
              f"{a['scored_with_family_in_search']}/{a['n_scored']}; max token Jaccard {max(i['max_token_jaccard'] for i in a['items'])}",
              f"- contrast_prompts vs refusal_probes: shared ids {len(b['shared_ids'])}; scored items with `template` in search set "
              f"{b['scored_with_template_in_search']}/{b['n_scored']}; family {b['scored_with_family_in_search']}/{b['n_scored']}; "
              f"max token Jaccard {max(i['max_token_jaccard'] for i in b['items'])}",
              "- Search-based claims (top-k steering) need template-disjoint search/eval splits. `template` fields are now disjoint "
              "(Kenji's `template_overlap.md`), but the pressure search set is still built from the same two dialogue families; "
              "neuron rankings stay EXPLORATORY until Asha's scorer confirms zero overlap against this tree.", ""]
    if (HERE / "raw_hunt.json").exists():
        from report import hunt_section
        hunt = json.loads((HERE / "raw_hunt.json").read_text())
        wall = sum(c["wall_s"] for c in hunt["conditions"].values())
        L += ["## Neuron hunt (see `HUNT.md`, `raw_hunt.json`)", "",
              f"{len(hunt['conditions'])} scored conditions, {wall / 3600:.1f} h of generation+scoring wall time, peak RSS {hunt['peak_rss_gb']:.2f} GB"
              + ("" if "finished_utc" in hunt["meta"] else " (run still in progress when this card was rendered)") + ".", "",
              *hunt_section(hunt), ""]
    (HERE / "COST_CARD.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
