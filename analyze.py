"""Tables, statistics and figures for the v1 runs.

    python analyze.py    # reads artifacts/v1/*/results.jsonl
                         # writes artifacts/v1/analysis/{numbers.json, *.png} and artifacts/v1/RESULTS.md

Every number in the paper comes from numbers.json.
"""

import json
from collections import Counter, defaultdict
from math import comb, sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

V1 = Path("artifacts/v1")
OUT = V1 / "analysis"
LABELS = ("hold", "hedge", "fold", "other")
CONDITIONS = ("foil", "gold", "neutral", "none")


def load(name):
    path = V1 / name / "results.jsonl"
    return [json.loads(l) for l in open(path, encoding="utf-8")] if path.exists() else []


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def rate(k, n):
    return {"k": k, "n": n, "p": round(k / n, 3) if n else None, "ci95": wilson(k, n)}


def mcnemar(b, c):
    """Exact two-sided McNemar p-value from the two discordant-pair counts."""
    n = b + c
    if n == 0:
        return 1.0
    return round(min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n), 4)


def auroc(pos, neg):
    """P(score of a random `pos` > score of a random `neg`); ties count half."""
    if not pos or not neg:
        return None
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return round(wins / (len(pos) * len(neg)), 3)


def held(rec):
    return rec["label"] == "hold"


def trajectory(rows):
    """Per-turn labels and hold rate, over all items and over the items that held turn 0."""
    base = [r for r in rows if held(r["turns"][0])]
    per_turn = []
    for t in range(max(len(r["turns"]) for r in rows)):
        recs = [r["turns"][t] for r in rows if t < len(r["turns"])]
        recs0 = [r["turns"][t] for r in base if t < len(r["turns"])]
        per_turn.append({"turn": t, "labels": {k: sum(x["label"] == k for x in recs) for k in LABELS},
                         "hold": rate(sum(map(held, recs)), len(recs)),
                         "hold_given_t0": rate(sum(map(held, recs0)), len(recs0))})
    recovered = sum(any(not held(x) for x in r["turns"][1:-1]) and held(r["turns"][-1]) for r in base)
    ever_fold = sum(any(x["label"] == "fold" for x in r["turns"][1:]) for r in base)
    return {"n_items": len(rows), "held_t0": rate(len(base), len(rows)),
            "held_final": rate(sum(held(r["turns"][-1]) for r in rows), len(rows)),
            "held_final_given_t0": rate(sum(held(r["turns"][-1]) for r in base), len(base)),
            "ever_fold_given_t0": rate(ever_fold, len(base)),
            "broke_then_held_final_given_t0": rate(recovered, len(base)),
            "final_labels": dict(Counter(r["turns"][-1]["label"] for r in rows)),
            "per_turn": per_turn}


def truncation(rows):
    cells = [x for r in rows for x in r["turns"]]
    return rate(sum(bool(x["truncated"]) for x in cells), len(cells))


def controls_analysis(rows):
    by_base = defaultdict(dict)
    for r in rows:
        by_base[r["base_id"]][r["condition"]] = r
    out = {"n_questions": len(by_base), "conditions": {}, "paired_vs_foil": {},
           "t0_identical_across_conditions": all(
               len({r["turns"][0]["answer"] for r in v.values()}) == 1 for v in by_base.values())}
    for c in CONDITIONS:
        rs = [v[c] for v in by_base.values() if c in v]
        out["conditions"][c] = {
            "held_final": rate(sum(held(r["turns"][-1]) for r in rs), len(rs)),
            "final_labels": dict(Counter(r["turns"][-1]["label"] for r in rs)),
            "per_turn_hold": [rate(sum(held(r["turns"][t]) for r in rs), len(rs)) for t in range(len(rs[0]["turns"]))]}
    for c in CONDITIONS[1:]:
        pairs = [(held(v["foil"]["turns"][-1]), held(v[c]["turns"][-1])) for v in by_base.values()]
        b, d = sum(f and not o for f, o in pairs), sum(o and not f for f, o in pairs)
        out["paired_vs_foil"][c] = {"n": len(pairs), "held_only_under_foil": b, "held_only_under_other": d,
                                    "mcnemar_exact_p": mcnemar(b, d)}
    return out


def delta_analysis(rows):
    """Does the teacher-forced gold-vs-foil contrast agree with the text verdict? hold/fold cells only."""
    cells = [x for r in rows for x in r["turns"] if x["label"] in ("hold", "fold")]
    n_hold = sum(map(held, cells))
    out = {"n_cells": len(cells), "n_hold": n_hold, "n_fold": len(cells) - n_hold,
           "majority_baseline": round(max(n_hold, len(cells) - n_hold) / len(cells), 3) if cells else None}
    for key in ("delta_logprob", "delta_logprob_per_token"):
        agree = sum((x[key] > 0) == held(x) for x in cells)
        out[key] = {"sign_agreement": rate(agree, len(cells)),
                    "auroc_hold_vs_fold": auroc([x[key] for x in cells if held(x)],
                                                [x[key] for x in cells if not held(x)])}
    return out


def concession_analysis(rows):
    """Apology/concession language vs. actually folding, on pressure turns."""
    cells = [x for r in rows for x in r["turns"][1:]]
    conceded = [x for x in cells if x["concedes"]]
    plain = [x for x in cells if not x["concedes"]]
    return {"n_cells": len(cells),
            "table": {f"{'concedes' if c else 'no_concession'}/{lab}":
                      sum(bool(x["concedes"]) == c and x["label"] == lab for x in cells)
                      for c in (True, False) for lab in LABELS},
            "concedes": rate(len(conceded), len(cells)),
            "hold_given_concedes": rate(sum(map(held, conceded)), len(conceded)),
            "fold_given_concedes": rate(sum(x["label"] == "fold" for x in conceded), len(conceded)),
            "fold_given_no_concession": rate(sum(x["label"] == "fold" for x in plain), len(plain))}


def harm_analysis(rows):
    cells = [x for r in rows for x in r["turns"]]
    with_lex = [x for x in cells if "label_lexical" in x]
    classes = defaultdict(list)
    for r in rows:
        classes[r["harm_class"]].append(held(r["turns"][-1]))
    return {**trajectory(rows),
            "nli_vs_lexical_label_agreement": rate(sum(x["label"] == x["label_lexical"] for x in with_lex), len(with_lex)),
            "nli_vs_lexical_confusion": dict(Counter(f"{x['label_lexical']}->{x['label']}" for x in with_lex)),
            "held_final_by_class": {c: rate(sum(v), len(v)) for c, v in sorted(classes.items())}}


def compare_runs(a_rows, b_rows):
    """Cell-by-cell agreement of two runs on the same items. Turns after the first divergent
    answer see different histories, so only the cells up to and including it are comparable."""
    b_by_id = {r["id"]: r for r in b_rows}
    same_text = same_label = cells = first_diverge = 0
    for a in a_rows:
        if a["id"] not in b_by_id:
            continue
        for x, y in zip(a["turns"], b_by_id[a["id"]]["turns"]):
            cells += 1
            same_label += x["label"] == y["label"]
            if x["answer"] != y["answer"]:
                first_diverge += 1
                break
            same_text += 1
    return {"cells_compared": cells, "identical_answers": same_text, "same_label": same_label,
            "items_diverged": first_diverge}


def fig_trajectories(curves, path):
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for name, traj in curves.items():
        ts = [p["turn"] for p in traj["per_turn"]]
        ps = [p["hold"]["p"] for p in traj["per_turn"]]
        lo = [p["hold"]["p"] - p["hold"]["ci95"][0] for p in traj["per_turn"]]
        hi = [p["hold"]["ci95"][1] - p["hold"]["p"] for p in traj["per_turn"]]
        ax.errorbar(ts, ps, yerr=[lo, hi], marker="o", capsize=3, label=f"{name} (n={traj['n_items']})")
    ax.set(xlabel="turn (0 = clean question)", ylabel="fraction holding", ylim=(0, 1.02))
    ax.set_xticks(range(max(len(t["per_turn"]) for t in curves.values())))
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_controls(ctrl, path):
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    bottom = [0] * len(CONDITIONS)
    colors = {"hold": "#2b8a3e", "hedge": "#f2c94c", "fold": "#c92a2a", "other": "#adb5bd"}
    for lab in LABELS:
        vals = [ctrl["conditions"][c]["final_labels"].get(lab, 0) for c in CONDITIONS]
        ax.bar(CONDITIONS, vals, bottom=bottom, color=colors[lab], label=lab)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set(xlabel="pressure condition", ylabel="items (final turn)")
    ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    main_rows, harm, ctrl = load("main"), load("harm"), load("controls")
    rep32, rep16 = load("repeat_fp32"), load("repeat_bf16")
    by_tmpl = defaultdict(list)
    for r in main_rows:
        by_tmpl[r["template"]].append(r)

    n = {"main": {t: trajectory(rs) for t, rs in by_tmpl.items()}}
    if main_rows:
        n["main"]["all_factual"] = trajectory(main_rows)
        n["delta"] = delta_analysis(main_rows + ctrl)
        n["concession"] = concession_analysis(main_rows + ctrl)
    if harm:
        n["harm"] = harm_analysis(harm)
    if ctrl:
        n["controls"] = controls_analysis(ctrl)
    n["truncation"] = {k: truncation(v) for k, v in (("main", main_rows), ("harm", harm), ("controls", ctrl)) if v}
    if rep32:
        n["determinism_fp32_rerun"] = compare_runs(rep32, harm)
    if rep16:
        n["precision_bf16_vs_fp32"] = compare_runs(rep16, rep32 or harm)
    json.dump(n, open(OUT / "numbers.json", "w", encoding="utf-8"), indent=2)

    curves = {t: n["main"][t] for t in ("false_premise", "memory_rewrite") if t in n["main"]}
    if harm:
        curves["escalating_harm"] = n["harm"]
    if curves:
        fig_trajectories(curves, OUT / "fig_trajectories.png")
    if ctrl:
        fig_controls(n["controls"], OUT / "fig_controls.png")
    write_results(n)
    print(f"wrote {OUT / 'numbers.json'}")


def fmt(r):
    return f"{r['k']}/{r['n']} ({r['p']:.2f}, 95% CI {r['ci95'][0]:.2f}-{r['ci95'][1]:.2f})" if r["n"] else "n/a"


def write_results(n):
    lines = ["# v1 results", "", "Generated by `python analyze.py` from `artifacts/v1/*/results.jsonl`.", ""]
    for name, t in [*n.get("main", {}).items(), *([("escalating_harm", n["harm"])] if "harm" in n else [])]:
        lines += [f"## {name}", "",
                  f"- held turn 0: {fmt(t['held_t0'])}",
                  f"- held final turn: {fmt(t['held_final'])}",
                  f"- held final, given held turn 0: {fmt(t['held_final_given_t0'])}",
                  f"- folded at least once, given held turn 0: {fmt(t['ever_fold_given_t0'])}",
                  f"- final labels: {t['final_labels']}", ""]
    if "controls" in n:
        c = n["controls"]
        lines += ["## controls (final turn)", ""]
        lines += [f"- {k}: {fmt(v['held_final'])}, labels {v['final_labels']}" for k, v in c["conditions"].items()]
        lines += [f"- foil vs {k}: McNemar exact p = {v['mcnemar_exact_p']} "
                  f"({v['held_only_under_foil']} vs {v['held_only_under_other']} discordant)"
                  for k, v in c["paired_vs_foil"].items()]
        lines += [f"- turn 0 identical across conditions: {c['t0_identical_across_conditions']}", ""]
    for key in ("delta", "concession", "truncation", "determinism_fp32_rerun", "precision_bf16_vs_fp32"):
        if key in n:
            lines += [f"## {key}", "", "```json", json.dumps(n[key], indent=2), "```", ""]
    (V1 / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
