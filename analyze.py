"""Tables, statistics and figures for the v1 runs.

    python analyze.py    # reads artifacts/v1/*/results.jsonl
                         # writes artifacts/v1/analysis/{numbers.json, *.png} and artifacts/v1/RESULTS.md

Every number in the paper comes from numbers.json.
"""

import json
from collections import Counter, defaultdict
from itertools import combinations
from math import comb, sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scorers.spine import spine_label

V1 = Path("artifacts/v1")
OUT = V1 / "analysis"
LABELS = ("hold", "hedge", "fold", "other")
CONDITIONS = ("foil", "gold", "neutral", "none")


SUITES = {"main": "handwritten.jsonl", "controls": "controls.jsonl"}
RESCORED = []


def load(name):
    """Results of one run; factual cells are re-labelled with the current suite's aliases
    (prompts and targets are unchanged, so only a label can move) and every change is logged."""
    path = V1 / name / "results.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    if name in SUITES:
        items = {r["id"]: r for r in map(json.loads, open(Path("suite/v1") / SUITES[name], encoding="utf-8"))}
        for r in rows:
            it = items[r["id"]]
            gold, foil = [it["gold"], *it["gold_aliases"]], [it["foil"], *it["foil_aliases"]]
            for x in r["turns"]:
                new = spine_label(x["answer"], gold, foil)
                if new != x["label"]:
                    RESCORED.append({"run": name, "id": r["id"], "turn": x["turn"], "as_run": x["label"], "rescored": new})
                    x["label"] = new
    return rows


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
    # Turn 2 is the last pressure turn; turn 3 (final) re-asks without restating the pressure.
    for when, t in (("peak", -2), ("final", -1)):
        out["paired_vs_foil"][when] = {}
        for c in CONDITIONS[1:]:
            pairs = [(held(v["foil"]["turns"][t]), held(v[c]["turns"][t])) for v in by_base.values()]
            b, d = sum(f and not o for f, o in pairs), sum(o and not f for f, o in pairs)
            out["paired_vs_foil"][when][c] = {"n": len(pairs), "held_only_under_foil": b,
                                              "held_only_under_other": d, "mcnemar_exact_p": mcnemar(b, d)}
    return out


def delta_analysis(rows):
    """Does the teacher-forced gold-vs-foil contrast agree with the text verdict? hold/fold cells only."""
    cells = [x for r in rows for x in r["turns"] if x["label"] in ("hold", "fold")]
    n_hold = sum(map(held, cells))
    folds = [x["delta_logprob"] for x in cells if not held(x)]
    out = {"n_cells": len(cells), "n_hold": n_hold, "n_fold": len(cells) - n_hold,
           "majority_baseline": round(max(n_hold, len(cells) - n_hold) / len(cells), 3) if cells else None,
           "hold_with_delta_le_0": sum(x["delta_logprob"] <= 0 for x in cells if held(x)),
           "fold_with_delta_gt_0": sum(d > 0 for d in folds), "max_fold_delta": round(max(folds), 2)}
    for key in ("delta_logprob", "delta_logprob_per_token"):
        agree = sum((x[key] > 0) == held(x) for x in cells)
        out[key] = {"sign_agreement": rate(agree, len(cells)),
                    "auroc_hold_vs_fold": auroc([x[key] for x in cells if held(x)],
                                                [x[key] for x in cells if not held(x)])}
    return out


def concession_analysis(rows):
    """Apology/concession language vs. actually folding, on turns after a wrong answer was pushed."""
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


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in LABELS) / n ** 2
    return round((po - pe) / (1 - pe), 3) if pe < 1 else 1.0


def annotator_agreement(label_of):
    """Blind AI annotators (labels/annotator_*.jsonl) vs. the scorer, the lexical refusal scorer
    (harm cells), and each other; plus scorer agreement on the cells where all annotators agree."""
    lab = V1 / "labels"
    files = sorted(lab.glob("annotator_*.jsonl"))
    if not files:
        return None
    key = {k["cell"]: k for k in map(json.loads, open(lab / "key.jsonl", encoding="utf-8"))}
    kind = {r["cell"]: r["kind"] for r in map(json.loads, open(lab / "sheet.jsonl", encoding="utf-8"))}
    ann = {f.stem.split("_", 1)[1]: {r["cell"]: r["label"].strip().lower() for r in map(json.loads, open(f, encoding="utf-8"))}
           for f in files}
    first = next(iter(ann.values()))
    scorer = {c: label_of[(k["run"], k["id"], k["turn"])] for c, k in key.items()}
    out = {}
    for group in ("factual", "harm", "all"):
        cells = sorted(c for c in key if group == "all" or kind[c] == group)
        sources = {"scorer": scorer, **{f"annotator_{k}": v for k, v in ann.items()}}
        if group == "harm":
            sources["lexical"] = {c: key[c]["lexical_label"] for c in cells}
        res = {"n": len(cells)}
        for (n1, s1), (n2, s2) in combinations(sources.items(), 2):
            a, b = [s1[c] for c in cells], [s2[c] for c in cells]
            res[f"{n1}_vs_{n2}"] = {"agree": rate(sum(x == y for x, y in zip(a, b)), len(cells)), "kappa": kappa(a, b)}
        consensus = [c for c in cells if len({v[c] for v in ann.values()}) == 1]
        res["annotators_agree"] = rate(len(consensus), len(cells))
        for name in ("scorer", "lexical"):
            if name in sources:
                res[f"{name}_on_consensus"] = rate(sum(sources[name][c] == first[c] for c in consensus), len(consensus))
        res["scorer_vs_consensus_disagreements"] = [
            {**{k: key[c][k] for k in ("run", "id", "turn")}, "scorer": scorer[c], "annotators": first[c]}
            for c in consensus if scorer[c] != first[c]]
        out[group] = res
    return out


LABEL_COLOR = {"hold": "#2E7D32", "hedge": "#F2A900", "fold": "#C62828", "other": "#A7B1B8"}
SUITE_COLOR = {"false_premise": "#C0392B", "memory_rewrite": "#2471A3", "escalating_harm": "#7D3C98"}
SUITE_NAME = {"false_premise": "false premise", "memory_rewrite": "memory rewrite", "escalating_harm": "escalating harm"}
COND_COLOR = {"foil": "#C0392B", "gold": "#2E7D32", "neutral": "#E67E22", "none": "#5D6D7E"}
COND_NAME = {"foil": "assert wrong answer", "gold": "assert correct answer", "neutral": "\u201care you sure?\u201d",
             "none": "re-ask, no challenge"}


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 8,
        "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#444444",
        "axes.grid": True, "axes.grid.axis": "y", "grid.alpha": 0.3, "grid.linewidth": 0.6,
        "legend.frameon": False, "legend.fontsize": 7.5, "savefig.bbox": "tight", "savefig.pad_inches": 0.03})


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=220)
    plt.close(fig)


def band(ax, traj_points, color, label, key="hold"):
    ts = range(len(traj_points))
    ps = [p[key]["p"] for p in traj_points]
    ax.fill_between(ts, [p[key]["ci95"][0] for p in traj_points], [p[key]["ci95"][1] for p in traj_points],
                    color=color, alpha=0.10, lw=0)
    ax.plot(ts, ps, color=color, marker="o", ms=4, lw=1.8, label=label)


def fig_hold(n):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"wspace": 0.28})
    below = {"loc": "upper center", "bbox_to_anchor": (0.5, -0.2), "ncol": 2, "fontsize": 7,
             "handlelength": 1.3, "handletextpad": 0.4, "columnspacing": 0.9}
    for t in ("false_premise", "memory_rewrite", "escalating_harm"):
        traj = n["harm"] if t == "escalating_harm" else n["main"][t]
        band(a, traj["per_turn"], SUITE_COLOR[t], f"{SUITE_NAME[t]} (n={traj['n_items']})")
    a.set(title="(a) Sustained pressure", xlabel="turn (0 = clean question; every later turn pushes)",
          ylabel="fraction of items holding", ylim=(0, 1.04), xticks=range(5))
    a.legend(**below)
    c = n["controls"]["conditions"]
    b.axvspan(0.55, 2.45, color="#B0B7BD", alpha=0.14, lw=0)
    for i, cond in enumerate(CONDITIONS):
        pts = c[cond]["per_turn_hold"]
        xs = [t + (i - 1.5) * 0.07 for t in range(len(pts))]
        ps = [p["p"] for p in pts]
        b.errorbar(xs, ps, yerr=[[p - q["ci95"][0] for p, q in zip(ps, pts)], [q["ci95"][1] - p for p, q in zip(ps, pts)]],
                   color=COND_COLOR[cond], marker="o", ms=4, lw=1.8, capsize=2, elinewidth=0.8, label=COND_NAME[cond])
    b.set(title=f"(b) Matched controls (n={n['controls']['n_questions']} questions each)",
          xlabel="turn (shaded: pressure turns; 3 = identical re-ask)", ylim=(0, 1.04), xticks=range(4))
    b.legend(**below)
    save(fig, "fig_hold")


def fig_labels(n):
    panels = [("false_premise", n["main"]["false_premise"]), ("memory_rewrite", n["main"]["memory_rewrite"]),
              ("escalating_harm", n["harm"])]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, (t, traj) in zip(axes, panels):
        bottom = [0.0] * len(traj["per_turn"])
        for lab in LABELS:
            vals = [p["labels"][lab] / sum(p["labels"].values()) for p in traj["per_turn"]]
            ax.bar(range(len(vals)), vals, bottom=bottom, color=LABEL_COLOR[lab], width=0.72,
                   label=lab, edgecolor="white", linewidth=0.5)
            bottom = [x + v for x, v in zip(bottom, vals)]
        ax.set(title=SUITE_NAME[t], xlabel="turn", xticks=range(len(bottom)), ylim=(0, 1))
        ax.grid(False)
    axes[0].set_ylabel("share of replies")
    axes[0].legend(ncol=4, loc="lower left", bbox_to_anchor=(0, 1.13), handlelength=1)
    save(fig, "fig_labels")


def fig_delta(cells, d):
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    bins = [x / 2 for x in range(-40, 33)]
    for lab in ("hold", "fold"):
        ax.hist([x["delta_logprob"] for x in cells if x["label"] == lab], bins=bins, color=LABEL_COLOR[lab],
                alpha=0.7, label=f"{lab} (n={sum(x['label'] == lab for x in cells)})")
    ax.axvline(0, color="#333333", lw=0.8, ls="--")
    ax.set(xlabel=r"$\Delta = \log P(\mathrm{gold}) - \log P(\mathrm{foil})$", ylabel="cells",
           title=f"AUROC {d['delta_logprob']['auroc_hold_vs_fold']:.2f}")
    ax.legend(loc="upper left")
    save(fig, "fig_delta")


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
        n["concession"] = concession_analysis(main_rows + [r for r in ctrl if r["condition"] == "foil"])
    if harm:
        n["harm"] = harm_analysis(harm)
    if ctrl:
        n["controls"] = controls_analysis(ctrl)
    n["truncation"] = {k: truncation(v) for k, v in (("main", main_rows), ("harm", harm), ("controls", ctrl)) if v}
    if rep32:
        n["determinism_fp32_rerun"] = compare_runs(rep32, harm)
    if rep16:
        n["precision_bf16_vs_fp32"] = compare_runs(rep16, rep32 or harm)
    label_of = {(name, r["id"], x["turn"]): x["label"]
                for name, rows in (("main", main_rows), ("controls", ctrl), ("harm", harm)) for r in rows for x in r["turns"]}
    agreement = annotator_agreement(label_of)
    if agreement:
        n["annotators"] = agreement
    n["rescored_cells"] = {"n": len(RESCORED), "cells": RESCORED}
    json.dump(n, open(OUT / "numbers.json", "w", encoding="utf-8"), indent=2)

    style()
    if main_rows and harm and ctrl:
        fig_hold(n)
        fig_labels(n)
        fig_delta([x for r in main_rows + ctrl for x in r["turns"] if x["label"] in ("hold", "fold")], n["delta"])
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
        lines += [f"- {when}: foil vs {k}: McNemar exact p = {v['mcnemar_exact_p']} "
                  f"({v['held_only_under_foil']} vs {v['held_only_under_other']} discordant)"
                  for when, tests in c["paired_vs_foil"].items() for k, v in tests.items()]
        lines += [f"- turn 0 identical across conditions: {c['t0_identical_across_conditions']}", ""]
    for key in ("delta", "concession", "truncation", "determinism_fp32_rerun", "precision_bf16_vs_fp32"):
        if key in n:
            lines += [f"## {key}", "", "```json", json.dumps(n[key], indent=2), "```", ""]
    (V1 / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
