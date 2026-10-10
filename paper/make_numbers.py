"""Write paper/numbers.tex, paper/tab_*.tex and paper/ex_*.tex from the analysis output.

    python analyze.py && python paper/make_numbers.py

Every number and quoted model reply in the paper comes from here; nothing is typed by hand.
"""

import json
import re
import shutil
from math import floor, log10
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
V1 = ROOT / "artifacts" / "v1"
N = json.load(open(V1 / "analysis" / "numbers.json", encoding="utf-8"))
macros = []


def tex_escape(s):
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    out = []
    for ch in s:
        out.append({"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
                    "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
                    "\u00b0": r"\textdegree{}"}.get(ch, ch))
    return "".join(out)


def define(name, value):
    macros.append(f"\\newcommand{{\\{name}}}{{{value}}}")


def rate(name, r):
    """\\name = k/n, \\namep = percent, \\nameci = CI in percent."""
    define(name, f"{r['k']}/{r['n']}")
    define(name + "p", f"{100 * r['p']:.0f}\\%")
    define(name + "ci", f"{100 * r['ci95'][0]:.0f}--{100 * r['ci95'][1]:.0f}\\%")


def pval(p):
    if p >= 0.001:
        return f"{p:.3g}"
    e = floor(log10(p))
    return f"${p / 10 ** e:.0f}\\times10^{{{e}}}$"


# --- sustained pressure ---
for tag, t in (("FP", N["main"]["false_premise"]), ("MR", N["main"]["memory_rewrite"]),
               ("ALL", N["main"]["all_factual"]), ("H", N["harm"])):
    define(tag + "n", t["n_items"])
    rate(tag + "zero", t["held_t0"])
    rate(tag + "one", t["per_turn"][1]["hold"])
    rate(tag + "three", t["per_turn"][3]["hold"])
    rate(tag + "final", t["held_final"])
    rate(tag + "finalgiven", t["held_final_given_t0"])
    rate(tag + "ever", t["ever_fold_given_t0"])
    rate(tag + "recover", t["broke_then_held_final_given_t0"])
    define(tag + "finalfold", t["final_labels"].get("fold", 0))
rate("HNliLex", N["harm"]["nli_vs_lexical_label_agreement"])
pressure = [p["labels"] for t in ("false_premise", "memory_rewrite") for p in N["main"][t]["per_turn"][1:]]
define("Phedge", sum(p["hedge"] for p in pressure))
define("Pother", sum(p["other"] for p in pressure))
conf = N["harm"]["nli_vs_lexical_confusion"]
define("HLexDis", sum(v for k, v in conf.items() if k.split("->")[0] != k.split("->")[1]))
define("HLexOtherFold", conf.get("other->fold", 0))

# --- controls ---
C = N["controls"]
define("Cn", C["n_questions"])
for cond in ("foil", "gold", "neutral", "none"):
    pts = C["conditions"][cond]["per_turn_hold"]
    rate("C" + cond + "zero", pts[0])
    rate("C" + cond + "one", pts[1])
    rate("C" + cond + "peak", pts[-2])
    rate("C" + cond + "final", pts[-1])
for when in ("peak", "final"):
    for cond, v in C["paired_vs_foil"][when].items():
        define(f"Cp{cond}{when}", pval(v["mcnemar_exact_p"]))
        define(f"Co{cond}{when}", v["held_only_under_other"])
        define(f"Cf{cond}{when}", v["held_only_under_foil"])

# --- delta, concession, truncation, rescoring ---
D = N["delta"]
define("Dn", D["n_cells"])
define("Dhold", D["n_hold"])
define("Dfold", D["n_fold"])
define("Dmaj", f"{100 * D['majority_baseline']:.1f}\\%")
define("Dholdneg", D["hold_with_delta_le_0"])
define("Dfoldpos", D["fold_with_delta_gt_0"])
define("Dfoldmax", f"{D['max_fold_delta']:.1f}")
for tag, key in (("D", "delta_logprob"), ("Dtok", "delta_logprob_per_token")):
    rate(tag + "sign", D[key]["sign_agreement"])
    define(tag + "auc", f"{D[key]['auroc_hold_vs_fold']:.2f}")
K = N["concession"]
define("Kn", K["n_cells"])
rate("Kcon", K["concedes"])
rate("Kholdcon", K["hold_given_concedes"])
rate("Kfoldcon", K["fold_given_concedes"])
rate("Kfoldplain", K["fold_given_no_concession"])
for k, v in N["truncation"].items():
    rate("T" + k, v)
define("Rn", N["rescored_cells"]["n"])

# --- determinism / precision ---
for tag, key in (("Rep", "determinism_fp32_rerun"), ("Prec", "precision_bf16_vs_fp32")):
    if key in N:
        v = N[key]
        define(tag + "cells", v["cells_compared"])
        define(tag + "same", v["identical_answers"])
        define(tag + "label", v["same_label"])
        define(tag + "div", v["items_diverged"])

# --- annotators ---
A = N.get("annotators")
if A:
    for group, tag in (("factual", "Af"), ("harm", "Ah"), ("all", "Aall")):
        g = A[group]
        define(tag + "n", g["n"])
        for pair, short in (("scorer_vs_annotator_A", "SA"), ("scorer_vs_annotator_B", "SB"),
                            ("annotator_A_vs_annotator_B", "AB"), ("annotator_A_vs_lexical", "LA"),
                            ("annotator_B_vs_lexical", "LB"), ("scorer_vs_lexical", "SL")):
            if pair in g:
                define(tag + short + "k", f"{g[pair]['kappa']:.2f}")
                define(tag + short + "agree", f"{100 * g[pair]['agree']['p']:.0f}\\%")
        rate(tag + "cons", g["annotators_agree"])
        rate(tag + "scons", g["scorer_on_consensus"])
        if "lexical_on_consensus" in g:
            rate(tag + "lcons", g["lexical_on_consensus"])

# --- provenance ---
prov = json.load(open(V1 / "main" / "summary.json", encoding="utf-8"))
define("ProvTorch", prov["torch"])
define("ProvTransformers", prov["transformers"])
define("ProvPython", prov["python"])
define("ProvCommit", prov["git_commit"][:7])
hours = sum(json.load(open(V1 / d / "summary.json", encoding="utf-8"))["elapsed_seconds"]
            for d in ("main", "harm", "controls", "repeat_fp32", "repeat_bf16") if (V1 / d / "summary.json").exists()) / 3600
define("ProvHours", f"{hours:.1f}")

(PAPER / "numbers.tex").write_text("% Generated by paper/make_numbers.py. Do not edit.\n" + "\n".join(macros) + "\n",
                                   encoding="utf-8")

# --- tables ---
rows = []
for cond, name in (("foil", "assert the wrong answer"), ("gold", "assert the correct answer"),
                   ("neutral", "``are you sure?''"), ("none", "re-ask, no challenge")):
    pts = C["conditions"][cond]["per_turn_hold"]
    cells = " & ".join(f"{p['k']}" for p in pts)
    pk = "---" if cond == "foil" else f"{pval(C['paired_vs_foil']['peak'][cond]['mcnemar_exact_p'])}"
    fn = "---" if cond == "foil" else f"{pval(C['paired_vs_foil']['final'][cond]['mcnemar_exact_p'])}"
    rows.append(f"\\texttt{{{cond}}} & {name} & {cells} & {pk} & {fn} \\\\")
(PAPER / "tab_controls.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

NICE = {"destructive_command": "destructive command", "unauthorized_access": "unauthorized access",
        "dangerous_instructions": "dangerous instructions", "data_exfiltration": "data exfiltration",
        "secrets_exfiltration": "secrets exfiltration", "prompt_exfiltration": "system-prompt extraction",
        "social_engineering": "social engineering", "hate_speech": "hate speech"}
cls = sorted(N["harm"]["held_final_by_class"].items(), key=lambda kv: (-kv[1]["n"], kv[0]))
(PAPER / "tab_harm.tex").write_text(
    "\n".join(f"{NICE.get(k, k)} & {v['n']} & {v['k']} \\\\" for k, v in cls) + "\n", encoding="utf-8")


# --- example dialogues (factual only; harmful replies are withheld) ---
def example(rid, path, turns=None):
    r = next(json.loads(l) for l in open(V1 / "main" / "results.jsonl", encoding="utf-8") if json.loads(l)["id"] == rid)
    lines = []
    for x in r["turns"]:
        if turns is not None and x["turn"] not in turns:
            continue
        lines.append(f"\\usr{{{x['turn']}}}{{{tex_escape(x['user'])}}}")
        lines.append(f"\\asst{{{x['label']}}}{{{tex_escape(x['answer'])}}}")
    (PAPER / path).write_text("\n".join(lines) + "\n", encoding="utf-8")


example("fp-007", "ex_fp007.tex")
example("mr-009", "ex_mr009.tex")

FIG = PAPER / "figures"
FIG.mkdir(exist_ok=True)
for f in ("fig_hold.pdf", "fig_labels.pdf", "fig_delta.pdf"):
    shutil.copy(V1 / "analysis" / f, FIG / f)
print(f"{len(macros)} macros, tables, examples and figures written to {PAPER}")
