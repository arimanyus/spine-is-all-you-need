"""Blind label sheet: a seeded random sample of scored cells, without the scorer's verdict.

    python make_label_sheet.py              # writes artifacts/v1/labels/{sheet.jsonl, sheet.csv, key.jsonl}
    python make_label_sheet.py --agree a.jsonl b.jsonl ...   # Cohen's kappa of each annotator vs. the scorer and pairwise

An annotator fills `label` in a copy of sheet.jsonl with one of hold / hedge / fold / other.
"""

import argparse
import csv
import json
import random
from collections import Counter
from itertools import combinations
from pathlib import Path

V1 = Path("artifacts/v1")
OUT = V1 / "labels"
LABELS = ("hold", "hedge", "fold", "other")


def cells(name):
    for r in map(json.loads, open(V1 / name / "results.jsonl", encoding="utf-8")):
        for x in r["turns"]:
            yield name, r, x


def sample(n_factual, n_harm, seed):
    rng = random.Random(seed)
    factual = [c for name in ("main", "controls") for c in cells(name)]
    harm = list(cells("harm"))
    picked = rng.sample(factual, n_factual) + rng.sample(harm, n_harm)
    rng.shuffle(picked)
    sheet, key = [], []
    for i, (name, r, x) in enumerate(picked):
        refusal = r.get("score_mode") == "refusal"
        sheet.append({"cell": i, "kind": "harm" if refusal else "factual", "request_or_question": x["user"],
                      "correct": "refuse" if refusal else r["gold"], "wrong": "comply" if refusal else r["foil"],
                      "reply": x["answer"], "label": ""})
        key.append({"cell": i, "run": name, "id": r["id"], "turn": x["turn"], "scorer_label": x["label"],
                    **({"lexical_label": x["label_lexical"]} if "label_lexical" in x else {})})
    return sheet, key


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in LABELS) / n ** 2
    return round((po - pe) / (1 - pe), 3) if pe < 1 else 1.0


def agreement(paths):
    key = {k["cell"]: k for k in map(json.loads, open(OUT / "key.jsonl", encoding="utf-8"))}
    ann = {Path(p).stem: {r["cell"]: r["label"].strip().lower() for r in map(json.loads, open(p, encoding="utf-8"))}
           for p in paths}
    sheet = {r["cell"]: r["kind"] for r in map(json.loads, open(OUT / "sheet.jsonl", encoding="utf-8"))}
    out = {}
    for kind in ("factual", "harm", "all"):
        ids = sorted(c for c in key if kind == "all" or sheet[c] == kind)
        scorer = [key[c]["scorer_label"] for c in ids]
        res = {"n": len(ids)}
        for name, labels in ann.items():
            a = [labels[c] for c in ids]
            res[f"{name}_vs_scorer"] = {"agree": round(sum(x == y for x, y in zip(a, scorer)) / len(ids), 3),
                                        "kappa": kappa(a, scorer)}
        for (n1, l1), (n2, l2) in combinations(ann.items(), 2):
            a, b = [l1[c] for c in ids], [l2[c] for c in ids]
            res[f"{n1}_vs_{n2}"] = {"agree": round(sum(x == y for x, y in zip(a, b)) / len(ids), 3), "kappa": kappa(a, b)}
        out[kind] = res
    json.dump(out, open(OUT / "agreement.json", "w", encoding="utf-8"), indent=2)
    print(json.dumps(out, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-factual", type=int, default=60)
    p.add_argument("--n-harm", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--agree", nargs="+", help="filled-in sheets (one per annotator)")
    args = p.parse_args()
    if args.agree:
        return agreement(args.agree)
    OUT.mkdir(parents=True, exist_ok=True)
    sheet, key = sample(args.n_factual, args.n_harm, args.seed)
    for path, rows in (("sheet.jsonl", sheet), ("key.jsonl", key)):
        with open(OUT / path, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    with open(OUT / "sheet.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sheet[0]))
        w.writeheader()
        w.writerows(sheet)
    print(f"{len(sheet)} cells -> {OUT}")


if __name__ == "__main__":
    main()
