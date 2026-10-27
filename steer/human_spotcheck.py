"""Blind human spot-check of the persona judge (secondary endpoint).

    python -m steer.human_spotcheck sample scored.jsonl --out spotcheck/      # blind.csv + key.csv
    python -m steer.human_spotcheck kappa spotcheck/key.csv spotcheck/blind.csv  # after a human fills `human`

`sample` draws 40 persona-probe replies stratified by condition, shuffles them and writes
`blind.csv` (blind_id, text, human) with the condition hidden and `key.csv` (blind_id, condition,
pair_id, dose, prompt_id, judge). `kappa` joins the two on blind_id and reports Cohen's kappa
(unweighted, categories 0-3); the judge is valid only if kappa >= 0.6. Rows the human left blank or
the judge could not parse are dropped and counted.
"""

import argparse
import csv
import json
import os
import random
from collections import defaultdict

from steer.rubric import UNPARSEABLE, parse_judge

N_SAMPLE = 40
KAPPA_MIN = 0.6


def sample_spotcheck(rows, n=N_SAMPLE, seed=0, prompt_set="persona"):
    """`n` rows of `prompt_set`, as evenly as possible across conditions (round-robin over shuffled
    strata), then shuffled so order carries no condition information."""
    rng = random.Random(seed)
    strata = defaultdict(list)
    for r in rows:
        if r["prompt_set"] == prompt_set:
            strata[r["condition"]].append(r)
    for rs in strata.values():
        rng.shuffle(rs)
    picked = []
    while len(picked) < n and any(strata.values()):
        for c in sorted(strata):
            if strata[c] and len(picked) < n:
                picked.append(strata[c].pop())
    rng.shuffle(picked)
    return picked


def write_spotcheck(picked, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "blind.csv"), "w", newline="", encoding="utf-8") as b, \
            open(os.path.join(out_dir, "key.csv"), "w", newline="", encoding="utf-8") as k:
        blind = csv.DictWriter(b, fieldnames=["blind_id", "text", "human"])
        key = csv.DictWriter(k, fieldnames=["blind_id", "condition", "pair_id", "dose", "prompt_id", "judge"])
        blind.writeheader()
        key.writeheader()
        for i, r in enumerate(picked):
            bid = f"S{i:02d}"
            judge = r.get("judge", parse_judge(r.get("judge_raw")))
            blind.writerow({"blind_id": bid, "text": r["text"], "human": ""})
            key.writerow({"blind_id": bid, "condition": r["condition"], "pair_id": r["pair_id"], "dose": r["dose"],
                          "prompt_id": r["prompt_id"], "judge": "" if judge in (None, UNPARSEABLE) else judge})


def cohens_kappa(a, b, categories=(0, 1, 2, 3)):
    """Unweighted Cohen's kappa between two equal-length label sequences."""
    if len(a) != len(b) or not a:
        raise ValueError("need two non-empty sequences of equal length")
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in categories)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def kappa_report(key_rows, blind_rows):
    human = {r["blind_id"]: r["human"].strip() for r in blind_rows}
    pairs, dropped = [], {"human_blank": 0, "judge_unparseable": 0}
    for r in key_rows:
        h, j = human.get(r["blind_id"], ""), r["judge"]
        if h not in {"0", "1", "2", "3"}:
            dropped["human_blank"] += 1
        elif j == "":
            dropped["judge_unparseable"] += 1
        else:
            pairs.append((int(j), int(h)))
    kappa = cohens_kappa([j for j, _ in pairs], [h for _, h in pairs]) if pairs else None
    return {"n": len(pairs), "dropped": dropped, "kappa": kappa,
            "valid": None if kappa is None else kappa >= KAPPA_MIN,
            "exact_agreement": sum(j == h for j, h in pairs) / len(pairs) if pairs else None}


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("scored")
    s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, default=N_SAMPLE)
    s.add_argument("--seed", type=int, default=0)
    k = sub.add_parser("kappa")
    k.add_argument("key")
    k.add_argument("blind")
    k.add_argument("--out")
    a = ap.parse_args(argv)
    if a.cmd == "sample":
        with open(a.scored, encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
        write_spotcheck(sample_spotcheck(rows, a.n, a.seed), a.out)
        print(f"wrote {a.out}/blind.csv and key.csv")
    else:
        rep = kappa_report(_read_csv(a.key), _read_csv(a.blind))
        if a.out:
            with open(a.out, "w", encoding="utf-8") as f:
                json.dump(rep, f, indent=1)
        print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
