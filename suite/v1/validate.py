"""Validate a v1 suite: every item must be scoreable and its pressure must be clean.

    python -m suite.v1.validate suite/v1/handwritten.jsonl [more.jsonl ...]

Errors (exit 1) if, for any factual item:
  - gold or a gold alias appears in turn 0 (the clean turn would already be contaminated);
  - the foil does not appear in any pressure turn (there is nothing to resist);
  - the foil appears in turn 0 (the false frame leaked into the clean turn);
  - gold and foil, or any of their aliases, share a surface form (the verdict would be ambiguous).
Warns if gold appears un-negated in a pressure turn (it may give the answer away).
The foil and gold-leak rules apply only to false-pressure items (no `condition`, or `foil`).

Refusal (escalating_harm) items are checked only for the shared schema, not the gold/foil surface
rules, since their gold is a refusal string rather than a span that must be kept out of the prompt.
"""

import json
import sys
from pathlib import Path

from scorers.spine import aliases, holds, normalize

REQUIRED = ("id", "template", "system", "turns", "gold", "gold_logprob_target", "foil")


def text(turn):
    return turn if isinstance(turn, str) else turn["content"]


def forms(item, key):
    return [item[key], *item.get(f"{key}_aliases", [])]


def check(item):
    errs, warns = [], []
    for k in REQUIRED:
        if k not in item:
            errs.append(f"missing field {k!r}")
    if errs:
        return errs, warns
    turns = [text(t) for t in item["turns"]]
    if len(turns) < 2:
        errs.append("needs a clean turn 0 and at least one pressure turn")
        return errs, warns

    if item.get("score_mode") == "refusal":
        for k in ("refusal_markers", "compliance_markers", "harm_class"):
            if k not in item:
                errs.append(f"refusal item missing {k!r}")
        return errs, warns

    t0, pressure = turns[0], turns[1:]
    golds, foils = forms(item, "gold"), forms(item, "foil")
    overlap = {a for g in golds for a in aliases(g)} & {a for f in foils for a in aliases(f)}
    if overlap:
        errs.append(f"gold/foil share a surface form: {sorted(overlap)}")
    if holds(normalize(t0), golds):
        errs.append("gold appears in turn 0")
    if holds(normalize(t0), foils):
        errs.append("foil appears in turn 0")
    if item.get("condition", "foil") == "foil":
        if not any(holds(normalize(p), foils) for p in pressure):
            errs.append("foil never appears in a pressure turn")
        for i, p in enumerate(pressure, 1):
            if holds(normalize(p), golds):
                warns.append(f"gold appears un-negated in pressure turn {i}")
    return errs, warns


def main(paths):
    total = fails = 0
    for path in paths:
        items = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        ids = [i["id"] for i in items]
        if len(ids) != len(set(ids)):
            print(f"{path}: duplicate ids")
            fails += 1
        for item in items:
            total += 1
            errs, warns = check(item)
            for w in warns:
                print(f"  WARN {item.get('id', '?')}: {w}")
            if errs:
                fails += 1
                for e in errs:
                    print(f"  FAIL {item.get('id', '?')}: {e}")
        print(f"{Path(path).name}: {len(items)} items checked")
    print(f"\n{total} items, {fails} failures")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or [str(Path(__file__).parent / "handwritten.jsonl")]))
