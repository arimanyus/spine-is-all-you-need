"""Offline: re-score saved refusal-mode answers with `refusal_flip_nli` and compare to the lexical
`refusal_flip` already in the file. No victim model is run; only the NLI classifier is loaded.

    python rescore_refusal_nli.py artifacts/day2_tool_bait_refusal_v2/results.jsonl \
        --out artifacts/day2_tool_bait_refusal_v2/nli_rescored.json
"""

import argparse
import json
from collections import Counter

from scorers.spine import first_sentence, load_refusal_nli, nli_refusal_probs, refusal_flip, refusal_flip_nli


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results")
    p.add_argument("--out", required=True)
    p.add_argument("--nli-checkpoint", default=None)
    args = p.parse_args()

    nli = load_refusal_nli(args.nli_checkpoint)
    if nli is None:
        raise SystemExit("NLI checkpoint unavailable")

    cells, pairs = [], Counter()
    for line in open(args.results):
        item = json.loads(line)
        if item.get("score_mode") != "refusal":
            continue
        for t in item["turns"]:
            head = first_sentence(t["answer"])
            probs = nli_refusal_probs(nli, head, t["user"])
            lexical = refusal_flip(t["answer"], item["refusal_markers"], item["compliance_markers"])
            flip = refusal_flip_nli(t["answer"], t["user"], item["compliance_markers"], nli=nli)
            pairs[(lexical, flip)] += 1
            cells.append({"id": item["id"], "turn": t["turn"], "lexical": lexical, "nli": flip,
                          "p_refuse": round(probs["refuse"], 3), "p_comply": round(probs["comply"], 3),
                          "unmarked": not t["refusal_hits"] and not t["compliance_hits"],
                          "first_sentence": head, "answer": t["answer"]})

    out = {"n_cells": len(cells), "lexical_vs_nli": {f"{k[0]}->{k[1]}": v for k, v in sorted(pairs.items())},
           "cells": cells}
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "cells"}, indent=1))
    for c in cells:
        if c["lexical"] != c["nli"]:
            print(f'{c["id"]} t{c["turn"]} lex={c["lexical"]} nli={c["nli"]} refuse={c["p_refuse"]:.2f} '
                  f'comply={c["p_comply"]:.2f} unmarked={c["unmarked"]} | {c["first_sentence"][:110]!r}')


if __name__ == "__main__":
    main()
