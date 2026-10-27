"""Redact withheld content before an outputs/scored file is committed.

The refusal probes (tool_bait) are withheld from the public tree (share_withheld.py), so every
reply to them is replaced by a stub; scores and ids stay. Any other reply that looks like a
complied harmful request (a compliance marker hit with no refusal) is stubbed too.

    python -m steer.redact artifacts/steer/run1/scored.jsonl -o artifacts/steer/run1/scored.redacted.jsonl
"""

import argparse
import json

WITHHELD_SETS = ("refusal",)
WITHHELD_FIELDS = ("text", "judge_raw", "prompt", "ask")


def stub(text, reason):
    return f"[REDACTED: {reason}; {len(text)} chars]"


def redact_row(row, withheld_sets=WITHHELD_SETS):
    r = dict(row)
    reason = None
    if r.get("prompt_set") in withheld_sets:
        reason = "withheld tool_bait reply"
    elif r.get("refused") is False or r.get("refusal") is False:
        reason = "complied on a harmful-looking ask"
    if reason:
        for k in WITHHELD_FIELDS:
            if isinstance(r.get(k), str):
                r[k] = stub(r[k], reason)
        r["redacted"] = True
    return r


def redact_file(src, dst):
    n = 0
    with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as g:
        for line in f:
            if line.strip():
                r = redact_row(json.loads(line))
                n += r.get("redacted", False)
                g.write(json.dumps(r) + "\n")
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args(argv)
    print(f"redacted {redact_file(a.src, a.out)} rows -> {a.out}")


if __name__ == "__main__":
    main()
