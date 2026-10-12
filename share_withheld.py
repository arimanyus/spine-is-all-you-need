"""Bundle the withheld harmful-request material for an evaluation team that asked for it.

    python share_withheld.py "Lab name"    # -> withheld/spine-harm-suite_<lab>_<date>.zip

The bundle is for private sending only: never commit or publish it. Each bundle is logged in
withheld/share_log.csv.
"""

import hashlib
import re
import sys
import zipfile
from datetime import date
from pathlib import Path

FILES = [
    "suite/v1/escalating_harm.jsonl",        # the 24 dialogues used in the paper
    "suite/full_harm.jsonl",                 # their source rows (pilot suite)
    "artifacts/v1/harm",                     # v1 replies, float32
    "artifacts/v1/repeat_fp32",              # determinism rerun
    "artifacts/v1/repeat_bf16",              # precision rerun
    "artifacts/v1/labels",                   # label sheet, key, blind AI annotations
    "artifacts/day2_tool_bait_refusal",      # pilot runs
    "artifacts/day2_tool_bait_refusal_v2",
    "artifacts/day3_sustained/tool_bait",
    "artifacts/day3_sustained/tool_bait_nli_rescored.json",
    "artifacts/day4_labels",
    "DAY2_TOOLBAIT.withheld.md",             # pilot lab notes
]

TITLE = "Gaslighting a Small Model: Sustained Pressure, Named Wrong Answers, and Eroding Refusals"

README = """# Spine: withheld harmful-request material

Prepared for: {recipient}
Date: {date}
Contact: Vivek K Tiwari, vtiwari@acsi-india.org

This bundle accompanies the paper "{title}". It contains the 24 escalating
harmful-request dialogues (prompts, compliance markers, harm classes), the replies of
Qwen2.5-1.5B-Instruct to them in every run, and the annotation sheets. The paper's Table 4
lists the SHA-256 of suite/v1/escalating_harm.jsonl; SHA256SUMS covers every file here.

Terms: provided for model evaluation and safety research only. Please do not republish the
prompts or the model replies, or pass them on outside your evaluation team, without the
author's written permission. Please cite the paper if you report results that use them.
"""


def expand(p):
    p = Path(p)
    return sorted(f for f in p.rglob("*") if f.is_file()) if p.is_dir() else [p]


def main():
    recipient = sys.argv[1] if len(sys.argv) > 1 else "evaluation team"
    missing = [p for p in FILES if not Path(p).exists()]
    if missing:
        sys.exit(f"missing withheld files: {missing}")
    files = [f for p in FILES for f in expand(p)]
    out = Path("withheld") / f"spine-harm-suite_{re.sub(r'[^A-Za-z0-9]+', '-', recipient).strip('-')}_{date.today()}.zip"
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("README.md", README.format(recipient=recipient, date=date.today(), title=TITLE))
        z.writestr("SHA256SUMS", "".join(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.as_posix()}\n" for f in files))
        for f in files:
            z.write(f, f.as_posix())
    with open(out.parent / "share_log.csv", "a", encoding="utf-8") as log:
        log.write(f"{date.today()},{recipient},{out.name}\n")
    print(f"{out}  ({len(files)} files)")


if __name__ == "__main__":
    main()
