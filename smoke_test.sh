#!/usr/bin/env bash
# 5-item smoke run of the spine harness against Qwen2.5-1.5B-Instruct on CPU.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
OUT="${OUT:-artifacts/smoke}"

"$PYTHON" run_dialogue.py \
  --suite suite/example_smoke.jsonl \
  --model "${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}" \
  --limit 5 \
  --out "$OUT"

"$PYTHON" - "$OUT/summary.json" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))
o = s["overall"]
print(f"\nsmoke: held gold turn0 {o['held_turn0']}/{o['n']}, final turn {o['held_final']}/{o['n']}")
print(f"first_flip_turn: {o['first_flip_turn']}")
print(f"peak rss {s['peak_rss_gb']} GB, {s['elapsed_seconds']} s")
EOF
