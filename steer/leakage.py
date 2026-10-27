"""Leakage report: search sets vs scored sets from Kenji's probe branch (ids and template names only).

    steer/.venv/bin/python steer/leakage.py   # writes steer/LEAKAGE.md and steer/leakage.json

Pairs: spine_contrast (search) vs pressure_probes (scored); contrast_prompts (search) vs
refusal_probes (scored). Per scored item: max word-token Jaccard and max word-3-gram Jaccard
against any search row (text of all user turns), plus the id of the closest search row.
Template overlap: scored items whose `template` field also appears in the search set, and (stricter)
whose template *family* by id/source_id prefix (fp-/mr-/tb-) appears in the search set.

Probe tree: `research/cursor/steering-probes-clean-5853` (arimanyus/steering_research draft PR #3):
    git remote add research https://github.com/arimanyus/steering_research.git
    git fetch research cursor/steering-probes-clean-5853
The tree holds refusal_probes.ids.jsonl only; refusal text comes from the local gitignored
steer/private/refusal_probes.jsonl when present (Jaccard columns are n/a without it).
"""

import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF = sys.argv[1] if len(sys.argv) > 1 else "research/cursor/steering-probes-clean-5853"
REFUSAL_TEXT = HERE / "private" / "refusal_probes.jsonl"  # withheld; local only
# template *family* from the id / source_id prefix: fp-* and mr-* items are false-premise / memory-rewrite
# dialogues whatever their `template` field says (the search tree renames them *_search).
FAMILY = {"fp": "false_premise", "mr": "memory_rewrite", "tb": "tool_bait"}


def load(name):
    text = subprocess.check_output(["git", "show", f"{REF}:steer/probes/{name}.jsonl"], cwd=HERE.parent, text=True)
    return [json.loads(l) for l in text.splitlines() if l.strip()]


def text_of(row):
    if "prompt" in row:
        return row["prompt"]
    if "turns" not in row:
        return None
    return " ".join(t["content"] if isinstance(t, dict) else t for t in row["turns"])


def tokens(s):
    return re.findall(r"[a-z0-9']+", s.lower())


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 0.0


def ngrams(toks, n):
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def family_of(row):
    return FAMILY.get(row.get("source_id", row["id"])[:2])


def compare(search, scored):
    s_feats = [(r["id"], set(tokens(text_of(r))), ngrams(tokens(text_of(r)), 3)) for r in search if text_of(r) is not None]
    s_templates = {r.get("template") for r in search} - {None}
    s_families = {family_of(r) for r in search} - {None}
    items = []
    for r in scored:
        item = {"id": r["id"], "template": r.get("template"), "template_in_search": r.get("template") in s_templates,
                "family": family_of(r), "family_in_search": family_of(r) in s_families,
                "max_token_jaccard": None, "closest_by_tokens": None, "max_3gram_jaccard": None, "closest_by_3gram": None}
        if text_of(r) is not None:
            toks = tokens(text_of(r))
            tset, tri = set(toks), ngrams(toks, 3)
            best_tok = max(s_feats, key=lambda f: jaccard(tset, f[1]))
            best_tri = max(s_feats, key=lambda f: jaccard(tri, f[2]))
            item.update({"max_token_jaccard": round(jaccard(tset, best_tok[1]), 3), "closest_by_tokens": best_tok[0],
                         "max_3gram_jaccard": round(jaccard(tri, best_tri[2]), 3), "closest_by_3gram": best_tri[0]})
        items.append(item)
    return {"n_search": len(search), "n_search_with_text": len(s_feats), "n_scored": len(scored),
            "n_scored_with_text": sum(i["max_token_jaccard"] is not None for i in items), "search_templates": sorted(s_templates),
            "search_families": sorted(s_families),
            "scored_with_template_in_search": sum(i["template_in_search"] for i in items),
            "scored_with_family_in_search": sum(i["family_in_search"] for i in items),
            "shared_ids": sorted({r["id"] for r in search} & {r["id"] for r in scored}),
            "shared_source_ids": sorted({r.get("source_id") for r in search} & {r["id"] for r in scored}),
            "items": items}


def main():
    sha = subprocess.check_output(["git", "rev-parse", REF], cwd=HERE.parent, text=True).strip()
    refusal = load("refusal_probes.ids")
    if REFUSAL_TEXT.exists():
        turns_by_id = {json.loads(l)["id"]: json.loads(l)["turns"] for l in REFUSAL_TEXT.read_text().splitlines() if l.strip()}
        refusal = [{**r, "turns": turns_by_id[r["id"]]} for r in refusal if r["id"] in turns_by_id]
    report = {"probes_ref": REF, "probes_commit": sha, "refusal_text_available_locally": REFUSAL_TEXT.exists(),
              "spine_contrast_vs_pressure_probes": compare(load("spine_contrast"), load("pressure_probes")),
              "contrast_prompts_vs_refusal_probes": compare(load("contrast_prompts"), refusal)}
    (HERE / "leakage.json").write_text(json.dumps(report, indent=1))

    lines = ["# Leakage report: neuron-search sets vs scored sets", "",
             f"Probe files from `{REF}` @ `{sha[:12]}` (not copied into this branch). Ids and template names only; "
             "no prompt text. Generated by `steer/leakage.py`; numbers in `steer/leakage.json`. The tree carries "
             f"`refusal_probes.ids.jsonl` only; refusal prompt text for the Jaccard columns was "
             f"{'resolved from a local gitignored copy' if report['refusal_text_available_locally'] else 'NOT available (columns n/a)'}.", ""]
    for key, title in (("spine_contrast_vs_pressure_probes", "spine_contrast (search) vs pressure_probes (scored)"),
                       ("contrast_prompts_vs_refusal_probes", "contrast_prompts (search) vs refusal_probes (scored)")):
        c = report[key]
        lines += [f"## {title}", "",
                  f"- search rows: {c['n_search']} ({c['n_search_with_text']} with prompt text in the tree); scored items: {c['n_scored']} "
                  f"({c['n_scored_with_text']} with text available for Jaccard)",
                  f"- shared ids: {c['shared_ids'] or 'none'}; scored ids appearing as a search `source_id`: {c['shared_source_ids'] or 'none'}",
                  f"- `template` values in search set: {c['search_templates'] or 'none (no `template` field)'}; "
                  f"template families by id prefix (fp/mr/tb): {c['search_families'] or 'none'}",
                  f"- **scored items whose `template` also appears in the search set: {c['scored_with_template_in_search']}/{c['n_scored']}**; "
                  f"whose template *family* (id prefix) appears in the search set: {c['scored_with_family_in_search']}/{c['n_scored']}", "",
                  "| scored id | template | template in search | family in search | max token Jaccard (closest) | max 3-gram Jaccard (closest) |",
                  "|---|---|---|---|---|---|"]
        for i in c["items"]:
            lines.append(f"| {i['id']} | {i['template']} | {i['template_in_search']} | {i['family_in_search']} | {i['max_token_jaccard']} ({i['closest_by_tokens']}) "
                         f"| {i['max_3gram_jaccard']} ({i['closest_by_3gram']}) |")
        lines.append("")
    lines += ["## Reading", "",
              "Ids and `template` field values are disjoint by construction (Kenji's `template_overlap.md`). The family "
              "column is the stricter check: `spine_contrast` is still built from fp-*/mr-* false-premise / memory-rewrite "
              "dialogues, i.e. the same two template *families* as `pressure_probes`, under renamed templates with a "
              "reworded scaffold; the Jaccard columns measure how far that rewording goes. Neuron rankings stay EXPLORATORY "
              "until Asha's scorer confirms zero overlap against this tree. The smoke-run search in `raw.json` is worse "
              "still (it used the scored items themselves) and is labelled EXPLORATORY."]
    (HERE / "LEAKAGE.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "items"} for k, v in report.items() if isinstance(v, dict)}, indent=1))


if __name__ == "__main__":
    main()
