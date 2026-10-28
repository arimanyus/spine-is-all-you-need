"""Follow-up to hunt.py: factual accuracy for the matched random draws at each arm's smallest unlocking k
(the main run scores random draws on refusal only), so "specific unlock" can be told from generic damage.
Updates steer/raw_hunt.json in place (adds summary.factual_accuracy + results.factual to those draws).

    steer/.venv/bin/python steer/hunt_followup.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from steer import MODEL_ID, REVISION, Harness, load_probe, materialize, rss_gb  # noqa: E402
from hunt import Neurons  # noqa: E402
from scorers.spine import holds, normalize  # noqa: E402


def main():
    path = HERE / "raw_hunt.json"
    raw = json.loads(path.read_text())
    todo = []
    for arm, u in raw["unlock"].items():
        k = u.get("smallest_k")
        if k is not None:
            todo += [f"{arm}_random{k}_draw{d}" for d in range(raw["meta"]["n_random_draws"])]
    todo = [n for n in todo if n in raw["conditions"] and raw["conditions"][n]["summary"].get("factual_accuracy") is None]
    print("scoring factual for", todo, flush=True)
    if not todo:
        return
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION, dtype=torch.bfloat16).eval()
    materialize(model)
    H = Harness(model, tok)
    H.resid_median = json.loads(Path(raw["meta"]["calib"]).read_text())["median_resid_norm"]
    factual = [r for r in load_probe(raw["meta"]["probes_ref"], "factual_probes") if r["id"] in raw["meta"]["sets"]["factual"]]
    private = json.loads((HERE / "private" / "hunt_generations_full.json").read_text())
    for name in todo:
        c = raw["conditions"][name]
        iv = Neurons(H, [(n["layer"], n["neuron"]) for n in c["neurons"]], c["intervention"])
        t0 = time.monotonic()
        iv.enable()
        rec, texts = {}, {}
        for item in factual:
            g = H.generate([{"role": "user", "content": item["prompt"]}])
            rec[item["id"]] = {"correct": holds(normalize(g["text"]), [item["gold"], *item.get("aliases", [])]),
                               "n_tokens": g["n_tokens"], "seconds": g["seconds"]}
            texts[item["id"]] = g["text"]
        iv.disable()
        c["results"]["factual"] = rec
        c["summary"]["factual_accuracy"] = float(np.mean([r["correct"] for r in rec.values()]))
        c["summary"]["factual_scored_in"] = "hunt_followup.py"
        c["wall_s"] += time.monotonic() - t0
        c["peak_rss_gb"] = max(c["peak_rss_gb"], rss_gb())
        private.setdefault(name, {})["factual"] = texts
        print(name, c["summary"]["factual_accuracy"], flush=True)
        path.write_text(json.dumps(raw, indent=1))
        (HERE / "private" / "hunt_generations_full.json").write_text(json.dumps(private, indent=1, ensure_ascii=False))
    for arm, u in raw["unlock"].items():
        k = u.get("smallest_k")
        if k is not None:
            u["random_factual_at_smallest_k"] = [raw["conditions"][f"{arm}_random{k}_draw{d}"]["summary"]["factual_accuracy"]
                                                 for d in range(raw["meta"]["n_random_draws"])]
    path.write_text(json.dumps(raw, indent=1))


if __name__ == "__main__":
    main()
