"""Shared scoring loop for the direction-ablation and neuron-sweep runs (scores only; all generated
text goes to steer/private/, which is gitignored).

An *intervention* is any object with `enable()` / `disable()`. `steer_turns="pressure"` enables it
only for dialogue turns >= 1 (the pressure / bait turns) and for single-turn factual prompts;
`"all"` enables it everywhere. Perplexity of every generation is scored under the unsteered model
(intervention disabled) right after each condition.
"""

import hashlib
import json
import statistics
import time
from pathlib import Path

from steer import crude_refused, load_jsonl, load_probe, rss_gb, timing
from scorers.spine import FLIP, holds, marker_hits, normalize, refusal_label, spine_label

HERE = Path(__file__).resolve().parent


class NoOp:
    def enable(self):
        pass

    def disable(self):
        pass


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


class Eval:
    def __init__(self, H, probes_ref, refusal_text, n_refusal=8, n_factual=10, n_pressure=12):
        self.H = H
        self.factual = load_probe(probes_ref, "factual_probes")[:n_factual]
        self.pressure = load_probe(probes_ref, "pressure_probes")[:n_pressure]
        meta = load_probe(probes_ref, "refusal_probes.ids")
        self.refusal, self.refusal_source = [], {"path": str(refusal_text), "available": Path(refusal_text).exists()}
        if self.refusal_source["available"]:
            text = {r["id"]: r["turns"] for r in load_jsonl(refusal_text)}
            self.refusal_source["sha256"] = hashlib.sha256(Path(refusal_text).read_bytes()).hexdigest()
            self.refusal = [{**m, "turns": text[m["id"]]} for m in meta if m["id"] in text][:n_refusal]
        self.sets_meta = {"factual": [r["id"] for r in self.factual], "pressure": [r["id"] for r in self.pressure],
                          "refusal": [r["id"] for r in self.refusal], "refusal_text_source": self.refusal_source}
        self.private = {}  # condition -> set -> id -> texts (never committed)

    def run(self, name, intervention, steer_turns="all", sets=("factual", "pressure", "refusal"), log=print):
        H, t_start = self.H, time.monotonic()
        res, gens, secs, toks = {"factual": {}, "pressure": {}, "refusal": {}}, [], [], []
        priv = self.private.setdefault(name, {"factual": {}, "pressure": {}, "refusal": {}})

        def gen(hist, active):
            (intervention.enable if active else intervention.disable)()
            g = H.generate(hist)
            secs.append(g["seconds"]); toks.append(g["n_tokens"])
            return g

        if "factual" in sets:
            for item in self.factual:
                hist = [{"role": "user", "content": item["prompt"]}]
                g = gen(hist, True)
                gens.append(("factual", item["id"], None, hist, g["ids"]))
                res["factual"][item["id"]] = {"correct": holds(normalize(g["text"]), [item["gold"], *item.get("aliases", [])]),
                                              "n_tokens": g["n_tokens"], "seconds": g["seconds"], "truncated": g["truncated"]}
                priv["factual"][item["id"]] = g["text"]
                log(f"[{name}] factual/{item['id']} {g['n_tokens']} tok {g['seconds']:.1f}s")
        for set_name in ("pressure", "refusal"):
            if set_name not in sets:
                continue
            for item in getattr(self, set_name):
                hist, turns, texts = [], [], []
                for ti, msg in enumerate(item["turns"]):
                    hist.append({"role": msg["role"], "content": msg["content"]})
                    active = steer_turns == "all" or ti >= 1
                    g = gen(hist, active)
                    gens.append((set_name, item["id"], ti, list(hist), g["ids"]))
                    rec = {"turn": ti, "steered": active, "n_tokens": g["n_tokens"], "seconds": g["seconds"], "truncated": g["truncated"]}
                    if set_name == "pressure":
                        (intervention.enable if active else intervention.disable)()
                        rec.update({"label": spine_label(g["text"], item["gold"], item["foil"]),
                                    **H.delta_logprob(hist, item["gold_logprob_target"], item["foil"])})
                    else:
                        refused_repo = bool(marker_hits(g["text"], item["refusal_markers"], ignore_case=True))
                        rec.update({"len_chars": len(g["text"]), "refused_crude": crude_refused(g["text"]),
                                    "refused_repo_lexical": refused_repo,
                                    "label_repo_lexical": refusal_label(refused_repo, g["text"], item["compliance_markers"])})
                    turns.append(rec); texts.append(g["text"])
                    hist.append({"role": "assistant", "content": g["text"]})
                f = turns[-1]
                res[set_name][item["id"]] = {"turns": turns, "final_label": f.get("label"), "final_delta_logprob": f.get("delta_logprob"),
                                             "final_refused_crude": f.get("refused_crude"), "final_refused_repo_lexical": f.get("refused_repo_lexical")}
                priv[set_name][item["id"]] = texts
                log(f"[{name}] {set_name}/{item['id']} {sum(t['n_tokens'] for t in turns)} tok {sum(t['seconds'] for t in turns):.1f}s rss {rss_gb():.2f}")
        intervention.disable()
        t_ppl = time.monotonic()
        for set_name, i, ti, hist, ids in gens:
            p = H.ppl(hist, ids)
            rec = res[set_name][i]
            (rec["turns"][ti] if ti is not None else rec)["ppl_unsteered"] = p
        return {"results": res, "summary": self.summarize(res, sets), "timing": timing(secs, toks),
                "wall_s": time.monotonic() - t_start, "ppl_pass_s": time.monotonic() - t_ppl,
                "peak_rss_gb": rss_gb(), "steer_turns": steer_turns, "sets": list(sets)}

    @staticmethod
    def summarize(res, sets):
        s = {}
        if "pressure" in sets and res["pressure"]:
            labels = [r["final_label"] for r in res["pressure"].values()]
            s.update({"pressure_n": len(labels), "final_labels": {l: labels.count(l) for l in ("hold", "hedge", "fold", "other")},
                      "fold_rate_final": labels.count("fold") / len(labels),
                      "mean_final_delta_logprob": mean(r["final_delta_logprob"] for r in res["pressure"].values()),
                      "mean_consistency_flip_final": mean(FLIP[l] for l in labels)})
        if "refusal" in sets and res["refusal"]:
            rs = list(res["refusal"].values())
            s.update({"refusal_n": len(rs), "refusal_rate_final_crude": mean(r["final_refused_crude"] for r in rs),
                      "refusal_rate_final_repo_lexical": mean(r["final_refused_repo_lexical"] for r in rs),
                      "refusal_rate_any_turn_crude": mean(any(t["refused_crude"] for t in r["turns"]) for r in rs)})
        if "factual" in sets and res["factual"]:
            s["factual_accuracy"] = mean(r["correct"] for r in res["factual"].values())
        ppls = [t["ppl_unsteered"] for k in sets for r in res[k].values() for t in (r["turns"] if "turns" in r else [r])
                if t.get("ppl_unsteered") is not None]
        s["mean_ppl_unsteered"] = mean(ppls)
        s["median_ppl_unsteered"] = statistics.median(ppls) if ppls else None
        return s

    def dump_private(self, path):
        Path(path).write_text(json.dumps(self.private, indent=1, ensure_ascii=False))
