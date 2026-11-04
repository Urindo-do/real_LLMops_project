"""Compare the experiment-card run with the 11/2 v3 snapshot on the six problems in WORKFLOW2_EVIDENCE_APPROACH.md.

No API calls. Reads results/cards.json, cached card responses under draft_history/, and the v3 snapshot ZIP.
"""

import collections
import json
import statistics
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation/extraction/comprehensive"))

from policy_report import DEFAULT_ZIP, load_from_zip, reference_proxy  # noqa: E402

HERE = Path(__file__).resolve().parent


def usage_stats(bodies):
    prompt, completion = [], []
    for body in bodies:
        usage = body.get("usage") if isinstance(body, dict) else None
        if usage:
            prompt.append(usage["prompt_tokens"])
            completion.append(usage["completion_tokens"])
    if not prompt:
        return {"calls_with_usage": 0}
    return {"calls_with_usage": len(prompt), "prompt_tokens_total": sum(prompt), "completion_tokens_total": sum(completion),
            "prompt_tokens_mean": round(statistics.mean(prompt)), "prompt_tokens_max": max(prompt)}


def v3_usage(path):
    archive = zipfile.ZipFile(path)
    bodies = []
    for name in archive.namelist():
        if "/research-evidence-v3/" in name and name.endswith(".native.json"):
            try:
                bodies.append(json.loads(archive.read(name)))
            except json.JSONDecodeError:
                pass
    return usage_stats(bodies)


def card_usage(sha):
    folder = ROOT / "draft_history" / "experiment-cards" / sha[:12] / "responses"
    return usage_stats(json.loads(p.read_text(encoding="utf-8")) for p in folder.glob("*.native*.json"))


def main():
    cards = json.loads((HERE / "results" / "cards.json").read_text(encoding="utf-8"))
    policy = json.loads((ROOT / "evaluation/extraction/comprehensive/results/policy-report.json").read_text(encoding="utf-8"))
    v3, elements, _ = load_from_zip(DEFAULT_ZIP)
    comma = [f"{fact['id']}: {issue}" for fact in cards["items"] for issue in fact["numeric_issues"] if "소수 쉼표" in issue]
    report = {
        "B1_gate": {"v3_unresolved_original": policy["before"]["unresolved_source_items"],
                    "v3_blocking_after_gate_split": sum(policy["after"]["blocking"].values()),
                    "cards_blocking": len(cards["completion"]["blocking"]),
                    "cards_warnings": len(cards["completion"]["warnings"]),
                    "cards_ready_for_drafting": cards["completion"]["ready_for_drafting"]},
        "B2_context_per_call": {"v3": v3_usage(DEFAULT_ZIP), "cards": card_usage(cards["source"]["sha256"])},
        "B3_calls_and_time": {"v3_model_responses": None, "cards_calls": cards["diagnostics"]["calls"],
                              "cards_seconds": cards["diagnostics"]["seconds"]},
        "B4_reference_proxy": {"v3": reference_proxy(v3, elements)["counts"],
                               "cards": reference_proxy(cards, elements)},
        "B5_decimal_comma": {"cards_flags": comma},
        "B6_granularity": {"v3_items": len(v3["items"]), "v3_writing_facts": len(v3["writing_fact_ids"]),
                           "cards": len(cards["cards"]), "cards_writing": len(cards["writing_fact_ids"]),
                           "cards_kinds": cards["diagnostics"]["kinds"],
                           "cards_writing_roles": cards["diagnostics"]["writing_roles"]},
    }
    report["B3_calls_and_time"]["v3_model_responses"] = report["B2_context_per_call"]["v3"]["calls_with_usage"]
    (HERE / "results" / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    summary = {key: value for key, value in report.items()}
    summary["B4_reference_proxy"] = {"v3": report["B4_reference_proxy"]["v3"], "cards": report["B4_reference_proxy"]["cards"]["counts"]}
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
