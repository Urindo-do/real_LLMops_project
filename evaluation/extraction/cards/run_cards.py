"""Run experiment-card collection on the 11/2 journal snapshot read directly from the portable ZIP.

--dry-run (default) shows segments and estimated size without any API call.
--live sends the segment texts to Upstage Solar. Model responses are cached under draft_history/ (git-ignored).
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation/extraction/comprehensive"))

from experiment_cards import MAX_FEEDBACK_ROUNDS, build_segments, collect_cards, render_cards, source_view  # noqa: E402
from policy_report import DEFAULT_ZIP, load_from_zip  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default=str(DEFAULT_ZIP))
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--max-calls", type=int)
    args = parser.parse_args()
    extraction, elements, _ = load_from_zip(args.zip)
    pages, dropped = source_view(elements)
    segments = build_segments(pages)
    chars = [sum(len(e["text"]) for e in s["elements"]) for s in segments]
    plan = {"source_elements": len(elements), "kept_elements": sum(len(v) for v in pages.values()),
            "dropped_reread_or_blank": len(dropped), "segments": len(segments),
            "segment_chars": {"total": sum(chars), "max": max(chars), "min": min(chars)},
            "max_calls": args.max_calls or len(segments) * (1 + MAX_FEEDBACK_ROUNDS),
            "pages_per_segment": [f"p{s['pages'][0]}-{s['pages'][-1]}" for s in segments]}
    print(json.dumps(plan, ensure_ascii=False, indent=1))
    if not args.live:
        return
    from planned_paper import PlannedPaper
    engine = PlannedPaper(base_dir=ROOT)
    folder = ROOT / "draft_history" / "experiment-cards" / extraction["source"]["sha256"][:12]

    def progress(done, total, record):
        print(f"[{time.strftime('%H:%M:%S')}] 구간 {record['index']} {record['status']} "
              f"카드 {len(record['cards'])}개 ({done}/{total})", flush=True)

    result = collect_cards(engine, elements, folder, title=extraction["title"], source=extraction["source"],
                           max_calls=args.max_calls, progress=progress)
    result["plan"] = plan
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "cards.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    (RESULTS / "cards.md").write_text(render_cards(result), encoding="utf-8")
    print(json.dumps({"completion": {k: v if not isinstance(v, list) else len(v) for k, v in result["completion"].items()},
                      "diagnostics": result["diagnostics"], "cards": len(result["cards"])}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
