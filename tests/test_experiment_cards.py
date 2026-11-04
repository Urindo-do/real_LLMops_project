import unittest
from pathlib import Path

from experiment_cards import (CallBudget, build_segments, collect_cards, source_view, uncited_numeric_elements,
                              verify_card)

ROOT = Path(__file__).resolve().parents[1]


def element(element_id, page, text, category="paragraph", origin="standard_parse"):
    return {"element_id": element_id, "page": page, "text": text, "category": category, "origin": origin}


ELEMENTS = {e["element_id"]: e for e in [
    element("p1:e1", 1, "v1 GPU 버전 실행 결과 CROWN 검증 시간이 355ms로 측정됨"),
    element("p1:e2", 1, "======"),
    element("p1:review:e1", 1, "v1 GPU 버전 실행 결과 CROWN 검증 시간이 355ms로 측정됨", origin="enhanced_numeric_ocr_review"),
    element("p2:e1", 2, "total elapsed 1560.37s, bound 131,248s (8.4%)"),
    element("p2:e2", 2, "다음에는 타일링을 시도할 계획"),
]}


def card(**overrides):
    value = {"title": "GPU 이식", "summary": "v1 GPU 버전의 검증 시간을 측정했다.", "kind": "result",
             "claim_status": "observed", "source_origin": "research_note", "goal": "", "change": "",
             "setup": [], "results": [{"metric": "CROWN 검증 시간", "value": "355", "unit": "ms",
                                       "condition": "v1 GPU", "element_id": "p1:e1"}],
             "interpretation": "", "limits": "", "paper_roles": ["result"], "element_ids": ["p1:e1"]}
    value.update(overrides)
    return value


class FakeEngine:
    def __init__(self, responses):
        self.responses, self.requests = responses, []

    def call(self, folder, name, system, packet, schema, inference_settings=None):
        self.requests.append((name, packet))
        response = self.responses[name]
        if isinstance(response, Exception):
            raise response
        return response


class ExperimentCardTest(unittest.TestCase):
    def test_source_view_drops_rereads_and_separators_only(self):
        pages, dropped = source_view(ELEMENTS)
        self.assertEqual(dropped, {"p1:e2": None, "p1:review:e1": "p1:e1"})
        self.assertEqual([e["element_id"] for e in pages[2]], ["p2:e1", "p2:e2"])

    def test_reread_with_new_number_is_kept(self):
        elements = {**ELEMENTS, "p1:review:e1": element("p1:review:e1", 1, "v1 GPU 버전 실행 결과 CROWN 검증 시간이 353ms로 측정됨")}
        pages, _ = source_view(elements)
        self.assertIn("p1:review:e1", [e["element_id"] for e in pages[1]])

    def test_segments_respect_size_and_page_order(self):
        pages, _ = source_view(ELEMENTS)
        segments = build_segments(pages, limit=60)
        self.assertEqual([s["pages"] for s in segments], [[1], [2]])

    def test_verify_keeps_grounded_value_and_rejects_invented_one(self):
        checked = verify_card(card(results=[
            {"metric": "시간", "value": "355", "unit": "ms", "condition": "v1", "element_id": "p1:e1"},
            {"metric": "시간", "value": "300", "unit": "ms", "condition": "v1", "element_id": "p1:e1"}]), ELEMENTS)
        self.assertEqual([m["value"] for m in checked["measurements"]], ["355"])
        self.assertEqual(checked["rejected_results"][0]["value"], "300")
        self.assertEqual(checked["numeric_issues"], [])

    def test_verify_flags_computed_number_in_summary(self):
        checked = verify_card(card(summary="CPU보다 약 3.7배 느렸다."), ELEMENTS)
        self.assertTrue(any("3.7" in issue for issue in checked["numeric_issues"]))

    def test_verify_flags_decimal_comma_without_rewriting(self):
        checked = verify_card(card(element_ids=["p2:e1"], results=[
            {"metric": "bound 시간", "value": "131,248", "unit": "s", "condition": "", "element_id": "p2:e1"}]), ELEMENTS)
        self.assertEqual(checked["measurements"][0]["raw_value"], "131,248")
        self.assertTrue(any("소수 쉼표" in issue for issue in checked["numeric_issues"]))

    def test_number_elsewhere_in_segment_is_citation_gap_not_exclusion(self):
        segment_text = "\n".join(e["text"] for e in ELEMENTS.values())
        checked = verify_card(card(summary="total elapsed 1560.37s 중 일부다."), ELEMENTS, segment_text)
        self.assertEqual(checked["numeric_issues"], [])
        self.assertEqual(len(checked["citation_gaps"]), 1)
        self.assertEqual(len(checked["feedback_issues"]), 1)
        invented = verify_card(card(summary="약 3.7배 느렸다."), ELEMENTS, segment_text)
        self.assertTrue(invented["numeric_issues"])

    def test_list_markers_and_element_refs_are_not_numbers(self):
        checked = verify_card(card(summary="(1) 측정했고 2) 비교했다. 표1과 같다 (p2:e1의 total elapsed 1560.37s)."), ELEMENTS)
        self.assertEqual(checked["numeric_issues"], [])
        self.assertIn("p2:e1", checked["element_ids"])

    def test_ocr_split_digits_are_accepted_with_note(self):
        elements = {**ELEMENTS, "p3:e1": element("p3:e1", 3, "7 87s ⇒ 630s로 157초 정도 줄었다")}
        checked = verify_card(card(summary="787s에서 630s로 줄었다.", results=[], element_ids=["p3:e1"]), elements)
        self.assertEqual(checked["numeric_issues"], [])
        self.assertEqual(len(checked["ocr_spacing_notes"]), 1)

    def test_missing_headline_total_is_reported(self):
        checked = verify_card(card(element_ids=["p2:e1"], results=[]), ELEMENTS)
        self.assertEqual(checked["missing_headlines"], ["p2:e1: total elapsed 1560.37s"])
        kept = verify_card(card(element_ids=["p2:e1"], results=[
            {"metric": "total elapsed", "value": "1560.37", "unit": "s", "condition": "", "element_id": "p2:e1"}]), ELEMENTS)
        self.assertEqual(kept["missing_headlines"], [])

    def test_uncited_numeric_elements(self):
        pages, _ = source_view(ELEMENTS)
        segment = build_segments(pages)[0]
        self.assertEqual(uncited_numeric_elements(segment, [verify_card(card(), ELEMENTS)]), ["p2:e1"])

    def test_budget_and_circuit_breaker(self):
        budget = CallBudget(2, max_consecutive_failures=1)
        budget.take()
        budget.record(False)
        with self.assertRaises(RuntimeError):
            budget.take()

    def test_collect_runs_one_feedback_round_and_only_failures_block(self):
        fixed = card(element_ids=["p1:e1", "p2:e1"], results=[
            {"metric": "시간", "value": "355", "unit": "ms", "condition": "v1", "element_id": "p1:e1"},
            {"metric": "total elapsed", "value": "1560.37", "unit": "s", "condition": "", "element_id": "p2:e1"}])
        engine = FakeEngine({"segment-01": {"cards": [card()]}, "segment-01-feedback-1": {"cards": [fixed]}})
        result = collect_cards(engine, ELEMENTS, Path("unused"), title="t", source={"sha256": "x"})
        self.assertEqual([name for name, _ in engine.requests], ["segment-01", "segment-01-feedback-1"])
        self.assertEqual(engine.requests[1][1]["uncited_numeric_elements"], ["p2:e1"])
        self.assertTrue(result["completion"]["ready_for_drafting"])
        self.assertEqual(result["research"]["facts"][0]["id"], "S01C01")
        failing = collect_cards(FakeEngine({"segment-01": RuntimeError("HTTP 500")}), ELEMENTS, Path("unused"),
                                title="t", source={"sha256": "x"})
        self.assertFalse(failing["completion"]["ready_for_drafting"])
        self.assertEqual(len(failing["completion"]["blocking"]), 1)

    def test_question_based_modules_do_not_use_cards(self):
        for name in ("requirement_extraction.py", "requirement_intervention.py", "evidence_requirements.py"):
            path = ROOT / name
            if path.is_file():
                self.assertNotIn("experiment_cards", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
