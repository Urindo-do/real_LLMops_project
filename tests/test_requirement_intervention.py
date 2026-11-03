import copy
import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from document_parse import write_json
from paper_context import build_contexts
from paper_plan import normalize_research
from planned_paper import PlannedPaper
from requirement_extraction import assess_requirements, finish_assessment, extract_requirement_evidence
from requirement_intervention import (RequirementInterruption, apply_decisions, pause, read_json,
    previous_assessment_interruption, reference_decisions, restore_pending, submit_instruction, supplemental_facts)


def question():
    return {"id": "Q1", "question": "정확도를 어떻게 측정했는가?", "expected_fields": ["정확도 지표"],
            "priority": "required", "guideline_ids": ["g_required_1"],
            "applicability": {"kind": "universal", "condition": "모든 연구"}}


def fact():
    return {"id": "E00001", "statement": "실행 시간은 12 ms였다.", "limits": "해당 조건의 기록이다.",
        "kind": "result", "claim_status": "observed", "source_origin": "measurement_log", "support": "supported",
        "numeric_issues": [], "attribution_preserved": True, "writing_role": "research_fact",
        "sources": [{"element_id": "p1:e1", "page": 1, "origin": "standard_parse", "quote": "실행 시간은 12 ms였다."}],
        "requirement_ids": ["Q1"], "measurements": [], "conditions": [], "observations": [], "parameters": []}


def row():
    return {"status": "missing", "evidence_ids": ["E00001"], "missing_fields": ["정확도 지표"], "reason": "시간 기록만 있다."}


def decision(action="exclude"):
    return {"id": "d1234567890123456789", "question_id": "Q1", "question": question(), "action": action,
        "instruction": "정확도를 측정하지 않았으니 해당 질문은 빼줘.", "provided_text": "", "pages": [],
        "writing_instruction": "정확도 평가와 정확도 유지 주장을 제외한다.", "explanation": "해당 질문 제외"}


class InterventionTests(unittest.TestCase):
    def test_pre_checkpoint_failure_is_presented_without_recollecting_or_overriding_the_result(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            response = {"request_fingerprint": "original-request", "value": row()}
            write_json(root / "responses/requirement-check-Q1-original.json", response)
            with self.assertRaises(RequirementInterruption) as caught:
                previous_assessment_interruption(root, {"requirements": [question()]})
            self.assertEqual(caught.exception.pending["assessment"], row())
            write_json(root / "human-decisions.json", [decision()])
            previous_assessment_interruption(root, {"requirements": [question()]})
            self.assertEqual(read_json(root / "responses/requirement-check-Q1-original.json"), response)

    def test_real_contradiction_pauses_and_preserves_original_assessment(self):
        original = row()
        with TemporaryDirectory() as tmp:
            engine = Mock()
            engine.call.return_value = original
            with self.assertRaises(RequirementInterruption) as caught:
                assess_requirements(engine, tmp, {"requirements": [question()]}, [fact()], {"Q1": {"status": "applicable"}})
            pending = read_json(Path(tmp) / "pending-question.json")
            self.assertEqual(pending["assessment"], original)
            self.assertEqual(pending["status"], "waiting_for_user")
            self.assertEqual(caught.exception.pending["id"], pending["id"])
            self.assertEqual(original, row())

    def test_free_text_exclusion_is_saved_once_and_cannot_answer_stale_question(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(RequirementInterruption) as caught:
                pause(tmp, question(), row(), [fact()], "어떻게 처리할까요?")
            engine = Mock()
            engine.call.return_value = {k: v for k, v in decision().items() if k in
                ("action", "provided_text", "pages", "writing_instruction", "explanation")}
            record = submit_instruction(engine, tmp, caught.exception.pending["id"], decision()["instruction"])
            self.assertEqual(record["question_id"], "Q1")
            self.assertEqual(read_json(Path(tmp) / "human-decisions.json")[0]["instruction"], decision()["instruction"])
            with self.assertRaises(ValueError):
                submit_instruction(engine, tmp, caught.exception.pending["id"], "다른 지시")
            self.assertEqual(engine.call.call_count, 1)

    def test_clarification_or_invented_addendum_does_not_resolve_pending_question(self):
        for response in ({**decision(), "action": "clarify", "explanation": "어떤 내용을 제외할까요?"},
                         {**decision(), "action": "provide_information", "provided_text": "정확도는 99%"}):
            with self.subTest(response=response), TemporaryDirectory() as tmp:
                with self.assertRaises(RequirementInterruption) as caught:
                    pause(tmp, question(), row(), [fact()], "대기")
                engine = Mock()
                engine.call.return_value = response
                with self.assertRaises(ValueError):
                    submit_instruction(engine, tmp, caught.exception.pending["id"], "계속해")
                self.assertIsNone(read_json(Path(tmp) / "human-decisions.json"))
                self.assertEqual(read_json(Path(tmp) / "pending-question.json")["status"], "waiting_for_user")

    def source(self, root):
        pdf = root / "journal.pdf"
        pdf.write_bytes(b"fixture")
        source = {"source": {"file": pdf.name, "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest()},
                  "page_count": 1, "run_dir": str(root / "source"), "pdf_image_inventory": [],
                  "elements": [{"element_id": "p1:e1", "page": 1, "category": "paragraph",
                                "origin": "standard_parse", "text": fact()["statement"]}]}
        return pdf, source

    def test_exclusion_reaches_research_and_guides_without_opening_unrelated_source_gate(self):
        for pending in ([], [1]):
            with self.subTest(pending=pending), TemporaryDirectory() as tmp:
                root = Path(tmp)
                pdf, source = self.source(root)
                write_json(root / "human-decisions.json", [decision()])
                engine = Mock()
                engine.call.side_effect = AssertionError("Excluded question must not call model")
                req = {"scope": "topic_independent", "requirements": [question()]}
                result = finish_assessment(pdf, engine, root, source, req, [fact()], {"Q1": {"status": "applicable"}}, [1], pending)
                self.assertEqual(result["completion"]["ready_for_drafting"], not bool(pending))
                self.assertEqual(result["requirement_coverage"]["Q1"]["status"], "excluded_by_user")
                normalized = normalize_research(result["research"])
                self.assertEqual(normalized["human_guidance"], [decision()])
                catalog = {"g": {"criteria": [{"id": "g_required_1", "description": "정확도 평가"},
                    {"id": "g_required_2", "description": "시간 평가"}], "required_content": ["정확도 평가", "시간 평가"]}}
                filtered = reference_decisions(catalog, normalized)
                self.assertEqual(filtered["g"]["required_content"], ["시간 평가"])
                self.assertEqual(len(catalog["g"]["criteria"]), 2)

    def test_shared_criterion_is_retained_for_an_active_question(self):
        research = {"evidence_requirements": {"requirements": [question(), {**question(), "id": "Q2"}]},
                    "requirement_coverage": {"Q1": {"status": "excluded_by_user"}, "Q2": {"status": "sufficient"}}}
        catalog = {"g": {"criteria": [{"id": "g_required_1", "description": "shared"}]}}
        self.assertEqual(reference_decisions(catalog, research)["g"]["criteria"], catalog["g"]["criteria"])

    def test_user_addendum_is_source_linked_verified_cached_and_not_labelled_as_pdf(self):
        calls = []
        def client(system, packet, schema, path):
            calls.append(str(path))
            if "search-" in str(path):
                source_id = packet["elements"][0]["element_id"]
                return {"title": "추가 기록", "coverage": {}, "facts": [{"statement": "정확도는 90%였다.",
                    "limits": "사용자 보고", "kind": "result", "claim_status": "observed", "source_origin": "research_note",
                    "conditions": [], "observations": [], "parameters": [], "measurements": [], "element_ids": [source_id]}]}
            ids = schema["properties"]["checks"]["properties"]
            return {"checks": {key: {"support": "supported", "claim_status": "observed", "source_origin": "research_note",
                "attribution_preserved": True, "contradictions": [], "missing_support": [], "evidence_limitations": [],
                "note": "제공한 추가 기록과 일치"} for key in ids}}
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, source = self.source(root)
            d = {**decision("provide_information"), "provided_text": "정확도는 90%였다."}
            engine = PlannedPaper(root)
            with patch("planned_paper.request_json", side_effect=lambda system, packet, schema, path, *a, **kw: client(system, packet, schema, path)):
                first = supplemental_facts(engine, root, d, source)
                self.assertEqual(first, supplemental_facts(engine, root, d, source))
            self.assertEqual(len(calls), 2)
            self.assertEqual(first[0]["sources"][0]["origin"], "user_addendum")
            self.assertIn("PDF 원문", first[0]["limits"])
            self.assertTrue(normalize_research({"title": "연구", "facts": first})["facts"][0]["user_provided"])

    def test_research_uses_archived_target_pages_and_rejects_invalid_pages(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, source = self.source(root)
            engine = Mock()
            engine.call.return_value = {"facts": []}
            d = {**decision("recheck"), "pages": [1]}
            self.assertEqual(supplemental_facts(engine, root, d, source), [])
            self.assertEqual(engine.call.call_args.args[3]["elements"][0]["element_id"], "p1:e1")
            with self.assertRaises(ValueError):
                supplemental_facts(engine, root, {**d, "id": "different", "pages": [2]}, source)

    def test_resume_checkpoint_skips_collection_and_does_not_accept_changed_checkpoint(self):
        from evidence_requirements import fingerprint
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf, source = self.source(root)
            source["run_dir"] = str(root / "doc" / "sources")
            req = {"scope": "topic_independent", "requirements": [question()]}
            engine = Mock(base_dir=root)
            engine.call.return_value = row()
            with patch("requirement_extraction.load_complete_sources", return_value=source), \
                 patch("requirement_extraction.assess_applicability", return_value={"Q1": {"status": "applicable"}}), \
                 patch("requirement_extraction.collect_page", return_value={"facts": [], "existing_links": []}), \
                 patch("requirement_extraction.assess_requirements", side_effect=lambda e, folder, *a: pause(folder, question(), row(), [fact()], "대기")):
                with self.assertRaises(RequirementInterruption) as caught:
                    extract_requirement_evidence(pdf, engine, req)
            folder = Path(caught.exception.pending["run_dir"])
            checkpoint = read_json(folder / "assessment-input.json")
            checkpoint["value"]["items"] = [fact()]
            checkpoint["sha256"] = fingerprint(checkpoint["value"])
            write_json(folder / "assessment-input.json", checkpoint)
            write_json(folder / "human-decisions.json", [decision()])
            with patch("requirement_extraction.load_complete_sources", return_value=source), \
                 patch("requirement_extraction.collect_page", side_effect=AssertionError("recollected")), \
                 patch("requirement_extraction.assess_applicability", side_effect=AssertionError("reassessed")):
                result = extract_requirement_evidence(pdf, engine, req)
            self.assertTrue(result["completion"]["ready_for_drafting"])
            checkpoint["value"]["items"][0]["statement"] = "tampered"
            write_json(folder / "assessment-input.json", checkpoint)
            with patch("requirement_extraction.load_complete_sources", return_value=source):
                with self.assertRaisesRegex(ValueError, "변경"):
                    extract_requirement_evidence(pdf, engine, req)

    def test_refresh_restores_only_the_matching_pdf_and_question_bank(self):
        from evidence_requirements import fingerprint
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf, source = self.source(root)
            req = {"requirements": [question()]}
            folder = root / "draft_history/parse-api-v1/doc/research-evidence-requirements-v2/run"
            body = {"source": source, "requirements": req}
            write_json(folder / "assessment-input.json", {"value": body, "sha256": fingerprint(body)})
            with self.assertRaises(RequirementInterruption):
                pause(folder, question(), row(), [fact()], "대기")
            write_json(folder / "interrupted-request.json", {"plan_only": True, "research_pdf_path": "old/path.pdf"})
            pending, request = restore_pending(root, pdf, req)
            self.assertEqual(pending["question"]["id"], "Q1")
            self.assertEqual(request["research_pdf_path"], str(pdf))
            self.assertIsNone(restore_pending(root, pdf, {"requirements": []}))
            pdf.write_bytes(b"other document")
            self.assertIsNone(restore_pending(root, pdf, req))

    def test_streamlit_pause_free_text_and_resume_original_plan_request(self):
        from streamlit.testing.v1 import AppTest
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(RequirementInterruption) as caught:
                pause(root, question(), row(), [fact()], "이 질문을 어떻게 처리할까요?")
            result = {"draft_markdown": "", "plan": {"title": "계획 완료", "rationale": "사용자 지시 반영",
                                                     "sections": [], "unresolved": []}}
            engine = Mock()
            engine.call.return_value = {k: v for k, v in decision().items() if k in
                ("action", "provided_text", "pages", "writing_instruction", "explanation")}
            def submit(_engine, folder, pending_id, text):
                return submit_instruction(engine, folder, pending_id, text)
            with patch("paper_draft_agent.PaperDraftOrchestrator.run_pipeline", side_effect=[caught.exception, result]) as pipeline, \
                 patch("planned_paper.load_latest", return_value=None), \
                 patch("requirement_intervention.restore_pending", return_value=None), \
                 patch("requirement_intervention_view.submit_instruction", side_effect=submit), \
                 patch("requests.sessions.Session.request", side_effect=AssertionError("Unexpected network")):
                app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py"), default_timeout=15).run()
                next(b for b in app.button if b.label == "작성 계획만 만들기").click().run()
                self.assertFalse(list(app.exception))
                self.assertIn("pending_requirement", app.session_state)
                self.assertFalse(list(app.error))
                next(t for t in app.text_area if t.label == "이 질문을 어떻게 처리할까요?").input(decision()["instruction"])
                next(b for b in app.button if b.label == "지시를 반영하고 계속").click().run()
                self.assertFalse(list(app.exception))
                self.assertFalse(list(app.error))
                self.assertEqual(pipeline.call_count, 2)
                self.assertTrue(pipeline.call_args.kwargs["plan_only"])
                self.assertEqual(app.session_state["paper_plan"]["title"], "계획 완료")
                self.assertNotIn("pending_requirement", app.session_state)
                self.assertEqual(read_json(root / "human-decisions.json")[0]["instruction"], decision()["instruction"])


if __name__ == "__main__":
    unittest.main()
