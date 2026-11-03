import copy
import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from evidence_requirements import (generate_requirements, inputs_for, load_requirements,
                                   materialize_response, requirement_folder, validate_response)
from paper_draft_agent import PaperDraftOrchestrator
from paper_plan import normalize_research, project_profile
from planned_paper import PlannedPaper
from requirement_extraction import (extract_requirement_evidence, load_saved_requirement_evidence,
                                    merge_exact_facts, validate_links)


def reference():
    return {"profiles": [{"observed_outline": [{"title": "Methods", "role": "methods", "children": []}],
        "section_guidelines": [{"role": "methods", "purpose": "Explain method",
            "required_content": ["Explain implementation"], "paragraph_flow": [], "figures_tables": []}],
        "style_rules": [], "not_to_transfer": []}]}


def response():
    return {"requirements": [{"id": "Q1", "section_ids": ["ref1_s1"],
        "guideline_ids": ["ref1_methods_required_1"], "question": "What changed?",
        "expected_fields": ["implementation"], "priority": "required", "applies_when": None,
        "applicability": {"kind": "universal", "condition": "모든 연구의 수행 방법 확인"}}],
        "criterion_assignments": {"ref1_methods_required_1": {"question_ids": ["Q1"], "reason": "Needs source evidence"}},
        "guideline_dispositions": {"ref1_methods_required_1": {
            "disposition": "evidence_question", "reason": "Needs source evidence"}}}


def raw_fact(page=1):
    return {"statement": "메모리를 재사용한다.", "limits": "구현 기록이다.",
        "kind": "method", "claim_status": "documented_method", "source_origin": "research_note",
        "conditions": [], "observations": [], "parameters": [], "measurements": [],
        "element_ids": [f"p{page}:e1"], "requirement_ids": ["Q1"], "supplemental_reason": ""}


class RequirementTests(unittest.TestCase):
    def test_common_bank_is_identical_across_topics_and_projects_without_mutating_golden(self):
        ref = reference()
        original = copy.deepcopy(ref)
        with TemporaryDirectory() as tmp:
            engine = PlannedPaper(tmp, client=lambda *args: response())
            req = generate_requirements(engine, ref, "CUDA scope", project_profile("generic"))
            self.assertTrue((Path(req["run_dir"]) / "requirements.json").is_file())
            self.assertIn("Q1", (Path(req["run_dir"]) / "requirements.md").read_text(encoding="utf-8"))
            self.assertEqual(load_requirements(tmp, ref, "CUDA scope", project_profile("generic")), req)
            self.assertEqual(load_requirements(tmp, ref, "different scope", project_profile("cuda")), req)
            self.assertNotIn("research_scope", req)
            self.assertNotIn("project", req)
            self.assertEqual(inputs_for(ref, "CUDA", project_profile("cuda")), inputs_for(ref, "biology", project_profile("generic")))
            changed = copy.deepcopy(ref)
            changed["profiles"][0]["section_guidelines"][0]["required_content"].append("new criterion")
            self.assertIsNone(load_requirements(tmp, changed))
            self.assertEqual(ref, original)

    def test_roles_are_generated_separately_without_question_id_collisions_or_missing_tail_criteria(self):
        ref = reference()
        ref["profiles"][0]["observed_outline"].append({"title": "Results", "role": "evaluation", "children": []})
        ref["profiles"][0]["section_guidelines"].append({"role": "evaluation", "purpose": "Results",
            "required_content": ["Report results"], "paragraph_flow": [], "figures_tables": []})
        seen = []
        def client(system, packet, schema, path):
            material = packet["reference_material"]
            self.assertNotIn("target_research_scope", packet)
            self.assertNotIn("project", packet)
            self.assertEqual(len(material["reference_guides"]), 1)
            section = material["sections"][0]
            self.assertNotIn("reference_title", section)
            guide = next(iter(material["reference_guides"].values()))
            seen.append(guide["role"])
            return {"requirements": [{"id": "Q1", "section_ids": [section["id"]], "question": "Question?",
                "expected_fields": ["answer"], "priority": "required", "applies_when": None,
                "applicability": {"kind": "universal", "condition": "모든 연구"}}], "criterion_assignments": {
                    guide["criteria"][0]["id"]: {"question_ids": ["Q1"], "reason": "Needs evidence"}}}
        with TemporaryDirectory() as tmp:
            req = generate_requirements(PlannedPaper(tmp, client=client), ref, "scope", project_profile("generic"))
        self.assertEqual(seen, ["methods", "evaluation"])
        self.assertEqual({q["id"] for q in req["requirements"]}, {"methods_Q1", "evaluation_Q1"})
        self.assertEqual(len(req["guideline_dispositions"]), 2)
        self.assertTrue(all(v["disposition"] == "evidence_question" for v in req["guideline_dispositions"].values()))

    def test_literature_is_context_when_project_omits_unsupported_related_work(self):
        ref = json.loads(json.dumps(reference()).replace("methods", "related_work"))
        raw = json.loads(json.dumps(response()).replace("methods", "related_work"))
        with TemporaryDirectory() as tmp:
            req = generate_requirements(PlannedPaper(tmp, client=lambda *args: raw), ref, "scope", project_profile("generic"))
        self.assertEqual(req["requirements"][0]["priority"], "context")

    def test_missing_or_mislinked_guidelines_fail_validation(self):
        inputs = inputs_for(reference(), "scope", project_profile("generic"))
        for mutate in (lambda r: r["guideline_dispositions"].clear(),
                       lambda r: r["requirements"][0].update(guideline_ids=["invented"]),
                       lambda r: r["requirements"][0].update(section_ids=["invented"]),
                       lambda r: r["guideline_dispositions"]["ref1_methods_required_1"].update(disposition="writing_only")):
            value = response()
            mutate(value)
            with self.assertRaises(ValueError):
                validate_response(value, inputs)

    def test_dispositions_are_derived_from_edges_and_invalid_question_links_are_rejected(self):
        inputs = inputs_for(reference(), "scope", project_profile("generic"))
        value = response()
        value.pop("guideline_dispositions")
        result = materialize_response(value, inputs)
        self.assertEqual(result["guideline_dispositions"]["ref1_methods_required_1"]["disposition"], "evidence_question")
        value["criterion_assignments"]["ref1_methods_required_1"]["question_ids"] = ["unknown"]
        with self.assertRaises(ValueError):
            materialize_response(value, inputs)

    def test_every_criterion_must_be_accounted_for_even_when_writing_only(self):
        ref = reference()
        ref["profiles"][0]["section_guidelines"][0]["required_content"].append("Paragraph order")
        inputs = inputs_for(ref, "scope", project_profile("generic"))
        value = response()
        with self.assertRaises(ValueError):
            materialize_response(value, inputs)
        value["criterion_assignments"]["ref1_methods_required_2"] = {"question_ids": [], "reason": "Only paragraph order"}
        result = materialize_response(value, inputs)
        self.assertEqual(result["guideline_dispositions"]["ref1_methods_required_2"]["disposition"], "writing_only")

    def test_explicit_apply_condition_cannot_also_be_classified_universal(self):
        inputs = inputs_for(reference())
        value = response()
        value["requirements"][0]["applies_when"] = "여러 방법을 비교한 경우"
        value["requirements"][0]["applicability"]["kind"] = "universal"
        result = materialize_response(value, inputs)
        self.assertEqual(result["requirements"][0]["applicability"], {
            "kind": "conditional", "condition": "여러 방법을 비교한 경우"})

    def test_requirements_only_never_parses_source_or_writes_draft(self):
        with TemporaryDirectory() as tmp:
            gateway = PaperDraftOrchestrator("test-key")
            gateway.base_dir = Path(tmp)
            package = {"run_dir": tmp}
            with patch("paper_draft_agent.load_paper_structure", return_value=reference()), \
                    patch("evidence_requirements.generate_requirements", return_value=package), \
                    patch.object(PlannedPaper, "load_evidence", side_effect=AssertionError("parsed source")), \
                    patch.object(PlannedPaper, "run", side_effect=AssertionError("drafted")):
                result = gateway.run_pipeline(requirements_only=True, evidence_mode="requirements", research_scope="scope")
            self.assertEqual(result["evidence_requirements"], package)
            self.assertEqual(result["draft_markdown"], "")

    def test_identical_facts_share_questions_and_sources_but_changed_conditions_stay_separate(self):
        first, repeated, changed = raw_fact(), raw_fact(2), raw_fact(3)
        repeated["requirement_ids"] = ["Q2"]
        changed["conditions"] = ["다른 조건"]
        result = merge_exact_facts([first, repeated, changed])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["requirement_ids"], ["Q1", "Q2"])
        self.assertEqual(result[0]["element_ids"], ["p1:e1", "p2:e1"])
        self.assertEqual(first["element_ids"], ["p1:e1"])

    def test_question_led_gateway_reaches_new_extractor_without_comprehensive_collection(self):
        with TemporaryDirectory() as tmp:
            gateway = PaperDraftOrchestrator("test-key")
            gateway.base_dir = Path(tmp)
            req = {"run_dir": str(Path(tmp) / "requirements")}
            package = {"run_dir": str(Path(tmp) / "evidence")}
            with patch("paper_draft_agent.load_paper_structure", return_value=reference()), \
                    patch("evidence_requirements.generate_requirements", return_value=req), \
                    patch("requirement_extraction.extract_requirement_evidence", return_value=package) as collect, \
                    patch("research_extract_v3.extract_evidence", side_effect=AssertionError("whole-source collection")), \
                    patch.object(PlannedPaper, "run", side_effect=AssertionError("drafted")):
                result = gateway.run_pipeline(research_pdf_path="source.pdf", evidence_mode="requirements",
                    research_scope="scope", extract_only=True)
            self.assertEqual(result["evidence_extraction"], package)
            self.assertEqual(collect.call_args.args[2], req)

    def test_unknown_question_and_unexplained_supplement_are_rejected(self):
        elements = {"p1:e1": {"page": 1}}
        for ids in (["unknown"], []):
            value = raw_fact()
            value["requirement_ids"] = ids
            with self.assertRaises(ValueError):
                validate_links(value, {"Q1": {}}, elements, 1)

    def run_collection(self, tmp, sufficient=True, unsupported=False, unreadable=False):
        root = Path(tmp)
        pdf = root / "source.pdf"
        pdf.write_bytes(b"fake source")
        source = {"run_dir": str(root / "draft_history/parse-api-v1/document/research-sources-v3"),
            "page_count": 2, "source": {"file": pdf.name, "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest()},
            "elements": [{"element_id": f"p{p}:e1", "page": p, "category": "paragraph",
                          "origin": "standard_parse", "text": "메모리를 재사용한다."} for p in (1, 2)],
            "pdf_image_inventory": [{"page": 2, "state": "unreadable_image"}] if unreadable else []}
        req = {**inputs_for(reference(), "scope", project_profile("generic")), **response(),
               "input_fingerprint": "test", "run_dir": str(root / "requirements")}
        # Two questions deliberately share one evidence item.
        req["requirements"].append({**req["requirements"][0], "id": "Q2", "question": "How implemented?"})
        calls = []
        def client(system, packet, schema, path):
            calls.append(str(path))
            if "requirement-page-0001" in str(path):
                return {"facts": [raw_fact()], "existing_links": []}
            if "requirement-page-0002" in str(path):
                self.assertEqual(len(packet["preserved_evidence"]), 1)
                return {"facts": [], "existing_links": [{"fact_id": "E00001", "element_ids": ["p2:e1"], "requirement_ids": ["Q2"]}]}
            if "-verify" in str(path):
                return {"checks": {"E00001": {"support": "unsupported" if unsupported else "supported",
                    "claim_status": "documented_method", "source_origin": "research_note", "attribution_preserved": True,
                    "contradictions": [], "missing_support": [], "evidence_limitations": [], "note": "원문과 일치"}}}
            if "requirement-check-" in str(path):
                return {"status": "sufficient" if sufficient else "partial", "evidence_ids": ["E00001"],
                        "missing_fields": [] if sufficient else ["implementation"], "reason": "원문 대조"}
            raise AssertionError("Unexpected call " + str(path))
        engine = PlannedPaper(root, client=client)
        engine.call = lambda folder, name, system, packet, schema, **kwargs: client(system, packet, schema, Path(folder) / name)
        with patch("requirement_extraction.load_complete_sources", return_value=source):
            result = extract_requirement_evidence(pdf, engine, req)
        return result, pdf, req, calls

    def test_extraction_reuses_one_fact_across_questions_and_preserves_guidelines_for_writing(self):
        with TemporaryDirectory() as tmp:
            result, pdf, req, calls = self.run_collection(tmp)
            self.assertEqual(len(result["items"]), 1)
            self.assertEqual(result["items"][0]["requirement_ids"], ["Q1", "Q2"])
            self.assertEqual(len(result["items"][0]["sources"]), 2)
            self.assertEqual(result["writing_fact_ids"], ["E00001"])
            self.assertTrue(result["completion"]["ready_for_drafting"])
            research = normalize_research(result["research"])
            self.assertEqual(research["evidence_requirements"], req)
            self.assertEqual(research["facts"][0]["requirement_ids"], ["Q1", "Q2"])
            self.assertEqual(load_saved_requirement_evidence(pdf, tmp, req), result)
            changed = copy.deepcopy(req)
            changed["requirements"][0]["question"] = "changed"
            self.assertIsNone(load_saved_requirement_evidence(pdf, tmp, changed))
            self.assertFalse(any("audit" in c or "rescue" in c or "extract-block" in c for c in calls))

    def test_partial_requirements_unverified_facts_and_unreadable_sources_each_keep_gate_closed(self):
        for kwargs in ({"sufficient": False}, {"unsupported": True}, {"unreadable": True}):
            with self.subTest(kwargs=kwargs), TemporaryDirectory() as tmp:
                result, _, _, _ = self.run_collection(tmp, **kwargs)
                self.assertFalse(result["completion"]["ready_for_drafting"])
                with patch.object(PlannedPaper, "load_evidence", return_value=result):
                    with self.assertRaises(ValueError):
                        PlannedPaper(tmp).load_research(project_profile("generic"), research_pdf="source.pdf")


if __name__ == "__main__":
    unittest.main()
