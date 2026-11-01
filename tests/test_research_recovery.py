import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from paper_plan import normalize_research
from research_dedup import approved_duplicate_candidates, duplicate_group_fingerprint
from research_extract_v3 import (apply_saved_duplicate_decisions, apply_source_review, confirm_recovered_duplicate, ground,
                                 _recheck_recovery_sources, run_unconfirmed_recovery, writing_eligible)
from research_recovery import _checks_schema, current_duplicate_suggestions, recover_unconfirmed


def sources():
    return {"p1:e1": {"element_id": "p1:e1", "page": 1, "category": "paragraph",
                      "origin": "standard_parse", "text": "HYBRID 실행 시간 12.5ms. 교수님이 답지를 제공해 주기로 했다."}}


def raw():
    return {"statement": "HYBRID 실행 시간은 12.5ms다.", "limits": "해당 실행 기록이다.",
            "kind": "result", "claim_status": "observed", "source_origin": "research_note",
            "conditions": [], "observations": [], "parameters": [], "measurements": [],
            "element_ids": ["p1:e1"]}


def fact(identifier="F1", approved=False):
    value = ground(raw(), sources())
    value.update(id=identifier, support="supported" if approved else "unsupported",
                 support_note="검토 필요", attribution_preserved=approved, writing_role="research_fact")
    return value


def check(**updates):
    value = {"support": "supported", "claim_status": "observed", "source_origin": "research_note",
             "attribution_preserved": True, "contradictions": [], "missing_support": [],
             "evidence_limitations": [], "preservation_issues": [], "duplicate_fully_covered": False,
             "confidence": "high", "note": "연결 원문과 일치하며 원래 핵심 정보가 보존된다."}
    value.update(updates)
    return value


def repair_response(value=None):
    return {"decisions": {"F1": {"action": "repair", "representative_fact_id": "",
                                  "corrections": [value or raw()], "reason": "잘못 붙인 버전명을 원문 모드명으로 고쳤다."}}}


def duplicate_response():
    return {"decisions": {"F1": {"action": "duplicate", "representative_fact_id": "F2",
                                  "corrections": [], "reason": "같은 실행의 동일한 내용이다."}}}


class PendingRecoveryTests(unittest.TestCase):
    def run_recovery(self, engine, facts, elements=None, **kwargs):
        with TemporaryDirectory() as temporary:
            return recover_unconfirmed(engine, Path(temporary), facts, elements or sources(), **kwargs)

    def test_planned_record_is_eligible_with_documentary_limitations(self):
        value = fact()
        value["statement"] = "교수님이 답지를 제공해 주기로 했다."
        apply_source_review(value, check(claim_status="planned",
            evidence_limitations=["실제 제공 완료 여부는 원문에 없다."]), sources())
        value["writing_role"] = "context_only"
        self.assertTrue(writing_eligible(value))
        research = normalize_research({"title": "연구", "facts": [value]})
        self.assertEqual(research["facts"][0]["evidence_limitations"], value["evidence_limitations"])

    def test_limitations_never_override_real_missing_support(self):
        value = fact()
        apply_source_review(value, check(missing_support=["원문에 v4 라벨이 없다."],
            evidence_limitations=["상세 재현 조건 없음"]), sources())
        self.assertFalse(writing_eligible(value))

    def test_verified_repair_preserves_original_and_applies_same_id(self):
        original = fact()
        original["statement"] = "v9 HYBRID 실행 시간은 12.5ms다."
        engine = Mock()
        engine.call.side_effect = [repair_response(), {"checks": {"F1": check()}}]
        output, report = self.run_recovery(engine, [original])
        self.assertEqual(output[0]["id"], "F1")
        self.assertTrue(writing_eligible(output[0]))
        self.assertEqual(original["statement"], report["records"][0]["original"]["statement"])
        self.assertEqual(report["counts"], {"repaired": 1})
        self.assertEqual(engine.call.call_count, 2)

    def test_repair_can_record_an_approved_hint_but_still_requires_independent_review(self):
        response = repair_response()
        response["decisions"]["F1"]["representative_fact_id"] = "F2"
        engine = Mock()
        engine.call.side_effect = [response, {"checks": {"F1": check()}}]
        output, report = self.run_recovery(engine, [fact(), fact("F2", True)])
        self.assertTrue(writing_eligible(output[0]))
        self.assertEqual(report["records"][0]["related_approved_fact_id"], "F2")
        self.assertEqual(engine.call.call_count, 2)

    def test_numeric_failure_gets_one_precise_feedback_attempt_then_independent_review(self):
        wrong = raw()
        wrong["statement"] = "실행 시간은 900ms다."
        engine = Mock()
        engine.call.side_effect = [repair_response(wrong), repair_response(), {"checks": {"F1": check()}}]
        output, report = self.run_recovery(engine, [fact()])
        self.assertTrue(writing_eligible(output[0]))
        self.assertEqual(engine.call.call_count, 3)
        feedback = engine.call.call_args_list[1].args[3]["validation_feedback"]["F1"]
        self.assertIn("900", feedback["errors"][0])
        self.assertEqual(len(report["records"][0]["attempts"]), 2)

    def test_blocked_duplicate_is_repaired_instead_of_approved_by_its_representative(self):
        original = fact()
        original["numeric_issues"] = ["잘못 붙인 숫자"]
        engine = Mock()
        engine.call.side_effect = [duplicate_response(), repair_response(), {"checks": {"F1": check()}}]
        output, report = self.run_recovery(engine, [original, fact("F2", True)])
        self.assertTrue(writing_eligible(output[0]))
        self.assertEqual(report["counts"], {"repaired": 1})
        self.assertEqual(engine.call.call_count, 3)

    def test_blank_json_batch_splits_and_recovers_without_final_transport_error(self):
        engine = Mock()
        def respond(folder, name, system, packet, schema, **kwargs):
            if name.endswith('-verify'):
                return {"checks": {key: check() for key in packet["proposals"]}}
            if len(packet["targets"]) > 1:
                raise RuntimeError("모델이 완결된 JSON을 반환하지 못했습니다.")
            key = next(iter(packet["targets"]))
            return {"decisions": {key: repair_response()["decisions"]["F1"]}}
        engine.call.side_effect = respond
        output, report = self.run_recovery(engine, [fact(), fact("F2")])
        self.assertTrue(all(writing_eligible(value) for value in output))
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["request_events"][0]["action"], "split")
        self.assertEqual(engine.call.call_count, 4)

    def test_blank_singletons_remain_pending_after_bounded_split(self):
        engine = Mock()
        engine.call.side_effect = RuntimeError("모델이 완결된 JSON을 반환하지 못했습니다.")
        originals = [fact(), fact("F2")]
        output, report = self.run_recovery(engine, originals)
        self.assertEqual(output, originals)
        self.assertEqual(engine.call.call_count, 3)
        self.assertEqual(len(report["errors"]), 2)
        self.assertTrue(all(record.get("reason") for record in report["records"]))

    def test_review_schema_requires_canonical_status_and_origin_values(self):
        fields = _checks_schema(["F1"])["properties"]["checks"]["properties"]["F1"]["properties"]
        self.assertIn("observed", fields["claim_status"]["enum"])
        self.assertNotIn("observed (preserved)", fields["claim_status"]["enum"])
        self.assertIn("research_note", fields["source_origin"]["enum"])

    def test_repair_cannot_discard_accepted_measurement(self):
        original = fact()
        original["measurements"] = [{"metric": "time", "value": "12.5", "unit": "ms", "condition": "",
                                     "value_kind": "scalar", "element_ids": ["p1:e1"]}]
        engine = Mock()
        engine.call.return_value = repair_response()
        output, report = self.run_recovery(engine, [original])
        self.assertEqual(output, [original])
        self.assertEqual(engine.call.call_count, 2)
        self.assertIn("승인 측정값", report["records"][0]["reason"])

    def test_independent_reviewer_blocks_loss_of_caveat_or_rejected_measurement(self):
        original = fact()
        original["rejected_measurements"] = [{"metric": "time", "value": "", "unit": "ms",
            "condition": "HYBRID", "value_kind": "scalar", "element_ids": ["p1:e1"], "validation_error": "빈 값"}]
        engine = Mock()
        failed_check = {"checks": {"F1": check(preservation_issues=["측정 지표를 삭제함"])}}
        engine.call.side_effect = [repair_response(), failed_check, repair_response(), failed_check]
        output, report = self.run_recovery(engine, [original])
        self.assertEqual(output, [original])
        self.assertEqual(report["counts"], {"unresolved": 1})
        self.assertEqual(engine.call.call_count, 4)
        self.assertEqual(len(report["records"][0]["attempts"]), 2)

    def test_preservation_feedback_can_recover_once_and_requires_new_independent_check(self):
        engine = Mock()
        engine.call.side_effect = [repair_response(), {"checks": {"F1": check(preservation_issues=["비교 주장 누락"]) }},
                                   repair_response(), {"checks": {"F1": check()}}]
        output, report = self.run_recovery(engine, [fact()])
        self.assertTrue(writing_eligible(output[0]))
        self.assertEqual(report["counts"], {"repaired": 1})
        self.assertEqual(engine.call.call_count, 4)
        feedback = engine.call.call_args_list[2].args[3]["validation_feedback"]["F1"]
        self.assertIn("비교 주장 누락", feedback["errors"])

    def test_code_and_content_feedback_share_one_correction_budget_per_fact(self):
        first = repair_response()
        wrong = raw()
        wrong['statement'] = '실행 시간은 900ms다.'
        first['decisions']['F2'] = repair_response(wrong)['decisions']['F1']
        second = {'decisions': {'F2': repair_response()['decisions']['F1']}}
        failed = check(preservation_issues=['원래 비교 주장 누락'])
        engine = Mock()
        engine.call.side_effect = [first, second, {'checks': {'F1': failed, 'F2': failed}},
                                  repair_response(), {'checks': {'F1': check()}}]
        output, report = self.run_recovery(engine, [fact(), fact('F2')])
        self.assertTrue(writing_eligible(output[0]))
        self.assertFalse(writing_eligible(output[1]))
        self.assertEqual([len(row['attempts']) for row in report['records']], [2, 2])
        self.assertEqual(set(engine.call.call_args_list[3].args[3]['targets']), {'F1'})

    def test_content_feedback_failure_preserves_successful_sibling(self):
        first = repair_response()
        first['decisions']['F2'] = repair_response()['decisions']['F1']
        engine = Mock()
        engine.call.side_effect = [first, {'checks': {'F1': check(), 'F2': check(preservation_issues=['측정 삭제'])}},
                                  RuntimeError('HTTP 503')]
        output, report = self.run_recovery(engine, [fact(), fact('F2')])
        self.assertTrue(writing_eligible(output[0]))
        self.assertFalse(writing_eligible(output[1]))
        self.assertTrue(report['errors'])

    def test_numeric_hallucination_is_blocked_before_independent_call(self):
        wrong = raw()
        wrong["statement"] = "실행 시간은 900ms다."
        original = fact()
        engine = Mock()
        engine.call.return_value = repair_response(wrong)
        output, _ = self.run_recovery(engine, [original])
        self.assertEqual(output, [original])
        self.assertEqual(engine.call.call_count, 2)

    def test_explicit_caption_conflict_blocks_even_supported_verifier(self):
        elements = sources()
        elements["p1:e1"].update(category="figure", text="forward backward 12.5ms")
        elements["p1:e2"] = {**elements["p1:e1"], "element_id": "p1:e2", "category": "paragraph",
                             "text": "실행 인자: backward only"}
        proposal = raw()
        proposal["statement"] = "forward 실행 시간은 12.5ms다."
        original = fact()
        engine = Mock()
        engine.call.side_effect = [repair_response(proposal), {"checks": {"F1": check()}}]
        output, _ = self.run_recovery(engine, [original], elements)
        self.assertEqual(output, [original])

    def test_full_duplicate_is_only_a_human_suggestion_not_an_approval(self):
        original, representative = fact(), fact("F2", True)
        engine = Mock()
        engine.call.side_effect = [duplicate_response(), {"checks": {"F1": check(duplicate_fully_covered=True)}}]
        output, report = self.run_recovery(engine, [original, representative])
        self.assertEqual(output, [original, representative])
        self.assertEqual(report["counts"], {"duplicate_confirmation": 1})
        extraction = {"items": output, "writing_fact_ids": ["F2"], "evidence_recovery": report}
        self.assertIn("F1", current_duplicate_suggestions(extraction))
        with patch("research_extract_v3.set_human_evidence_exclusion") as save:
            confirm_recovered_duplicate(extraction, "F1")
            self.assertEqual(save.call_args.args[2:5], ("F1", "F1", "duplicate"))
        extraction["items"][1]["statement"] += " 改"
        self.assertEqual(current_duplicate_suggestions(extraction), {})
        with self.assertRaises(ValueError):
            confirm_recovered_duplicate(extraction, "F1")

    def test_partial_duplicate_or_low_confidence_does_not_hide_pending(self):
        for options in ({"duplicate_fully_covered": False}, {"duplicate_fully_covered": True, "confidence": "low"}):
            with self.subTest(options=options):
                original = fact()
                engine = Mock()
                engine.call.side_effect = [duplicate_response(), {"checks": {"F1": check(**options)}}]
                output, report = self.run_recovery(engine, [original, fact("F2", True)])
                self.assertEqual(output[0], original)
                self.assertEqual(report["counts"], {"unresolved": 1})

    def test_excluded_representative_cannot_be_reused(self):
        engine = Mock()
        engine.call.return_value = duplicate_response()
        _, report = self.run_recovery(engine, [fact(), fact("F2", True)], excluded_ids={"F2"})
        self.assertEqual(engine.call.call_count, 2)
        self.assertEqual(report["counts"], {"unresolved": 1})

    def test_duplicate_cannot_bypass_unresolved_numeric_validation(self):
        value = fact()
        value["numeric_issues"] = ["원문 미확인 숫자"]
        engine = Mock()
        engine.call.return_value = duplicate_response()
        output, report = self.run_recovery(engine, [value, fact("F2", True)])
        self.assertEqual(output[0], value)
        self.assertEqual(report["counts"], {"unresolved": 1})
        self.assertEqual(engine.call.call_count, 2)

    def test_independent_call_failure_keeps_original_and_records_error(self):
        original = fact()
        engine = Mock()
        engine.call.side_effect = [repair_response(), RuntimeError("HTTP 503")]
        output, report = self.run_recovery(engine, [original])
        self.assertEqual(output, [original])
        self.assertTrue(report["errors"])

    def test_human_reviewed_or_excluded_fact_is_protected(self):
        engine = Mock()
        originals = [fact(), fact("F2")]
        output, report = self.run_recovery(engine, originals, protected_ids={"F1"}, excluded_ids={"F2"})
        self.assertEqual(output, originals)
        self.assertEqual(report["records"], [])
        engine.call.assert_not_called()

    def test_missing_response_id_never_applies_a_repair(self):
        original = fact()
        engine = Mock()
        engine.call.return_value = {"decisions": {}}
        output, report = self.run_recovery(engine, [original])
        self.assertEqual(output, [original])
        self.assertTrue(report["errors"])

    def test_modified_representative_invalidates_a_simultaneous_duplicate_suggestion(self):
        representative = fact("F2", True)
        representative["rejected_measurements"] = [{"metric": "time", "value": "", "unit": "ms",
            "condition": "", "value_kind": "scalar", "element_ids": ["p1:e1"]}]
        decisions = duplicate_response()
        decisions["decisions"]["F2"] = {"action": "repair", "representative_fact_id": "",
            "corrections": [raw()], "reason": "원문 검토 후 측정을 복구했다."}
        engine = Mock()
        engine.call.side_effect = [decisions, {"checks": {"F1": check(duplicate_fully_covered=True), "F2": check()}}]
        _, report = self.run_recovery(engine, [fact(), representative])
        self.assertEqual(report["counts"], {"unresolved": 1, "repaired": 1})

    def test_saved_duplicate_decision_is_invalid_when_current_fact_changes(self):
        with TemporaryDirectory() as temporary:
            values = [fact(approved=True), fact("F2", True)]
            fingerprint = duplicate_group_fingerprint(values)
            group = {"group_id": "G1", "group_fingerprint": fingerprint, "fact_ids": ["F1", "F2"]}
            record = {**group, "decision": "keep_one", "kept_fact_ids": ["F2"],
                      "excluded_fact_ids": ["F1"], "source_sha256": "same"}
            (Path(temporary) / "human-duplicate-decisions.json").write_text(json.dumps({
                "source_sha256": "same", "events": [record]}), encoding="utf-8")
            package = {"title": "연구", "items": values, "source": {"sha256": "same"},
                       "run_dir": temporary, "semantic_duplicate_review": {"groups": [group]}}
            self.assertTrue(apply_saved_duplicate_decisions(package))
            self.assertEqual(package["writing_fact_ids"], ["F2"])
            package["items"][0]["limits"] += " 추가 한계"
            self.assertFalse(apply_saved_duplicate_decisions(package))
            self.assertEqual(package["writing_fact_ids"], ["F1", "F2"])

    def test_approved_fact_with_rejected_measurement_can_find_a_duplicate(self):
        value = fact(approved=True)
        value["rejected_measurements"] = [{"value": ""}]
        self.assertEqual(approved_duplicate_candidates([value, fact("F2", True)], {"F1", "F2"})["F1"], ["F2"])

    def test_saved_recovery_rechecks_ledger_without_rescuing_more_facts(self):
        for remaining_blocker in (False, True):
            with self.subTest(remaining_blocker=remaining_blocker), TemporaryDirectory() as temporary:
                folder = Path(temporary)
                archive = folder / "source-archive.json"
                archive.write_text(json.dumps({"source": {"sha256": "same"}, "elements": list(sources().values())}), encoding="utf-8")
                package = {"title": "연구", "source": {"sha256": "same"}, "run_dir": str(folder),
                    "source_archive_path": str(archive), "items": [fact()], "writing_fact_ids": [], "research": None,
                    "completion": {"pages_reviewed": 1, "source_pages": 1, "source_elements": 1,
                        "elements_reviewed": 1, "unresolved_source_items": ["p1:e1: 누락"] + (["이미지 판독 불가"] if remaining_blocker else []),
                        "ready_for_drafting": False},
                    "element_ledger": {"p1:e1": {"state": "unresolved", "fact_ids": [], "note": "누락"}}}
                engine = Mock()
                engine.call.side_effect = [repair_response(), {"checks": {"F1": check()}}]
                with patch("research_extract_v3._reconcile_page", return_value={"ledger": {
                    "p1:e1": {"state": "represented", "fact_ids": ["F1"], "note": "원문 대조 완료"}}}) as audit:
                    report = run_unconfirmed_recovery(package, engine)
                self.assertEqual(package["writing_fact_ids"], ["F1"])
                self.assertEqual(package["completion"]["ready_for_drafting"], not remaining_blocker)
                self.assertEqual(audit.call_args.kwargs["max_rounds"], 0)
                self.assertEqual(report["ledger_rechecked_pages"], [1])
                saved = json.loads((folder / "evidence.json").read_text(encoding="utf-8"))
                self.assertEqual(saved["evidence_recovery"]["records"][0]["original"]["id"], "F1")

    def test_source_hash_mismatch_stops_saved_recovery_before_any_model_call(self):
        with TemporaryDirectory() as temporary:
            archive = Path(temporary) / "source-archive.json"
            archive.write_text(json.dumps({"source": {"sha256": "other"}, "elements": []}), encoding="utf-8")
            engine = Mock()
            with self.assertRaises(ValueError):
                run_unconfirmed_recovery({"run_dir": temporary, "source_archive_path": str(archive),
                                          "source": {"sha256": "same"}}, engine)
            engine.call.assert_not_called()

    def test_failed_source_reaudit_closes_gate_even_if_old_ledger_was_complete(self):
        with TemporaryDirectory() as temporary:
            package = {"items": [fact(approved=True)], "writing_fact_ids": ["F1"],
                "element_ledger": {"p1:e1": {"state": "represented", "fact_ids": ["F1"], "note": "완료"}},
                "completion": {"pages_reviewed": 1, "source_pages": 1, "ready_for_drafting": True,
                               "unresolved_source_items": []}}
            with patch("research_extract_v3._reconcile_page", side_effect=RuntimeError("HTTP 503")):
                report = _recheck_recovery_sources(package, Mock(), sources(), {1}, Path(temporary))
            self.assertFalse(package["completion"]["ready_for_drafting"])
            self.assertEqual(package["element_ledger"]["p1:e1"]["state"], "unresolved")
            self.assertTrue(report["ledger_recheck_errors"])


if __name__ == "__main__":
    unittest.main()
