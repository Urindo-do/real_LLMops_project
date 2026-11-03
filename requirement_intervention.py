"""Durable, question-scoped human instructions; original model responses stay intact."""

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from document_parse import filesystem_path, write_json
from evidence_requirements import fingerprint
from paper_plan import object_schema


class RequirementInterruption(ValueError):
    def __init__(self, pending):
        self.pending = pending
        super().__init__("근거 질문에 대한 사용자 지시를 기다리고 있습니다: " + pending["question"]["question"])


def read_json(path, default=None):
    path = filesystem_path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def decisions(folder):
    return read_json(Path(folder) / "human-decisions.json", [])


def restore_pending(base_dir, pdf, requirements):
    """Recover a waiting form after refresh/restart, only for the selected source and question bank."""
    digest = hashlib.sha256(Path(pdf).read_bytes()).hexdigest()
    root = filesystem_path(Path(base_dir) / "draft_history/parse-api-v1")
    candidates = sorted(root.glob("*/research-evidence-requirements-v2/*/pending-question.json"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        pending = read_json(path)
        if pending.get("status") not in ("waiting_for_user", "instruction_saved"):
            continue
        checkpoint = read_json(path.parent / "assessment-input.json", {})
        body = checkpoint.get("value", {})
        if not checkpoint:
            body = {"source": read_json(path.parent / "source-archive.json", {}),
                    "requirements": read_json(path.parent / "requirements.json")}
        request = read_json(path.parent / "interrupted-request.json")
        if body.get("source", {}).get("source", {}).get("sha256") != digest or body.get("requirements") != requirements or not request:
            continue
        if checkpoint and checkpoint.get("sha256") != fingerprint(body):
            raise ValueError("사용자 판단 대기 파일의 입력이 변경됐습니다.")
        # Current folder can differ after a portable restore.
        pending["run_dir"] = str(path.parent)
        request["research_pdf_path"] = str(pdf)
        return pending, request
    return None


def previous_assessment_interruption(folder, requirements):
    """Surface a previous contradictory response before replaying pre-checkpoint extraction."""
    handled = {d["question_id"] for d in decisions(folder)}
    for q in requirements["requirements"]:
        if q["id"] in handled:
            continue
        paths = sorted((Path(folder) / "responses").glob("requirement-check-" + q["id"] + "-*.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        for path in paths:
            if path.name.endswith(".native.json"):
                continue
            cached = read_json(path, {})
            row = cached.get("value", {})
            if cached.get("request_fingerprint") and row.get("status") == "missing" and row.get("evidence_ids"):
                pause(folder, q, row, [], "이전 실행이 이 질문의 모순된 근거 판정에서 중단됐습니다. 어떻게 처리할까요? 원응답은 보존되어 있습니다.")


def pause(folder, question, assessment, facts, reason):
    pending = {"run_dir": str(folder), "question": question, "assessment": assessment,
               "facts": facts, "reason": reason, "status": "waiting_for_user"}
    pending["id"] = fingerprint({**pending, "decisions": decisions(folder)})[:20]
    write_json(Path(folder) / "pending-question.json", pending)
    raise RequirementInterruption(pending)


INTERPRET = """사용자가 현재 근거 질문을 어떻게 처리하라고 지시했는지 해석한다.
exclude: 해당 질문의 내용을 작성 계획·초안에서 제외하라는 명시적 지시.
provide_information: 사용자가 질문에 답할 실제 추가 연구 기록을 제공한 경우.
recheck: 기존 연구 원문을 다시 찾아보거나 판정을 재검토하라는 지시.
clarify: 의미가 불명확하거나, 근거 없이 충족 처리·검사 우회만 요구한 경우.
자동으로 제외하지 않는다. 한 질문의 지시를 다른 질문에 확대하지 않는다.
provided_text에는 사용자 메시지에 실제 있는 추가 기록 부분을 그대로 복사한다. 새 사실을 작성하지 않는다.
추가 기록의 미수행·추정·계획·AI 조언 성격을 유지한다. 페이지를 명시했다면 pages에 실제 번호만 넣는다.
writing_instruction에는 계획·초안에 적용할 범위와 제한을 적는다. clear하지 않으면 clarify와 확인할 질문을 explanation에 쓴다."""


def submit_instruction(engine, folder, pending_id, text):
    folder = filesystem_path(folder)
    pending = read_json(folder / "pending-question.json")
    if not pending or pending["id"] != pending_id or pending["status"] != "waiting_for_user":
        raise ValueError("대기 질문이 변경됐습니다. 최신 질문을 확인해주세요.")
    if not text.strip():
        raise ValueError("어떻게 처리할지 지시를 입력해주세요.")
    schema = object_schema({"action": {"type": "string", "enum": ["exclude", "provide_information", "recheck", "clarify"]},
        "provided_text": {"type": "string"}, "pages": {"type": "array", "items": {"type": "integer", "minimum": 1}},
        "writing_instruction": {"type": "string"}, "explanation": {"type": "string"}})
    value = engine.call(folder / "human-input", "interpret-instruction", INTERPRET,
        {"question": pending["question"], "assessment": pending["assessment"], "user_instruction": text}, schema)
    action = value["action"]
    if action not in {"exclude", "provide_information", "recheck", "clarify"}:
        raise ValueError("사용자 지시의 처리 방식이 잘못됐습니다.")
    if action == "clarify":
        raise ValueError(value["explanation"])
    supplied = value["provided_text"]
    if action == "provide_information" and (not supplied.strip() or supplied not in text):
        raise ValueError("추가 정보는 입력한 메시지의 원문으로 확인되어야 합니다.")
    if any(type(p) is not int or p < 1 for p in value["pages"]):
        raise ValueError("재탐색 페이지 번호가 잘못됐습니다.")
    history = decisions(folder)
    record = {**value, "question_id": pending["question"]["id"], "question": pending["question"],
              "instruction": text, "pending_id": pending_id, "created_at": datetime.now(timezone.utc).isoformat()}
    record["id"] = fingerprint(record)[:20]
    history.append(record)
    write_json(folder / "human-decisions.json", history)
    write_json(folder / "pending-question.json", {**pending, "status": "instruction_saved", "decision_id": record["id"]})
    return record


def supplemental_facts(engine, folder, decision, source):
    """Re-search archived source or extract from a separately attributed user addendum."""
    from requirement_applicability import source_batches
    from research_extract_v2 import extraction_schema, source_packet
    from research_extract_v3 import ground, verify_page

    target = filesystem_path(Path(folder) / "human-input" / decision["id"])
    saved = read_json(target / "facts.json")
    if saved is not None:
        return saved
    if decision["action"] == "provide_information":
        element_id = "user:" + decision["id"]
        elements = {element_id: {"element_id": element_id, "page": 0, "category": "paragraph",
                    "origin": "user_addendum", "text": decision["provided_text"]}}
    else:
        pages = decision["pages"]
        if any(p > source["page_count"] for p in pages):
            raise ValueError("재탐색 페이지가 원문 범위를 벗어났습니다.")
        elements = {e["element_id"]: e for e in source["elements"] if not pages or e["page"] in pages}
    prompt = """현재 질문에 직접 답하는 근거를 원문에서 다시 찾는다. 없는 답은 facts=[]로 남긴다.
사용자 지시는 탐색 방향이며 사실의 출처가 아니다. elements만 사실의 출처다.
statement·조건·수치는 연결 원문으로만 작성하고, 미수행·계획·해석·AI 조언을 관측 성과로 바꾸지 않는다.
user_addendum은 사용자가 새로 제공한 기록이며 PDF에서 확인한 내용이나 독립 검증 결과가 아니다.
coverage는 빈 객체로 반환한다."""
    result = []
    for index, batch in enumerate(source_batches(elements), 1):
        value = engine.call(target, f"search-{index}", prompt,
            {"question": decision["question"], "user_instruction": decision["instruction"],
             "elements": source_packet(batch)}, extraction_schema(batch, []))
        grounded = []
        for raw in value["facts"]:
            raw = copy.deepcopy(raw)
            raw.update(id="H" + decision["id"][:12] + f"_{len(result) + len(grounded)}",
                       requirement_ids=[decision["question_id"]], supplemental_reason="사용자 지시에 따른 추가 탐색")
            grounded.append(ground(raw, batch))
        if grounded:
            verified = verify_page(engine, target, batch, {"page": index, "facts": grounded}, stage="human-input")["facts"]
            for fact in verified:
                if decision["action"] == "provide_information":
                    fact["limits"] += "\n사용자 추가 기록과 대조한 내용이며 PDF 원문 또는 독립 검증으로 확인한 사실은 아닙니다."
                    fact["user_provided"] = True
            result.extend(verified)
    write_json(target / "facts.json", result)
    return result


def apply_decisions(engine, folder, requirements, items, source):
    items = copy.deepcopy(items)
    history = decisions(folder)
    known = {q["id"] for q in requirements["requirements"]}
    for row in history:
        if row["question_id"] not in known:
            raise ValueError("사용자 지시가 현재 질문 파일과 맞지 않습니다.")
        if row["action"] in {"provide_information", "recheck"}:
            for fact in supplemental_facts(engine, folder, row, source):
                if fact["id"] not in {f["id"] for f in items}:
                    items.append(fact)
    return items, history


def reference_decisions(catalog, research):
    """Remove excluded criteria from all writer/reviewer guides, retaining shared active criteria."""
    catalog = copy.deepcopy(catalog)
    questions = research.get("evidence_requirements", {}).get("requirements", [])
    excluded = {key for q in questions if research.get("requirement_coverage", {}).get(q["id"], {}).get("status") == "excluded_by_user"
                for key in q.get("guideline_ids", [])}
    active = {key for q in questions if research.get("requirement_coverage", {}).get(q["id"], {}).get("status") != "excluded_by_user"
              for key in q.get("guideline_ids", [])}
    blocked = excluded - active
    for guide in catalog.values():
        guide["criteria"] = [c for c in guide.get("criteria", []) if c["id"] not in blocked]
        guide["required_content"] = [c["description"] for c in guide["criteria"]]
    return catalog


def interrupt_unresolved(package):
    questions = {q["id"]: q for q in package["requirements"]["requirements"]}
    pending = package["completion"].get("unresolved_requirement_ids", []) + package["completion"].get("unresolved_applicability_ids", [])
    if pending:
        key = pending[0]
        pause(package["run_dir"], questions[key], package["requirement_coverage"][key],
              [f for f in package["items"] if key in f["requirement_ids"]], "이 질문의 근거가 부족하거나 적용 여부가 미확인입니다.")
