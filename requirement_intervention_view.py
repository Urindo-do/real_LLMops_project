"""Streamlit pause/resume form; instructions are scoped to the displayed question."""

import streamlit as st

from planned_paper import PlannedPaper
from requirement_intervention import read_json, submit_instruction
from pathlib import Path


def render_intervention(base_dir):
    pending = st.session_state.get("pending_requirement")
    if not pending:
        return None
    saved = read_json(Path(pending["run_dir"]) / "pending-question.json", {})
    st.markdown("### 근거 질문 · 사용자 판단 대기")
    request = st.session_state.get("pending_requirement_request", {})
    if request.get("research_pdf_path"):
        st.caption("연구 자료: " + Path(request["research_pdf_path"]).name)
    st.info(pending["reason"])
    st.markdown("**" + pending["question"]["question"] + "**")
    st.write("확인할 항목: " + " / ".join(pending["question"]["expected_fields"]))
    assessment = pending.get("assessment", {})
    if assessment.get("reason"):
        st.write("AI 검토: " + assessment["reason"])
    if assessment.get("missing_fields"):
        st.write("부족 항목: " + " / ".join(assessment["missing_fields"]))
    if assessment.get("evidence_ids"):
        st.caption("AI 응답에 연결된 검토 근거 ID: " + ", ".join(assessment["evidence_ids"]))
    with st.expander(f"이 질문에서 검토한 근거 {len(pending['facts'])}개"):
        for fact in pending["facts"]:
            st.write(f"{fact['id']} · {fact['statement']}")
            st.caption(fact.get("limits", ""))
            for source in fact.get("sources", []):
                st.caption("사용자 추가 기록" if source.get("origin") == "user_addendum" else f"{source.get('page')}쪽")
                st.text(source["quote"])
    if saved.get("id") == pending["id"] and saved.get("status") == "instruction_saved":
        st.info("처리 지시가 저장되어 있습니다. 이어서 실행하면 같은 지시를 다시 해석하지 않습니다.")
        if st.button("저장한 지시로 다시 계속", key="resume_saved_requirement_" + pending["id"]):
            return st.session_state["pending_requirement_request"]
        return None
    with st.form("requirement_instruction_" + pending["id"]):
        instruction = st.text_area("이 질문을 어떻게 처리할까요?", placeholder=
            "예: 정확도는 측정하지 않았으니 해당 항목은 계획과 초안에서 빼줘. / 추가 기록은 다음과 같아: … / 30페이지를 다시 찾아봐.")
        st.caption("이 지시는 해당 질문에 적용됩니다. 지시 해석과 재검토에 Upstage를 사용하고, 완료한 추출은 재사용합니다.")
        submitted = st.form_submit_button("지시를 반영하고 계속", type="primary")
    if not submitted:
        return None
    try:
        with st.spinner("사용자 지시를 해석하고 저장 중입니다..."):
            record = submit_instruction(PlannedPaper(base_dir), pending["run_dir"], pending["id"], instruction)
        st.session_state["last_requirement_instruction"] = record
        st.success(record["explanation"])
        return st.session_state["pending_requirement_request"]
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        st.error(f"지시를 반영하지 못했습니다: {error}")
        return None
