"""페이지 2: 데이터 업로드 · 자동 채점 · 엑셀 추출 · 채팅."""
import hashlib

import streamlit as st

import config
from agent.agent import MSG_ERROR, MissingAPIKey, answer_stream, build_session, finalize_answer
from agent.guard import REFUSAL, leaks_internal
from core.exporter import build_excel
from core.grader import fmt_num, grade
from core.models import ParseError
from core.parsers.answer_key import parse_answer_key
from core.parsers.classify import ANSWER, KINDS, OMR, PAPER, classify_file
from core.parsers.omr import parse_omr
from core.parsers.question_paper import parse_question_paper

SKIP = "사용 안 함"
GREETING = ("채점이 완료되었습니다. 채점 결과에 대해 무엇이든 물어보세요.\n\n"
            "예) *학생들 평균이랑 중앙값이 뭐야?* · *가장 성적이 저조한 학생은 누구야?* · "
            "*가장 많이 틀린 문제와 비슷한 문제 1개 만들어 줘*")

for key, default in {"result": None, "excel_bytes": None, "messages": [], "analyzed_sig": None,
                     "chat_session": None, "notice": None}.items():
    st.session_state.setdefault(key, default)


@st.cache_data(show_spinner=False)
def _classify(name: str, data: bytes) -> str:
    return classify_file(name, data)


@st.cache_data(show_spinner=False)
def _parse_answer(data: bytes):
    return parse_answer_key(data)


@st.cache_data(show_spinner=False)
def _parse_omr(data: bytes, name: str):
    return parse_omr(data, name)


@st.cache_data(show_spinner=False)
def _parse_paper(data: bytes):
    return parse_question_paper(data)


def _md(text: str) -> str:
    """Streamlit 마크다운의 수식($)·취소선(~) 오작동 방지."""
    return text.replace("$", "\\$").replace("~", "\\~")


def _run_analysis(chosen: dict[str, tuple[str, bytes]]) -> None:
    sig = hashlib.sha256(b"".join(hashlib.sha256(chosen[k][1]).digest() for k in KINDS)).hexdigest()
    with st.status("채점 중…", expanded=True) as status:
        try:
            st.write("정답표를 읽는 중…")
            key = _parse_answer(chosen[ANSWER][1])
            st.write(f"✔ 정답표: {len(key.items)}문항, 배점 합계 {key.total_points}점")

            st.write("OMR 결과를 읽는 중…")
            omr = _parse_omr(chosen[OMR][1], chosen[OMR][0])
            st.write(f"✔ OMR: 학생 {len(omr.names)}명, {omr.responses.shape[1]}문항")

            st.write("문제지를 읽는 중…")
            extra = []
            try:
                questions = _parse_paper(chosen[PAPER][1])
                st.write(f"✔ 문제지: {len(questions)}문항")
            except Exception as e:  # noqa: BLE001  문제지 실패는 채점을 막지 않는다
                questions = {}
                reason = str(e) if isinstance(e, ParseError) else "문제지를 읽지 못했습니다."
                extra.append(f"문제지: {reason} (채점은 정상 진행, 유사 문제 생성 기능만 제한됩니다)")

            result = grade(key, omr, questions)
            result.warnings.extend(extra)
            excel = build_excel(result)
        except ParseError as e:
            status.update(label="채점 실패", state="error", expanded=True)
            st.error(str(e))
            return
        except Exception:  # noqa: BLE001  예상 못한 오류도 traceback 대신 안내
            status.update(label="채점 실패", state="error", expanded=True)
            st.error("파일을 처리하는 중 예기치 못한 오류가 발생했습니다. "
                     "제공된 샘플과 같은 형식(문제지 PDF · OMR 엑셀 · 정답 PDF)인지 확인해 주세요.")
            return

    if sig != st.session_state.analyzed_sig:
        st.session_state.messages = []
    st.session_state.update(result=result, excel_bytes=excel, analyzed_sig=sig, chat_session=None,
                            notice="채점이 완료되었습니다. 엑셀 추출과 채팅을 사용할 수 있습니다.")
    st.rerun()


# ── 사이드바 ─────────────────────────────────────────────────
with st.sidebar:
    st.subheader("데이터 업로드")
    files = st.file_uploader(
        "문제지 PDF · OMR 결과(xlsx/csv) · 정답 PDF를 한 번에 끌어다 놓으세요",
        type=["pdf", "xlsx", "csv"],
        accept_multiple_files=True,
    )

    chosen: dict[str, tuple[str, bytes]] = {}
    duplicated = False
    if files:
        st.caption("파일 종류 (자동 인식 · 틀리면 바꿔 주세요)")
        for f in files:
            data = f.getvalue()
            guess = _classify(f.name, data)
            options = KINDS + [SKIP]
            kind = st.selectbox(f.name, options, index=options.index(guess) if guess in options else len(KINDS),
                                key=f"kind_{f.file_id}")
            if kind in KINDS:
                duplicated |= kind in chosen
                chosen[kind] = (f.name, data)

    st.markdown(" · ".join(f"{'✅' if k in chosen else '⬜'} {k}" for k in KINDS))
    ready = all(k in chosen for k in KINDS) and not duplicated
    if duplicated:
        st.warning("같은 종류로 지정된 파일이 2개 이상입니다.")

    if st.button("분석 시작", type="primary", icon=":material/play_arrow:", disabled=not ready, width="stretch"):
        _run_analysis(chosen)

    result = st.session_state.result
    st.download_button(
        "엑셀 추출",
        data=st.session_state.excel_bytes or b"",
        file_name="채점결과.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        icon=":material/download:",
        disabled=result is None,
        on_click="ignore",
        width="stretch",
    )

# ── 메인: 요약 + 채팅 ────────────────────────────────────────
st.title(config.APP_TITLE)

if st.session_state.notice:
    st.toast(st.session_state.notice, icon="✅")
    st.session_state.notice = None

if result is None:
    st.info("왼쪽 사이드바에 **문제지 PDF, OMR 결과, 정답 PDF**를 올리고 **분석 시작**을 눌러 주세요. "
            "채점이 끝나면 엑셀 추출과 채팅이 활성화됩니다.")
else:
    s = result.summary
    cols = st.columns(5)
    cols[0].metric("응시자", f"{s['n_students']}명")
    cols[1].metric("평균", f"{fmt_num(s['mean'])}점")
    cols[2].metric("중앙값", f"{fmt_num(s['median'])}점")
    cols[3].metric("표준편차", f"{fmt_num(s['std'])}점")
    cols[4].metric("최다 오답", ", ".join(f"{t['number']}번" for t in s["top_wrong"]) or "없음(전원 정답)")
    if result.warnings:
        with st.expander(f"확인이 필요한 사항 {len(result.warnings)}건"):
            for w in result.warnings:
                st.write(f"- {w}")

history_box = st.container(height=520, border=True, autoscroll=True)
with history_box:
    if result is not None:
        with st.chat_message("assistant"):
            st.markdown(GREETING)
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(_md(m["content"]))

prompt = st.chat_input(
    "채점 결과에 대해 물어보세요" if result is not None else "채점을 먼저 진행해 주세요",
    disabled=result is None,
)

if prompt and result is not None:
    user_msg = {"role": "user", "content": prompt}
    with history_box:
        with st.chat_message("user"):
            st.markdown(_md(prompt))
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("⏳ 채점 데이터를 확인하고 있어요…")
            try:
                if st.session_state.chat_session is None:
                    st.session_state.chat_session = build_session(result)
                session = st.session_state.chat_session
                buf = ""
                for piece in answer_stream(session, st.session_state.messages + [user_msg]):
                    buf += piece
                    if leaks_internal(buf, session.canary):  # 스트리밍 중에도 내부 정보가 보이지 않게
                        buf = REFUSAL
                        break
                    placeholder.markdown(_md(buf) + "▌")
                answer = finalize_answer(session, buf) or "답변을 생성하지 못했습니다. 다시 질문해 주세요."
            except MissingAPIKey:
                answer = "OpenAI API 키가 설정되지 않아 채팅을 사용할 수 없습니다. 관리자에게 문의해 주세요."
            except Exception:  # noqa: BLE001
                answer = MSG_ERROR
            placeholder.markdown(_md(answer))
    # 질문과 답을 함께 저장 → 히스토리가 항상 user/assistant 쌍으로 유지됨
    st.session_state.messages += [user_msg, {"role": "assistant", "content": answer}]
