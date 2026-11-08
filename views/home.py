"""페이지 1: 서비스 소개 랜딩."""
import streamlit as st

import config

st.markdown(
    """
    <style>
    .hero-title {text-align:center; font-size:2.6rem; font-weight:800; margin:0.4rem 0 0.2rem;}
    .hero-sub {text-align:center; font-size:1.1rem; opacity:0.75; margin-bottom:1.6rem;}
    .feature {border:1px solid rgba(128,128,128,0.25); border-radius:14px; padding:1.1rem 1.2rem; height:100%;}
    .feature h4 {margin:0 0 0.4rem;}
    .feature p {margin:0; opacity:0.8; font-size:0.95rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

_, mid, _ = st.columns([2, 1, 2])
with mid:
    st.image(str(config.LOGO_PATH), width="stretch")

st.markdown(f'<div class="hero-title">{config.APP_TITLE}</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">문제지 · OMR 결과 · 정답표만 올리면 자동 채점부터 성적 분석, 맞춤 문제 생성까지 한 번에</div>',
    unsafe_allow_html=True,
)

_, btn, _ = st.columns([2, 1, 2])
with btn:
    if st.button("시작하기", type="primary", icon=":material/arrow_forward:", width="stretch"):
        st.switch_page("views/analysis.py")

st.write("")
features = [
    ("📝 자동 채점", "정답표 PDF에서 문항별 정답과 배점을 읽어 OMR 결과를 즉시 채점합니다."),
    ("📊 엑셀 리포트", "평균·중앙값·표준편차, 가장 많이 틀린 문항, 학생별 문항 정오표를 엑셀로 내려받습니다."),
    ("💬 AI 분석 챗봇", "\"가장 성적이 저조한 학생은?\"처럼 물어보면 채점 결과로 답하고, 오답이 많은 유형의 유사 문제도 만들어 줍니다."),
]
for col, (title, desc) in zip(st.columns(3), features):
    col.markdown(f'<div class="feature"><h4>{title}</h4><p>{desc}</p></div>', unsafe_allow_html=True)

st.write("")
with st.expander("사용 방법", expanded=True):
    st.markdown(
        """
1. 상단의 **채점 · 분석** 페이지로 이동합니다.
2. 왼쪽 사이드바에 **문제지 PDF, OMR 결과(xlsx/csv), 정답 PDF**를 한 번에 끌어다 놓습니다.
3. **분석 시작**을 누르면 자동 채점이 진행됩니다.
4. 채점이 끝나면 **엑셀 추출** 버튼과 **채팅**이 활성화됩니다.
"""
    )
