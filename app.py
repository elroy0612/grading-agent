"""엔트리포인트: streamlit run app.py"""
import streamlit as st

import config

st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=str(config.LOGO_PATH),
    layout="wide",
)

home = st.Page("views/home.py", title="홈", icon=":material/home:", default=True)
analysis = st.Page("views/analysis.py", title="채점 · 분석", icon=":material/analytics:")

st.navigation([home, analysis], position="top").run()
