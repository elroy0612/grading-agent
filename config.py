"""설정/비밀키 조회: os.environ(.env) → st.secrets 순서."""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent / ".env")
except ImportError:  # 배포 환경에서 dotenv가 없어도 동작
    pass

BASE_DIR = Path(__file__).parent
LOGO_PATH = BASE_DIR / "assets" / "gachon_logo.jpg"


def get_secret(key: str, default: str | None = None) -> str | None:
    value = os.environ.get(key)
    if value:
        return value
    try:
        import streamlit as st

        return st.secrets.get(key, default)
    except Exception:  # noqa: BLE001  (secrets.toml 없음 등)
        return default


ACADEMY_NAME = get_secret("ACADEMY_NAME", "OO 학원")
APP_TITLE = f"{ACADEMY_NAME} 평가 도우미"
OPENAI_MODEL = get_secret("OPENAI_MODEL", "gpt-5.4-mini")
OPENAI_REASONING_EFFORT = get_secret("OPENAI_REASONING_EFFORT", "low")  # 실측: none은 도구 확인 규칙을 건너뛰는 경우가 있음
# 유사 문제를 작성하는 호출에만 쓰는 추론 강도(정답이 하나만 성립하는 문제를 만들도록)
OPENAI_GENERATION_EFFORT = get_secret("OPENAI_GENERATION_EFFORT", "medium")
