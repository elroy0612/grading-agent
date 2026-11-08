"""업로드된 파일을 OMR / 정답표 / 문제지로 자동 분류. 내용 판정이 우선이고 파일명은 보조 단서."""
from __future__ import annotations

import io

import pdfplumber

OMR, ANSWER, PAPER, UNKNOWN = "OMR 결과", "정답표", "문제지", "알 수 없음"
KINDS = [PAPER, OMR, ANSWER]


def classify_file(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith((".xlsx", ".csv")):
        return OMR
    if not name.endswith(".pdf"):
        return UNKNOWN
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            pages = len(pdf.pages)
            text = "".join((p.extract_text() or "") for p in pdf.pages[:2])
    except Exception:  # noqa: BLE001
        pages, text = 0, ""
    if text.strip():
        return ANSWER if pages <= 2 and "정답" in text and "배점" in text else PAPER
    # 글자를 읽을 수 없는 PDF → 파일명으로 추정(이후 파서가 명확한 오류를 냄)
    return ANSWER if "정답" in filename else PAPER
