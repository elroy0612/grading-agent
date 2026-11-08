"""정답표 PDF → 문항 번호·정답·배점.

정답과 배점은 오직 업로드된 PDF에서 추출한다(코드/별도 파일에 정답 없음).
1순위: pdfplumber 표 추출(헤더의 번호/정답/배점 위치로 묶음 해석)
2순위: 텍스트 정규식
"""
from __future__ import annotations

import io
import re

import pdfplumber

from core.models import AnswerItem, AnswerKey, ParseError

CIRCLED = {"①": 1, "②": 2, "③": 3, "④": 4, "⑤": 5}
TEXT_PATTERN = re.compile(r"(\d{1,3})\s*([①②③④⑤])\s*(\d{1,2})")


def _clean(cell) -> str:
    return re.sub(r"\s+", "", str(cell)) if cell is not None else ""


def _to_answer(cell: str) -> int | None:
    if cell in CIRCLED:
        return CIRCLED[cell]
    if cell.isdigit() and 1 <= int(cell) <= 5:
        return int(cell)
    return None


def _parse_table(table: list[list]) -> dict[int, AnswerItem]:
    rows = [[_clean(c) for c in row] for row in table if row]
    header_idx = next((i for i, r in enumerate(rows) if any("번호" in c for c in r)), None)
    if header_idx is None:
        return {}
    header = rows[header_idx]
    num_cols = [i for i, c in enumerate(header) if "번호" in c]
    groups = []  # (번호 열, 정답 열, 배점 열)
    for k, ni in enumerate(num_cols):
        end = num_cols[k + 1] if k + 1 < len(num_cols) else len(header)
        ans_col = next((j for j in range(ni + 1, end) if "정답" in header[j]), None)
        pts_col = next((j for j in range(ni + 1, end) if "배점" in header[j]), None)
        if ans_col is not None and pts_col is not None:
            groups.append((ni, ans_col, pts_col))

    items: dict[int, AnswerItem] = {}
    for row in rows[header_idx + 1:]:
        for ni, ai, pi in groups:
            if max(ni, ai, pi) >= len(row):
                continue
            num, ans, pts = row[ni], _to_answer(row[ai]), row[pi]
            if num.isdigit() and ans is not None and pts.isdigit():
                items[int(num)] = AnswerItem(int(num), ans, int(pts))
    return items


def _parse_text(pdf) -> dict[int, AnswerItem]:
    text = "\n".join(p.extract_text() or "" for p in pdf.pages)
    items = {}
    for num, ans, pts in TEXT_PATTERN.findall(text):
        items.setdefault(int(num), AnswerItem(int(num), CIRCLED[ans], int(pts)))
    return items


def parse_answer_key(pdf_bytes: bytes) -> AnswerKey:
    warnings: list[str] = []
    try:
        pdf = pdfplumber.open(io.BytesIO(pdf_bytes))
    except Exception as e:  # noqa: BLE001
        raise ParseError(f"정답 PDF를 열 수 없습니다: {e}") from e

    with pdf:
        items: dict[int, AnswerItem] = {}
        tables_used = 0
        for page in pdf.pages:
            for table in page.extract_tables():
                parsed = _parse_table(table)
                if not parsed:
                    continue
                if items and set(parsed) & set(items):
                    warnings.append("정답표가 여러 개(예: 홀수형/짝수형) 있어 첫 번째 표만 사용했습니다.")
                    continue
                items.update(parsed)
                tables_used += 1
        if not items:
            items = _parse_text(pdf)
            if items:
                warnings.append("표 인식에 실패해 텍스트에서 정답을 추출했습니다. 결과를 확인해 주세요.")
        if not items:
            has_text = any((p.extract_text() or "").strip() for p in pdf.pages)
            if not has_text:
                raise ParseError("정답 PDF에서 글자를 읽을 수 없습니다. 스캔(이미지) PDF는 지원하지 않습니다.")
            raise ParseError("정답 PDF에서 '문항 번호 / 정답 / 배점' 표를 찾지 못했습니다.")

    numbers = sorted(items)
    expected = list(range(1, numbers[-1] + 1))
    if numbers != expected:
        missing = sorted(set(expected) - set(numbers))
        raise ParseError(f"정답표에서 일부 문항을 읽지 못했습니다. 누락 문항: {missing}")

    return AnswerKey(items=items, warnings=warnings)
