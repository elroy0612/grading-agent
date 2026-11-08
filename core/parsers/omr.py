"""OMR 인식 결과(xlsx/csv) → 학생별 응답표.

기대 형식(샘플): 학생명 | Q01 … Q45, 한 행 = 한 학생, 응답 1~5.
번호·반 같은 식별 열, 제목 행, 여러 시트, 무응답·원문자 응답도 처리한다.
"""
from __future__ import annotations

import csv
import io
import re
import unicodedata

import pandas as pd

from core.models import OMRData, ParseError

Q_COL = re.compile(r"^(?:q|문항|문제)?\s*0*(\d{1,3})\s*(?:번)?$", re.I)
NAME_COL = re.compile(r"(학생명|이름|성명|name)", re.I)
ID_COL = re.compile(r"^(번호|학번|no\.?|id)$", re.I)
CIRCLED = {"①": 1, "②": 2, "③": 3, "④": 4, "⑤": 5}
CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f​-‏  ‪-‮⁦-⁩﻿]")
MIN_QUESTION_COLS = 5
HEADER_SCAN_ROWS = 10


def _question_number(col) -> int | None:
    m = Q_COL.match(str(col).strip())
    if not m:
        return None
    n = int(m.group(1))
    return n if n >= 1 else None


def _read_csv(data: bytes) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp949"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ParseError("CSV 파일의 인코딩을 읽을 수 없습니다(UTF-8 또는 CP949로 저장해 주세요).")
    rows = list(csv.reader(io.StringIO(text)))
    width = max((len(r) for r in rows), default=0)
    if width == 0:
        raise ParseError("CSV 파일이 비어 있습니다.")
    padded = [[c if c != "" else None for c in r] + [None] * (width - len(r)) for r in rows]
    return pd.DataFrame(padded, dtype=object)


def _read_frames(data: bytes, filename: str) -> list[pd.DataFrame]:
    if filename.lower().endswith(".csv"):
        return [_read_csv(data)]
    try:
        sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, header=None, dtype=str,
                               keep_default_na=False, na_values=[""])
    except Exception as e:  # noqa: BLE001  (영문 예외 문구는 사용자에게 보이지 않게)
        raise ParseError("OMR 엑셀 파일을 열 수 없습니다. xlsx 또는 csv 형식인지 확인해 주세요.") from e
    return list(sheets.values())


def _locate_header(df: pd.DataFrame) -> tuple[int, int]:
    """(머리글 행, 서로 다른 문항 번호 수).

    응답 행은 값이 1~5뿐이라 '서로 다른 번호'가 최대 5개 → 문항 머리글(45개)과 구분된다.
    (셀 개수로 세면 번호·반 열이나 숫자 이름이 있는 데이터 행이 머리글로 오인됨)
    """
    best = (0, 0)
    for i in range(min(HEADER_SCAN_ROWS, len(df))):
        nums = {n for v in df.iloc[i].tolist() if (n := _question_number(v)) is not None}
        if len(nums) > best[1]:
            best = (i, len(nums))
    return best


def _clean_name(value) -> str:
    s = unicodedata.normalize("NFKC", str(value)).strip()
    return CONTROL.sub("", s)[:30]


def _to_response(value) -> int:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0
    s = str(value).strip()
    if s in CIRCLED:
        return CIRCLED[s]
    try:
        f = float(s)
    except ValueError:
        return 0
    return int(f) if f.is_integer() and 1 <= f <= 5 else 0


def parse_omr(data: bytes, filename: str) -> OMRData:
    candidates = []
    for df in _read_frames(data, filename):
        header_row, count = _locate_header(df)
        rows = int(df.iloc[header_row + 1:].dropna(how="all").shape[0]) if count else 0
        candidates.append((count, rows, header_row, df))
    count, n_rows, header_row, raw = max(candidates, key=lambda c: (c[0], c[1]), default=(0, 0, 0, None))
    if raw is None or count < MIN_QUESTION_COLS:
        raise ParseError("OMR 파일에서 문항 열(예: Q01, 1번)을 찾지 못했습니다.")

    df = raw.iloc[header_row + 1:].reset_index(drop=True)
    df.columns = [str(c).strip() if c is not None and not (isinstance(c, float) and pd.isna(c)) else f"열{i}"
                  for i, c in enumerate(raw.iloc[header_row].tolist())]
    df = df.dropna(how="all").reset_index(drop=True)
    if df.empty:
        raise ParseError("OMR 파일에 학생 응답 행이 없습니다.")

    q_cols = {c: _question_number(c) for c in df.columns if _question_number(c) is not None}
    numbers = list(q_cols.values())
    if len(numbers) != len(set(numbers)):
        dup = sorted({n for n in numbers if numbers.count(n) > 1})
        raise ParseError(f"OMR 파일에 같은 문항 번호의 열이 2개 이상 있습니다: {dup}")
    other_cols = [c for c in df.columns if c not in q_cols]

    name_col = next((c for c in other_cols if NAME_COL.search(c)), None)
    id_col = next((c for c in other_cols if ID_COL.match(c)), None)
    if name_col is None:
        name_col = next((c for c in other_cols if c != id_col), None)
    if name_col is None and id_col is None:
        raise ParseError("OMR 파일에서 학생 이름(또는 번호) 열을 찾지 못했습니다.")

    warnings: list[str] = []
    source = df[name_col] if name_col else df[id_col]
    names = []
    for k, v in enumerate(source.tolist(), start=1):
        name = _clean_name(v) if v is not None and not (isinstance(v, float) and pd.isna(v)) else ""
        if not name:
            name = f"(이름 없음 {k}번)"
            warnings.append(f"명단 {k}번째 학생의 이름이 비어 있어 '{name}'로 표시합니다.")
        names.append(name)
    student_ids = [_clean_name(v) if v is not None else "" for v in df[id_col]] if id_col and name_col else None

    n = len(df)
    responses = pd.DataFrame(
        {num: df[col].map(_to_response).to_numpy() for col, num in q_cols.items()},
        index=pd.RangeIndex(1, n + 1, name="list_position"),
    )
    responses = responses[sorted(responses.columns)]

    blank = int((responses == 0).to_numpy().sum())
    if blank:
        warnings.append(f"무응답·인식 불가 응답 {blank}건은 오답으로 처리했습니다.")
    if len(set(names)) != len(names):
        warnings.append("이름이 같은 학생이 있습니다. 채팅에서는 명단 순서(예: 5번째 학생)로 구분해 질문해 주세요.")

    return OMRData(names=names, responses=responses, student_ids=student_ids, warnings=warnings)
