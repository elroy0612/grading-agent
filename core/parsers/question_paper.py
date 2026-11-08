"""문제지 PDF(2단 편집) → 문항별 발문·본문·세트 지문·유형.

평가원 형식 가정: 좌우 2단, 문항은 'N.' 으로 시작, 세트 문항은 '[a～b] 지시문' 헤더.
파싱 실패는 채점을 막지 않는다(유사 문제 생성에만 사용).
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

import pdfplumber

from core.models import ParseError, QuestionText

NOISE = re.compile(r"^(\d{1,2}|[홀짝]수형|.*저작권은.*|.*수학능력시험 문제지|20\d\d학년도 대학수.*|제 ?\d ?교시.*)$")
SET_HDR = re.compile(r"^\[(\d{1,2})\s*[~～∼〜\-]\s*(\d{1,2})\]\s*(.*)$")
Q_START = re.compile(r"^(\d{1,2})\.(?:\s+|$)(.*)$")
STOP = re.compile(r"^(이제 듣기 문제가 끝났습니다|\* ?확인 사항)")
PUA = re.compile(r"[-\U000f0000-\U000ffffd]")
STEM_END = re.compile(r"(\?|시오\.)\s*(\[\d점\])?\s*$")
POINTS_TAG = re.compile(r"\s*\[\d점\]")
HANGUL = re.compile(r"[가-힣]")

# (패턴, 유형) — 위에서부터 먼저 맞는 것을 사용
TYPE_RULES: list[tuple[str, str]] = [
    (r"요약", "요약문 완성"),
    (r"빈칸", "빈칸 추론"),
    (r"어법", "어법"),
    (r"낱말의 쓰임", "어휘(낱말 쓰임)"),
    (r"가리키는 대상", "지칭 대상"),
    (r"밑줄 친.*의미", "함축 의미 추론"),
    (r"관계 없는", "무관한 문장"),
    (r"주어진 문장이 들어가기", "문장 삽입"),
    (r"순서", "글의 순서"),
    (r"그림", "그림 내용 불일치"),
    (r"표를 보면서", "도표 정보 활용"),
    (r"도표", "도표 내용 불일치"),
    (r"안내문", "안내문 내용 일치"),
    (r"응답", "마지막 말에 대한 응답"),
    (r"할 말", "상황에 적절한 말"),
    (r"할 일", "할 일"),
    (r"금액", "지불 금액"),
    (r"이유", "이유"),
    (r"언급(되지 않은|된.*아닌)", "언급되지 않은 것"),
    (r"심경", "심경 변화"),
    (r"주장", "필자의 주장"),
    (r"의견", "의견"),
    (r"목적", "목적"),
    (r"요지", "요지"),
    (r"주제", "주제"),
    (r"제목", "제목"),
    (r"일치|내용으로 적절", "내용 일치/불일치"),
]


def classify_type(stem: str) -> str:
    for pattern, name in TYPE_RULES:
        if re.search(pattern, stem):
            return name
    return "기타"


@dataclass
class _Q:
    number: int
    lines: list[str] = field(default_factory=list)
    group: tuple[int, int] | None = None


def _column_lines(pdf) -> list[str]:
    out = []
    for page in pdf.pages:
        w, h = page.width, page.height
        for x0, x1 in ((0, w / 2), (w / 2, w)):
            text = page.crop((x0, 0, x1, h)).extract_text() or ""
            for ln in text.splitlines():
                ln = PUA.sub("", ln).strip()
                if ln and not NOISE.match(ln):
                    out.append(ln)
    return out


def _split_stem(lines: list[str]) -> tuple[str, list[str]]:
    """앞쪽 최대 3줄 안에서 발문 종결('?', '시오.')을 찾는다. 없으면 발문 없음."""
    for i in range(min(3, len(lines))):
        if STEM_END.search(lines[i]):
            stem = " ".join(lines[: i + 1])
            if HANGUL.search(stem):
                return stem, lines[i + 1:]
            break
    return "", lines


def parse_question_paper(pdf_bytes: bytes) -> dict[int, QuestionText]:
    try:
        pdf = pdfplumber.open(io.BytesIO(pdf_bytes))
    except Exception as e:  # noqa: BLE001
        raise ParseError(f"문제지 PDF를 열 수 없습니다: {e}") from e
    with pdf:
        lines = _column_lines(pdf)
    if not lines:
        raise ParseError("문제지 PDF에서 글자를 읽을 수 없습니다. 스캔(이미지) PDF는 지원하지 않습니다.")

    qs: dict[int, _Q] = {}
    groups: dict[tuple[int, int], dict] = {}
    cur: _Q | None = None
    cur_group: tuple[int, int] | None = None
    mode = None  # 'q' | 'group' | 'skip'

    for ln in lines:
        if STOP.match(ln):
            mode = "skip"
            continue
        m = SET_HDR.match(ln)
        if m:
            cur_group = (int(m.group(1)), int(m.group(2)))
            groups[cur_group] = {"instruction": m.group(3).strip(), "passage": []}
            mode = "group"
            continue
        m = Q_START.match(ln)
        if m and int(m.group(1)) == (max(qs) + 1 if qs else 1):
            n = int(m.group(1))
            g = cur_group if cur_group and cur_group[0] <= n <= cur_group[1] else None
            cur = qs[n] = _Q(n, [m.group(2).strip()] if m.group(2).strip() else [], g)
            mode = "q"
            continue
        if mode == "group" and cur_group:
            gi = groups[cur_group]
            if not gi["passage"] and not STEM_END.search(gi["instruction"]):
                gi["instruction"] += " " + ln
            else:
                gi["passage"].append(ln)
        elif mode == "q" and cur is not None:
            cur.lines.append(ln)

    if not qs:
        raise ParseError("문제지에서 문항 번호('1.', '2.' …)를 찾지 못했습니다.")

    result: dict[int, QuestionText] = {}
    for n, q in qs.items():
        stem, body = _split_stem(q.lines)
        g = groups.get(q.group) if q.group else None
        instruction = g["instruction"] if g else ""
        if not stem and instruction:
            stem = instruction
        stem_clean = POINTS_TAG.sub("", stem).strip()
        listening = "듣고" in stem or "듣고" in instruction
        qtype = classify_type(stem_clean)
        result[n] = QuestionText(
            number=n,
            stem=stem_clean,
            body=" ".join(body),
            qtype=f"듣기 - {qtype}" if listening else qtype,
            is_listening=listening,
            set_range=q.group,
            set_instruction=instruction,
            set_passage=" ".join(g["passage"]) if g else "",
        )
    return result
