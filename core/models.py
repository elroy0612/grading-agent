"""채점 파이프라인에서 주고받는 데이터 구조."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


class ParseError(ValueError):
    """사용자에게 그대로 보여줄 수 있는 한국어 파싱 오류."""


@dataclass(frozen=True)
class AnswerItem:
    number: int
    answer: int  # 1~5
    points: int


@dataclass
class AnswerKey:
    items: dict[int, AnswerItem]
    warnings: list[str] = field(default_factory=list)

    @property
    def numbers(self) -> list[int]:
        return sorted(self.items)

    @property
    def total_points(self) -> int:
        return sum(i.points for i in self.items.values())


@dataclass
class OMRData:
    names: list[str]                 # 학생명 (OMR 원래 순서)
    responses: pd.DataFrame          # index: list_position(1..N), columns: 문항 번호(int), 값: 0(무응답)~5
    student_ids: list[str] | None    # 별도 번호/학번 열이 있으면 그 값
    warnings: list[str] = field(default_factory=list)


@dataclass
class QuestionText:
    number: int
    stem: str                        # 발문
    body: str                        # 문항 고유 본문(지문·선택지)
    qtype: str                       # 유형 (발문 키워드 기반)
    is_listening: bool
    set_range: tuple[int, int] | None = None
    set_instruction: str = ""
    set_passage: str = ""


@dataclass
class GradingResult:
    answer_key: AnswerKey
    responses: pd.DataFrame          # index: list_position, columns: 문항 번호
    correct: pd.DataFrame            # bool, 같은 shape
    students: pd.DataFrame           # index: list_position / 학생명, 학번, name_number, 총점, 정답수, 순위
    items: pd.DataFrame              # index: 문항 번호 / 정답, 배점, 정답자수, 오답자수, 정답률, 유형, 듣기, 선택분포
    summary: dict
    questions: dict[int, QuestionText]
    warnings: list[str] = field(default_factory=list)
