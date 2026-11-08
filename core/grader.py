"""정답표 + OMR → 채점 결과와 통계 (LLM 없이 pandas로 확정 계산)."""
from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

import pandas as pd

from core.models import AnswerKey, GradingResult, OMRData, ParseError, QuestionText

TOP_WRONG_N = 3


def fmt_num(x: float) -> str:
    """65.20 → '65.2', 65.0 → '65'."""
    return f"{x:.2f}".rstrip("0").rstrip(".")


def _r2(x) -> float:
    """소수 둘째 자리 사사오입(엑셀 ROUND와 동일). Python round()는 58.925 → 58.92가 된다."""
    return float(Decimal(repr(float(x))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def grade(answer_key: AnswerKey, omr: OMRData, questions: dict[int, QuestionText] | None = None) -> GradingResult:
    questions = questions or {}
    warnings = list(answer_key.warnings) + list(omr.warnings)

    key_nums = answer_key.numbers
    omr_nums = list(omr.responses.columns)
    common = [n for n in key_nums if n in omr_nums]
    if not common:
        raise ParseError("OMR의 문항 번호와 정답표의 문항 번호가 하나도 일치하지 않습니다.")
    if set(key_nums) != set(omr_nums):
        only_key = sorted(set(key_nums) - set(omr_nums))
        only_omr = sorted(set(omr_nums) - set(key_nums))
        warnings.append(
            f"정답표({len(key_nums)}문항)와 OMR({len(omr_nums)}문항)의 문항이 달라 공통 {len(common)}문항만 채점했습니다."
            + (f" 정답표에만 있음: {only_key}" if only_key else "")
            + (f" OMR에만 있음: {only_omr}" if only_omr else "")
        )

    responses = omr.responses[common]
    key = pd.Series({n: answer_key.items[n].answer for n in common})
    points = pd.Series({n: answer_key.items[n].points for n in common})

    n = len(responses)
    if n == 0:
        raise ParseError("OMR 파일에 학생 응답 행이 없습니다.")
    correct = responses.eq(key, axis="columns")
    score = correct.astype(int).mul(points, axis="columns").sum(axis=1)

    names = pd.Series(omr.names, index=responses.index)
    digits = names.str.extract(r"(\d+)")[0]
    digits = digits.where(digits.str.len() <= 9)  # 너무 긴 숫자는 번호로 보지 않음(Int64 변환 오류 방지)
    name_number = pd.to_numeric(digits, errors="coerce").astype("Int64")
    students = pd.DataFrame(
        {
            "학생명": names,
            "학번": omr.student_ids if omr.student_ids else pd.NA,
            "name_number": name_number,
            "총점": score.astype(int),
            "정답수": correct.sum(axis=1).astype(int),
            "순위": score.rank(method="min", ascending=False).astype(int),
        },
        index=responses.index,
    )

    correct_cnt = correct.sum(axis=0).astype(int)
    items = pd.DataFrame(
        {
            "정답": key,
            "배점": points,
            "정답자수": correct_cnt,
            "오답자수": n - correct_cnt,
            "정답률": (correct_cnt / n * 100).map(_r2),
            "오답률": ((n - correct_cnt) / n * 100).map(_r2),
        }
    )
    for choice in range(1, 6):
        items[f"선택_{choice}"] = (responses == choice).sum(axis=0).astype(int)
    items["무응답"] = (responses == 0).sum(axis=0).astype(int)
    items["유형"] = [questions[q].qtype if q in questions else "" for q in items.index]
    items["듣기"] = [questions[q].is_listening if q in questions else False for q in items.index]
    items.index.name = "문항"

    ranked = items.reset_index().sort_values(["오답자수", "문항"], ascending=[False, True])
    ranked = ranked[ranked["오답자수"] > 0]  # 전원이 맞힌 문항은 '많이 틀린 문항'이 아님
    top = ranked.head(TOP_WRONG_N)
    boundary = int(top["오답자수"].iloc[-1]) if len(top) else 0
    tied_out = ranked.iloc[TOP_WRONG_N:]
    tied_out = tied_out[tied_out["오답자수"] == boundary]["문항"].astype(int).tolist()

    def _who(value: int) -> list[str]:
        return students.loc[students["총점"] == value, "학생명"].tolist()

    max_score = int(points.sum())
    summary = {
        "n_students": n,
        "n_questions": len(common),
        "max_score": max_score,
        "mean": _r2(score.mean()),
        "median": _r2(score.median()),
        "std": _r2(score.std(ddof=1)) if n > 1 else 0.0,
        "highest": {"score": int(score.max()), "students": _who(int(score.max()))},
        "lowest": {"score": int(score.min()), "students": _who(int(score.min()))},
        "top_wrong": [
            {
                "number": int(r["문항"]),
                "wrong_count": int(r["오답자수"]),
                "wrong_rate": float(r["오답률"]),
                "type": r["유형"],
            }
            for _, r in top.iterrows()
        ],
        "top_wrong_ties_excluded": tied_out,
        "blank_responses": int((responses == 0).to_numpy().sum()),
    }

    return GradingResult(
        answer_key=answer_key,
        responses=responses,
        correct=correct,
        students=students,
        items=items,
        summary=summary,
        questions=questions,
        warnings=warnings,
    )
