"""앱 코드와 독립적인 기대값 계산기 (numpy + 순수 Python, pandas/앱 모듈 미사용)."""
from __future__ import annotations

import math
import re
import unicodedata
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction

import numpy as np

CIRCLED = {1: "①", 2: "②", 3: "③", 4: "④", 5: "⑤"}
_CTRL = re.compile(r"[\x00-\x1f\x7f]")


def half_up(x, nd: int = 2) -> float:
    """한국식 반올림(사사오입). x는 Fraction 또는 float."""
    if isinstance(x, Fraction):
        d = Decimal(x.numerator) / Decimal(x.denominator)
    else:
        d = Decimal(repr(float(x)))
    return float(d.quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def fmt(x: float) -> str:
    """65.20 → '65.2', 65.0 → '65' (앱 fmt_num과 독립 구현)."""
    s = f"{Decimal(repr(float(x))).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def clean_name(v) -> str:
    """ARCHITECTURE §4.3 명세: NFKC, 개행·제어문자 제거, 30자."""
    s = unicodedata.normalize("NFKC", str(v)).strip()
    return _CTRL.sub("", s)[:30]


def expected(names: list[str], resp: np.ndarray, qnums: list[int], key: dict[int, tuple[int, int]]) -> dict:
    """resp: (N, len(qnums)) 0~5 정수. key: 문항 → (정답, 배점)."""
    common = [q for q in sorted(key) if q in qnums]
    cols = [qnums.index(q) for q in common]
    R = np.asarray(resp)[:, cols]
    K = np.array([key[q][0] for q in common])
    P = np.array([key[q][1] for q in common])
    C = R == K[None, :]
    score = [int(x) for x in (C * P[None, :]).sum(axis=1)]
    ncorrect = [int(x) for x in C.sum(axis=1)]
    n = len(names)
    rank = [1 + sum(1 for t in score if t > s) for s in score]

    total = sum(score)
    mean_exact = Fraction(total, n)
    srt = sorted(score)
    median_exact = Fraction(srt[n // 2]) if n % 2 else Fraction(srt[n // 2 - 1] + srt[n // 2], 2)
    if n > 1:
        var = sum((Fraction(s) - mean_exact) ** 2 for s in score) / (n - 1)
        std_true = math.sqrt(float(var))
    else:
        std_true = 0.0

    correct_cnt = [int(x) for x in C.sum(axis=0)]
    wrong = [n - c for c in correct_cnt]
    order = sorted(range(len(common)), key=lambda j: (-wrong[j], common[j]))
    top = order[:3]
    boundary = wrong[top[-1]] if top else 0
    ties = [common[j] for j in order[3:] if wrong[j] == boundary]

    hi, lo = max(score), min(score)
    dist = {q: {c: int((R[:, j] == c).sum()) for c in range(0, 6)} for j, q in enumerate(common)}
    return {
        "n": n,
        "common": common,
        "correct": C,
        "score": score,
        "ncorrect": ncorrect,
        "rank": rank,
        "max_score": int(P.sum()),
        "mean_exact": mean_exact,
        "mean_half_up": half_up(mean_exact),
        "median": half_up(median_exact),
        "std_true": std_true,
        "top3": [(common[j], wrong[j]) for j in top],
        "top3_rates": [float(Fraction(wrong[j] * 100, n)) for j in top],
        "ties_excluded": ties,
        "highest": (hi, [names[i] for i in range(n) if score[i] == hi]),
        "lowest": (lo, [names[i] for i in range(n) if score[i] == lo]),
        "correct_cnt": dict(zip(common, correct_cnt)),
        "wrong_cnt": dict(zip(common, wrong)),
        "dist": dist,
        "blank_total": int((R == 0).sum()),
    }


def ranked(score: list[int], names: list[str], order: str, n: int) -> list[str]:
    """경계 동점자 포함 상/하위 n명 (이름 집합 비교용)."""
    bottom = order == "bottom"
    srt = sorted(score) if bottom else sorted(score, reverse=True)
    n = max(1, min(n, len(score)))
    b = srt[n - 1]
    return [names[i] for i, s in enumerate(score) if (s <= b if bottom else s >= b)]
