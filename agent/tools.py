"""채점 결과 조회 Tool. 모든 수치는 GradingResult에서 그대로 꺼낸다(LLM 계산 없음).

공통 규칙: JSON 문자열 반환(ensure_ascii=False), 예외 대신 found=false, 인자는 관대하게 정규화.
"""
from __future__ import annotations

import json
import re
import unicodedata

import pandas as pd
from langchain_core.tools import tool

from core.grader import fmt_num
from core.models import GradingResult

CIRCLED = {0: "무응답", 1: "①", 2: "②", 3: "③", 4: "④", 5: "⑤"}
# 숫자 질의에서 번호 외의 군더더기(조사·어미 포함). 이름 질의에는 쓰지 않는다('이은지'의 '이'가 지워지는 문제).
QUERY_FILLER = re.compile(r"(번째|번|학생|째|no\.?|#|점수|몇\s*점|명단|이야|이에요|인가요|알려줘|알려|줘|요|야"
                          r"|의|은|는|이|가|\s)", re.I)
PARTICLE_TAIL = re.compile(r"(학생|님)?(은|는|이|가|의|을|를|이야|이에요|요|야|한테|에게)?$")
STOP_TOKENS = {"학생", "점수", "몇", "점", "몇점", "알려줘", "알려", "줘", "누구", "어때", "성적"}
NON_WORD = re.compile(r"[^\w가-힣]")
CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f​-‏  ‪-‮⁦-⁩﻿]")
TEXT_LIMIT = 1500


def _dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _clip(text: str, limit: int = TEXT_LIMIT) -> str:
    return text if len(text) <= limit else text[:limit] + " …(이하 생략)"


# ── 학생 조회 ────────────────────────────────────────────────

def _student_record(r: GradingResult, pos: int) -> dict:
    st = r.students.loc[pos]
    wrong = [int(q) for q in r.correct.columns if not r.correct.at[pos, q]]
    rec = {
        "name": st["학생명"],
        "list_position": int(pos),
        "score": int(st["총점"]),
        "max_score": r.summary["max_score"],
        "correct_count": int(st["정답수"]),
        "wrong_count": len(wrong),
        "total_questions": r.summary["n_questions"],
        "rank": int(st["순위"]),
        "rank_of": r.summary["n_students"],
        "tied_count": int((r.students["총점"] == st["총점"]).sum()),
        "wrong_questions": wrong,
        "chosen_answers_on_wrong": {str(q): CIRCLED[int(r.responses.at[pos, q])] for q in wrong},
    }
    if isinstance(st["학번"], str) and st["학번"]:
        rec["student_id"] = st["학번"]
    return rec


def _available_range(r: GradingResult) -> str:
    names = r.students["학생명"].tolist()
    n = len(names)
    return f"{names[0]} ~ {names[-1]} (총 {n}명, 명단 순서 1~{n}번)"


def _label(r: GradingResult, pos: int) -> str:
    return f"{r.students.at[pos, '학생명']} (명단 {pos}번째)"


def _found(r: GradingResult, query, match_type: str, pos: int) -> dict:
    return {"found": True, "query": query, "match_type": match_type, "student": _student_record(r, pos)}


def _ambiguous(r: GradingResult, query, positions: list[int], why: str) -> dict:
    return {"found": False, "query": query, "candidates": [_label(r, p) for p in positions[:10]],
            "basis": f"{why} 어느 학생인지 명단 순서로 되물을 것."}


def _names_in_query(r: GradingResult, raw: str) -> list[int]:
    """질의 안에 통째로 들어 있는 학생 이름(숫자 경계 존중, 숫자만으로 된 이름 제외). 가장 긴 이름 우선."""
    hits = [p for p, nm in r.students["학생명"].items()
            if len(nm) >= 2 and not nm.isdigit() and re.search(rf"(?<!\d){re.escape(nm)}(?!\d)", raw)]
    if not hits:
        return []
    longest = max(len(r.students.at[p, "학생명"]) for p in hits)
    return [p for p in hits if len(r.students.at[p, "학생명"]) == longest]


def _name_tokens(raw: str) -> list[str]:
    """이름 일부 검색용 토큰: 공백 단위, 끝의 조사·'학생'만 떼고(가운데 글자는 보존), 2글자 이상."""
    out: list[str] = []
    for t in raw.split():
        t = NON_WORD.sub("", t)
        for c in (t, PARTICLE_TAIL.sub("", t)):
            if len(c) >= 2 and not c.isdigit() and c not in STOP_TOKENS and c not in out:
                out.append(c)
    return sorted(out, key=len, reverse=True)


def find_student(r: GradingResult, query: str) -> dict:
    st = r.students
    n = len(st)
    raw = CONTROL.sub("", unicodedata.normalize("NFKC", "" if query is None else str(query))).strip()
    if not raw:
        return {"found": False, "query": query, "available_range": _available_range(r), "basis": "학생이 지정되지 않음."}

    # 1) 이름이 정확히 일치  2) 질의 문장 안에 이름이 통째로 들어 있음
    exact = st.index[st["학생명"] == raw].tolist() or _names_in_query(r, raw)
    if len(exact) == 1:
        return _found(r, query, "exact_name", exact[0])
    if len(exact) > 1:
        same_name = len({st.at[p, "학생명"] for p in exact}) == 1
        return _ambiguous(r, query, exact, "같은 이름의 학생이 여러 명." if same_name else "질문에 여러 학생 이름이 들어 있음.")

    # 3) 숫자 질의: 번호(학번) 열 → 이름 속 번호(정확히 일치) → 명단 순서. 숫자는 부분 일치하지 않는다.
    digits = re.findall(r"\d+", raw)
    rest = NON_WORD.sub("", QUERY_FILLER.sub("", re.sub(r"\d+", "", raw)))
    rest_is_name = len(rest) >= 2 and st["학생명"].str.contains(re.escape(rest), na=False).any()
    if digits and not rest_is_name:
        num = int(digits[0])
        ordinal = "번째" in raw or "명단" in raw
        candidates: list[tuple[str, int]] = []
        if not ordinal:
            if st["학번"].notna().any():
                ids = st.index[pd.to_numeric(st["학번"].astype(str).str.extract(r"(\d+)")[0],
                                             errors="coerce") == num].tolist()
                if len(ids) > 1:
                    return _ambiguous(r, query, ids, f"번호가 {num}인 학생이 여러 명.")
                if len(ids) == 1:
                    candidates.append(("student_id", ids[0]))
            nn = st.index[st["name_number"] == num].tolist()
            if len(nn) > 1:
                return _ambiguous(r, query, nn, f"이름 속 번호가 {num}인 학생이 여러 명.")
            if len(nn) == 1:
                candidates.append(("name_number", nn[0]))
        if 1 <= num <= n:
            candidates.append(("list_position", num))
        if not candidates:
            return {"found": False, "query": query, "available_range": _available_range(r),
                    "basis": "학생 이름 속 번호와 명단 순서(1번부터) 모두에서 해당 번호를 찾지 못함."}

        match_type, pos = candidates[0]
        result = {"found": True, "query": query, "match_type": match_type, "student": _student_record(r, pos)}
        name_nums = set(st["name_number"].dropna().astype(int).tolist())
        numbering_is_sequential = name_nums == set(range(1, n + 1))
        if match_type != "list_position" and not numbering_is_sequential:
            alt = next((p for t, p in candidates[1:] if t == "list_position" and p != pos), None)
            if alt is not None:
                a = _student_record(r, alt)
                result["alternative"] = {"match_type": "list_position", "name": a["name"], "score": a["score"],
                                         "note": f"명단 {num}번째 학생 기준"}
        result["basis"] = {
            "student_id": "번호(학번) 열의 값과 일치.",
            "name_number": f"학생 이름 속 번호 {num}과 일치.",
            "list_position": f"OMR 명단의 {num}번째 학생(1번부터 셈).",
        }[match_type]
        if match_type == "list_position" and name_nums and not ordinal:
            result["basis"] += f" 이름 속 번호가 {num}인 학생은 없어서 명단 순서로 해석함(답변에 이 기준을 밝힐 것)."
        return result

    # 4) 이름 일부(토큰 단위)
    for token in _name_tokens(raw):
        partial = st.index[st["학생명"].str.contains(re.escape(token), na=False, regex=True)].tolist()
        if len(partial) == 1:
            return _found(r, query, "partial_name", partial[0])
        if len(partial) > 1:
            return _ambiguous(r, query, partial, "이름이 여러 학생과 겹침.")
    return {"found": False, "query": query, "available_range": _available_range(r), "basis": "해당 이름의 학생이 없음."}


def ranked_students(r: GradingResult, order: str = "bottom", n: int = 1) -> dict:
    o = str(order).lower()
    bottom = any(k in o for k in ("bottom", "low", "asc", "worst", "하위", "최저", "꼴", "저조", "못"))
    total = r.summary["n_students"]
    n = max(1, min(int(n or 1), total))
    scores = r.students["총점"]
    sorted_scores = scores.sort_values(ascending=bottom)
    boundary = int(sorted_scores.iloc[n - 1])
    mask = scores <= boundary if bottom else scores >= boundary
    picked = r.students[mask].sort_values("총점", ascending=bottom)
    rows = [{"rank": int(s["순위"]), "rank_of": total, "name": s["학생명"], "list_position": int(p),
             "score": int(s["총점"]), "correct_count": int(s["정답수"])} for p, s in picked.iterrows()]
    return {
        "order": "bottom(성적 낮은 순)" if bottom else "top(성적 높은 순)",
        "n_requested": n,
        "students": rows,
        "includes_ties": len(rows) > n,
        "basis": f"총점({r.summary['max_score']}점 만점) 기준. 경계 점수와 동점인 학생은 모두 포함. 순위는 공동 순위.",
    }


# ── 문항 조회 ────────────────────────────────────────────────

def _item_record(r: GradingResult, q: int) -> dict:
    it = r.items.loc[q]
    dist = {CIRCLED[k]: int(it[f"선택_{k}"]) for k in range(1, 6)}
    dist["무응답"] = int(it["무응답"])
    wrong_choices = {k: v for k, v in dist.items() if k not in (CIRCLED[int(it["정답"])], "무응답") and v > 0}
    most_wrong = max(wrong_choices, key=wrong_choices.get) if wrong_choices else None
    return {
        "number": int(q),
        "type": it["유형"] or "알 수 없음",
        "section": "듣기" if it["듣기"] else "독해",
        "answer": CIRCLED[int(it["정답"])],
        "points": int(it["배점"]),
        "correct_count": int(it["정답자수"]),
        "wrong_count": int(it["오답자수"]),
        "correct_rate": f"{fmt_num(float(it['정답률']))}%",
        "wrong_rate": f"{fmt_num(float(it['오답률']))}%",  # 요약 블록·엑셀과 같은 값(grader에서 한 번만 계산)
        "choice_distribution": dist,
        "most_chosen_wrong_answer": most_wrong,
    }


def most_wrong_questions(r: GradingResult, n: int = 3) -> dict:
    total_q = len(r.items)
    n = max(1, min(int(n or 3), total_q))
    ranked = r.items.reset_index().sort_values(["오답자수", "문항"], ascending=[False, True])
    top = ranked.head(n)
    boundary = int(top["오답자수"].iloc[-1])
    ties = ranked.iloc[n:]
    ties = ties[ties["오답자수"] == boundary]["문항"].astype(int).tolist()
    return {
        "items": [_item_record(r, int(q)) for q in top["문항"]],
        "ties_at_boundary_not_listed": ties,
        "basis": f"오답자 수 내림차순, 같으면 문항 번호 오름차순(엑셀 TOP3와 동일). 응시자 {r.summary['n_students']}명, 무응답은 오답.",
    }


def similar_question_source(r: GradingResult, question_number: int | None = None) -> dict:
    ranked = r.items.reset_index().sort_values(["오답자수", "문항"], ascending=[False, True])
    top_q, top_wrong = int(ranked["문항"].iloc[0]), int(ranked["오답자수"].iloc[0])
    # LLM이 오답 1위 번호를 직접 넘겨도 '가장 많이 틀린 문항' 문장이 유지되도록 같은 경로로 처리
    auto = question_number is None or str(question_number).strip() in ("", str(top_q))
    if auto:
        if top_wrong == 0:
            return {"found": False, "note": "모든 학생이 모든 문항을 맞혀 '가장 많이 틀린 문항'이 없음. 번호를 지정해 달라고 안내할 것."}
        q = top_q
        tied = [int(x) for x in ranked[ranked["오답자수"] == top_wrong]["문항"] if int(x) != q]
        reason = f"오답자 수 1위({top_wrong}/{r.summary['n_students']}명)"
    else:
        try:
            q = int(question_number)
        except (TypeError, ValueError):
            q = -1
        if q not in r.items.index:
            return {"found": False, "available_range": f"1~{len(r.items)}번"}
        tied, reason = [], "사용자가 지정한 문항"

    rec = _item_record(r, q)
    selected = {**rec, "selection_reason": reason, "tied_with": tied}
    desc = f"{q}번({rec['type']}, 오답률 {rec['wrong_rate']})"
    if auto:
        headline = f"가장 많이 틀린 문항은 {desc}입니다."
        if tied:
            headline += (f" {', '.join(f'{t}번' for t in tied)}도 오답자 수가 같아, "
                         f"번호가 가장 작은 {q}번을 기준으로 만들었습니다.")
    else:
        headline = f"요청하신 {desc} 문항과 같은 유형으로 만들었습니다."
    qt = r.questions.get(q)
    if qt is None:
        return {"found": True, "headline": headline, "selected": selected, "source": None,
                "note": "문제지 원문을 읽지 못함. 유형 정보만으로 같은 유형의 문제를 새로 만들 것."}

    selected.update({"is_listening": qt.is_listening, "set_range": list(qt.set_range) if qt.set_range else None})
    source = {
        "stem": qt.stem,
        "set_shared_passage": _clip(qt.set_passage) if qt.set_passage else None,
        "question_body_and_choices": _clip(qt.body) if qt.body else None,
    }
    if qt.is_listening:
        spec = "듣기 문항: 지문 대신 영어 [듣기 대본] 8~12줄 + 같은 형식의 발문·선택지 ①~⑤ + 정답 + 한국어 해설 2~3문장"
    elif qt.set_range:
        spec = "세트(장문) 문항: 짧은 새 지문 하나에 이 유형의 문항 1개만. 발문·①~⑤·정답·한국어 해설 2~3문장"
    else:
        spec = "새 영어 지문 120~170단어(원본과 다른 소재) + 같은 형식의 발문·①~⑤ + 정답 + 한국어 해설 2~3문장"
    if "빈칸" in rec["type"] or "요약" in rec["type"]:
        spec += ". 원문 PDF에서 빈칸 표시는 사라졌으므로 빈칸(______) 위치를 새로 설계"
    if any(k in rec["type"] for k in ("함축", "어법", "낱말", "지칭")):
        spec += ". 밑줄 대상은 지문 안에서 **굵게** 표시하고 발문에도 '밑줄 친(굵게 표시한)'으로 안내"
    if "어법" in rec["type"]:
        spec += (". 작성 절차: (1) 어법상 완전히 올바른 지문을 먼저 쓴다 (2) 서로 다른 문법 포인트 5곳"
                 "(예: 관계사, 분사, 수일치, to부정사/동명사, 병렬구조 — 수일치는 최대 1곳)에 ①~⑤를 붙인다"
                 " (3) 그중 정확히 1곳만 틀린 형태로 바꾸고 나머지 4곳과 표시하지 않은 부분은 올바른 형태로 둔다"
                 " (4) 출력 전 ①~⑤를 하나씩 다시 점검해 틀린 곳이 정확히 1곳인지 확인한다")
    if "낱말" in rec["type"]:
        spec += (". 작성 절차: 문맥에 맞는 낱말 5개에 표시한 뒤 그중 정확히 1개만 문맥과 반대되는 낱말로 바꾸고,"
                 " 출력 전 나머지 4개가 모두 문맥에 맞는지 다시 점검한다")
    stem = qt.stem.replace("윗글", "다음 글")  # 새 문제는 지문이 하나뿐이므로 '윗글' 대신 '다음 글'
    stem_rule = (f"발문은 원본 발문 '{stem}'과 같은 한국어 문장 구조로 쓰고, 표현만 새 지문에 맞게 바꿀 것"
                 if stem else "발문은 원본과 같은 형식의 한국어로")
    return {"found": True, "headline": headline, "selected": selected, "source": source,
            "spec": {"count": "정확히 1문항", "stem": stem_rule, "format": spec,
                     "must": ["발문은 한국어", "원문 문장 복사 금지", "정답이 하나만 성립"]}}


def group_stats(r: GradingResult, by: str = "section") -> dict:
    items = r.items
    if not r.questions:
        return {"by": str(by), "note": "문제지를 읽지 못해 영역 정보 없음(듣기/독해·유형 구분 불가).",
                "overall_correct_rate": f"{fmt_num(float(items['정답률'].mean()))}%"}
    if str(by).lower() in ("type", "유형"):
        g = items.groupby("유형")["정답률"].mean().round(1).sort_values()
        return {"by": "유형", "average_correct_rate": {k or "알 수 없음": f"{fmt_num(v)}%" for k, v in g.items()}}
    g = items.groupby(items["듣기"].map({True: "듣기", False: "독해"}))["정답률"].mean().round(1)
    counts = items["듣기"].map({True: "듣기", False: "독해"}).value_counts()
    return {"by": "영역", "average_correct_rate": {k: f"{fmt_num(v)}%" for k, v in g.items()},
            "question_counts": {k: int(v) for k, v in counts.items()}}


def score_summary(r: GradingResult) -> dict:
    s = r.summary
    return {
        "n_students": s["n_students"], "n_questions": s["n_questions"], "max_score": s["max_score"],
        "mean": s["mean"], "median": s["median"], "std": s["std"],
        "highest": s["highest"], "lowest": s["lowest"],
        "basis": "점수 = 맞힌 문항 배점의 합. 소수 둘째 자리 반올림. 표준편차는 표본 표준편차(n-1). 엑셀 '채점 결과' 시트와 같은 값.",
    }


# ── LangChain Tool 생성 ──────────────────────────────────────

def build_tools(r: GradingResult) -> list:
    @tool
    def get_score_summary() -> str:
        """전체 학생의 응시자 수, 평균, 중앙값, 표준편차, 최고점/최저점(동점 학생 포함)을 조회한다."""
        return _dump(score_summary(r))

    @tool
    def get_student_score(student: str) -> str:
        """특정 학생의 점수, 정답 수, 순위, 틀린 문항, 틀린 문항에서 고른 답을 조회한다.
        student에는 사용자가 쓴 표현을 그대로 넣는다. 예: "87번", "학생 87", "87번째", "김민준", "900학생"."""
        return _dump(find_student(r, student))

    @tool
    def get_ranked_students(order: str = "bottom", n: int = 1) -> str:
        """성적 순위로 학생을 조회한다. order="bottom"은 최하위·꼴등·가장 저조한 학생, order="top"은 최상위·1등.
        n은 인원 수(기본 1). 경계 점수의 동점자는 모두 포함된다."""
        return _dump(ranked_students(r, order, n))

    @tool
    def get_most_wrong_questions(n: int = 3) -> str:
        """학생들이 가장 많이 틀린 문항 n개(기본 3개)를 오답자 수, 오답률, 유형, 정답, 많이 고른 오답과 함께 조회한다."""
        return _dump(most_wrong_questions(r, n))

    @tool
    def prepare_similar_question(question_number: int | None = None) -> str:
        """유사(비슷한 유형) 문제를 만들기 위한 원본 문항 정보를 한 번에 가져온다.
        question_number를 비우면 가장 많이 틀린 문항(동점이면 번호가 가장 작은 문항)이 자동 선택된다.
        반환된 원문·유형·생성 규격을 참고해 새 문제 1개를 작성한다."""
        return _dump(similar_question_source(r, question_number))

    @tool
    def get_item_analysis(question_number: int) -> str:
        """특정 문항의 정답, 배점, 유형, 정답률, 선택지별 응답 분포를 조회한다."""
        try:
            q = int(question_number)
        except (TypeError, ValueError):
            q = -1
        if q not in r.items.index:
            return _dump({"found": False, "available_range": f"1~{len(r.items)}번"})
        return _dump({"found": True, **_item_record(r, q)})

    @tool
    def get_group_stats(by: str = "section") -> str:
        """영역별(by="section": 듣기/독해) 또는 유형별(by="type") 평균 정답률을 조회한다."""
        return _dump(group_stats(r, by))

    return [get_score_summary, get_student_score, get_ranked_students, get_most_wrong_questions,
            prepare_similar_question, get_item_analysis, get_group_stats]


TOOL_NAMES = ["get_score_summary", "get_student_score", "get_ranked_students", "get_most_wrong_questions",
              "prepare_similar_question", "get_item_analysis", "get_group_stats"]
