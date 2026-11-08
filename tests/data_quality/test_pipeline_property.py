"""120명 규모 속성 기반 테스트: 파서 → 채점 → 엑셀 → 조회 Tool 로직.

학생명 6가지 형식 × 시드 3개(N=120) + 인원 변형(N=1,2,3,500)에 대해
앱 결과를 oracle.py(numpy/순수 Python 독립 계산)와 비교한다. LLM 호출 없음.
[tests/로 옮길 후보] omr_gen.py, oracle.py와 함께 옮기면 된다.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402
from openpyxl import load_workbook  # noqa: E402

import oracle  # noqa: E402
from omr_gen import NAME_FORMATS, SHEET, make_dataset, omr_rows, write_xlsx  # noqa: E402
from agent.tools import find_student, most_wrong_questions, ranked_students, score_summary  # noqa: E402
from core.exporter import build_excel  # noqa: E402
from core.grader import grade  # noqa: E402
from core.parsers.omr import parse_omr  # noqa: E402

MAIN = [(fmt, 120, seed) for fmt in NAME_FORMATS for seed in (1, 2, 3)]
SIZES = [("n학생", n, 7) for n in (1, 2, 3, 500)]
TIED = [("900학생", 120, 99), ("실명", 120, 99)]   # seed 99: 최고점 2명·최저점 3명 동점을 강제
SIZES += TIED
DATASETS = MAIN + SIZES
ASCII = {"n학생": "nHS", "00n학생": "00nHS", "학생n": "HSn", "900학생": "900HS", "실명": "realname", "0학생": "0HS"}
IDS = [f"{ASCII[f]}-N{n}-s{s}" for f, n, s in DATASETS]

_CACHE: dict = {}


def pipeline(ds, answer_key, questions):
    if ds not in _CACHE:
        fmt, n, seed = ds
        key = {q: it.answer for q, it in answer_key.items.items()}
        names, resp, qnums, data = make_dataset(fmt, n, seed, key)
        if seed == 99:
            resp = resp.copy()
            resp[[0, 77]] = [key[q] for q in qnums]                 # 만점 2명
            resp[[10, 50, 119]] = [key[q] % 5 + 1 for q in qnums]   # 0점 3명
            data = write_xlsx({SHEET: omr_rows(names, resp, qnums)})
        omr = parse_omr(data, f"omr_{fmt}_{n}.xlsx")
        result = grade(answer_key, omr, questions)
        xbytes = build_excel(result)
        exp = oracle.expected([oracle.clean_name(x) for x in names], resp, qnums,
                              {q: (it.answer, it.points) for q, it in answer_key.items.items()})
        _CACHE[ds] = dict(names=[oracle.clean_name(x) for x in names], resp=resp, qnums=qnums, omr=omr,
                          result=result, xbytes=xbytes, exp=exp)
    return _CACHE[ds]


@pytest.fixture(params=DATASETS, ids=IDS)
def P(request, answer_key, questions):
    return pipeline(request.param, answer_key, questions) | {"ds": request.param}


# ── 파서 ────────────────────────────────────────────────────
def test_parse_roundtrip(P):
    omr = P["omr"]
    assert omr.names == P["names"]
    assert list(omr.responses.index) == list(range(1, P["exp"]["n"] + 1))
    assert list(omr.responses.columns) == P["qnums"]
    assert (omr.responses.to_numpy() == P["resp"]).all()
    blanks = P["exp"]["blank_total"]
    if blanks:
        assert any(f"{blanks}건" in w for w in omr.warnings), omr.warnings
    else:
        assert not omr.warnings


# ── 채점 ────────────────────────────────────────────────────
def test_scores_counts_ranks(P):
    st, e = P["result"].students, P["exp"]
    assert st["학생명"].tolist() == P["names"]
    assert st["총점"].tolist() == e["score"]
    assert st["정답수"].tolist() == e["ncorrect"]
    assert st["순위"].tolist() == e["rank"]
    assert (P["result"].correct.to_numpy() == e["correct"]).all()


def test_median_std_and_mean_precision(P):
    s, e = P["result"].summary, P["exp"]
    assert s["n_students"] == e["n"] and s["max_score"] == e["max_score"] == 100
    assert s["median"] == e["median"]
    # 반올림 결과는 참값과 반 단위(0.005) 이내여야 하고, 소수 둘째 자리까지만
    assert abs(s["mean"] - float(e["mean_exact"])) <= 0.005 + 1e-9
    assert abs(s["std"] - e["std_true"]) <= 0.005 + 1e-9
    assert round(s["mean"], 2) == s["mean"] and round(s["std"], 2) == s["std"]


def test_mean_rounding_is_half_up(P):
    """'소수 둘째 자리 반올림' = 사사오입(엑셀 ROUND와 동일)인지."""
    assert P["result"].summary["mean"] == P["exp"]["mean_half_up"], (
        f"exact={P['exp']['mean_exact']} = {float(P['exp']['mean_exact'])!r}")


def test_top_wrong_and_ties(P):
    s, e = P["result"].summary, P["exp"]
    assert [(t["number"], t["wrong_count"]) for t in s["top_wrong"]] == e["top3"]
    for t, rate in zip(s["top_wrong"], e["top3_rates"]):
        assert abs(t["wrong_rate"] - rate) <= 0.005 + 1e-9
    assert s["top_wrong_ties_excluded"] == e["ties_excluded"]


def test_highest_lowest_ties(P):
    s, e = P["result"].summary, P["exp"]
    assert (s["highest"]["score"], s["highest"]["students"]) == e["highest"]
    assert (s["lowest"]["score"], s["lowest"]["students"]) == e["lowest"]


def test_items_table(P):
    items, e = P["result"].items, P["exp"]
    n = e["n"]
    for q in e["common"]:
        it = items.loc[q]
        assert int(it["정답자수"]) == e["correct_cnt"][q] and int(it["오답자수"]) == e["wrong_cnt"][q]
        assert [int(it[f"선택_{c}"]) for c in range(1, 6)] + [int(it["무응답"])] == \
               [e["dist"][q][c] for c in range(1, 6)] + [e["dist"][q][0]]
        assert abs(float(it["정답률"]) - e["correct_cnt"][q] * 100 / n) <= 0.05 + 1e-9


# ── 엑셀 ────────────────────────────────────────────────────
def _read_excel(xbytes):
    wb = load_workbook(io.BytesIO(xbytes))
    ws = wb["채점 결과"]
    summary, r = {}, 2
    while ws.cell(r, 1).value is not None:
        summary[ws.cell(r, 1).value] = ws.cell(r, 2).value
        r += 1
    header_row = next(row for row in range(r, r + 5) if ws.cell(row, 1).value == "번호")
    headers = [c.value for c in ws[header_row]]
    rows = list(ws.iter_rows(min_row=header_row + 1, max_row=ws.max_row))
    return wb, summary, headers, rows


def test_excel_overview(P):
    wb, sm, _, _ = _read_excel(P["xbytes"])
    assert wb.sheetnames == ["채점 결과", "문항 분석"]
    e, s = P["exp"], P["result"].summary
    assert sm["응시자 수"] == f"{e['n']}명"
    assert sm["문항 수 / 만점"] == f"{len(e['common'])}문항 / {e['max_score']}점"
    assert sm["평균"] == s["mean"]                      # 숫자 셀, 엑셀 = 채팅 summary 와 같은 값
    assert abs(sm["평균"] - float(e["mean_exact"])) <= 0.005 + 1e-9
    assert oracle.fmt(sm["중앙값"]) == oracle.fmt(e["median"])
    assert sm["표준편차 (표본)"] == s["std"]
    hi, lo = e["highest"], e["lowest"]
    assert sm["최고점"] == f"{hi[0]}점 ({', '.join(hi[1])})"
    assert sm["최저점"] == f"{lo[0]}점 ({', '.join(lo[1])})"
    assert sm["가장 많이 틀린 문항 TOP3"] == ", ".join(f"{q}번" for q, _ in e["top3"])
    for i, (q, w) in enumerate(e["top3"], 1):
        assert sm[f"  {i}위"].startswith(f"{q}번 - 오답 {w}명 (")
    if e["ties_excluded"]:
        assert sm["  (3위 동점 문항)"] == ", ".join(f"{q}번" for q in e["ties_excluded"])
    else:
        assert "  (3위 동점 문항)" not in sm


def test_excel_student_table(P):
    _, _, headers, rows = _read_excel(P["xbytes"])
    e = P["exp"]
    assert headers[:5] == ["번호", "학생명", "총점", "정답수", "순위"]
    assert headers[5:] == [f"{q}번" for q in e["common"]]
    assert len(rows) == e["n"]
    for i, row in enumerate(rows):
        v = [c.value for c in row]
        assert v[:5] == [i + 1, P["names"][i], e["score"][i], e["ncorrect"][i], e["rank"][i]], i
        assert v[5:] == ["O" if ok else "X" for ok in e["correct"][i]], i
        assert all(c.data_type != "f" for c in row)


def test_excel_item_sheet(P):
    wb = load_workbook(io.BytesIO(P["xbytes"]))
    ws = wb["문항 분석"]
    e = P["exp"]
    head = [c.value for c in ws[1]]
    assert head == ["문항", "유형", "정답", "배점", "정답률(%)", "정답자 수", "오답자 수", "①", "②", "③", "④", "⑤", "무응답"]
    rows = [[c.value for c in r] for r in ws.iter_rows(min_row=2)]
    assert [r[0] for r in rows] == e["common"]
    for r in rows:
        q = r[0]
        assert sum(r[7:13]) == e["n"]
        assert r[7:12] == [e["dist"][q][c] for c in range(1, 6)] and r[12] == e["dist"][q][0]
        assert (r[5], r[6]) == (e["correct_cnt"][q], e["wrong_cnt"][q])


# ── 조회 Tool 로직 ─────────────────────────────────────────
def test_score_summary_and_most_wrong_tool(P):
    r, e = P["result"], P["exp"]
    ss = score_summary(r)
    assert ss["highest"]["students"] == e["highest"][1] and ss["lowest"]["students"] == e["lowest"][1]
    mw = most_wrong_questions(r, 3)
    assert [(i["number"], i["wrong_count"]) for i in mw["items"]] == e["top3"]
    assert mw["ties_at_boundary_not_listed"] == e["ties_excluded"]


@pytest.mark.parametrize("order,canon", [("bottom", "bottom"), ("top", "top"), ("lowest", "bottom"),
                                         ("최하위", "bottom"), ("1등", "top")])
@pytest.mark.parametrize("n", [1, 3])
def test_ranked_students(P, order, canon, n):
    r, e = P["result"], P["exp"]
    out = ranked_students(r, order, n)
    want = oracle.ranked(e["score"], P["names"], canon, n)
    got = [s["name"] for s in out["students"]]
    assert sorted(got) == sorted(want)
    assert out["includes_ties"] == (len(want) > min(n, e["n"]))
    scores = [s["score"] for s in out["students"]]
    assert scores == sorted(scores, reverse=(canon == "top"))
    for s in out["students"]:
        assert s["rank"] == e["rank"][s["list_position"] - 1]


def test_ranked_students_clamp(P):
    out = ranked_students(P["result"], "top", P["exp"]["n"] + 5)
    assert len(out["students"]) == P["exp"]["n"]


# ── find_student 매칭 사다리 ────────────────────────────────
def _number_expectation(fmt: str, names: list[str], k: int):
    """'k번 학생' 류 질의의 (기대 이름, 기대 alternative 이름 또는 None). 없으면 (None, None)."""
    n = len(names)
    if fmt in ("n학생", "00n학생", "학생n", "실명"):
        return (names[k - 1], None) if 1 <= k <= n else (None, None)
    if fmt == "900학생":
        if 900 <= k < 900 + n:
            alt = None  # 명단 k번째는 존재하지 않음(k > N)
            return names[k - 900], alt
        return (names[k - 1], None) if 1 <= k <= n else (None, None)
    if fmt == "0학생":
        if 0 <= k <= n - 1:
            return names[k], (names[k - 1] if 1 <= k <= n else None)
        return (names[k - 1], None) if 1 <= k <= n else (None, None)
    raise AssertionError(fmt)


def _probe_numbers(fmt, n):
    ks = {0, 1, 2, 3, 9, 10, 11, 19, 20, 99, 100, 101, 119, 120, 121, n, n + 1, 999}
    if fmt == "900학생":
        ks |= {899, 900, 901, 905, 999, 1000, 1019, 1020, 5000}
    return sorted(ks)


def _check(r, query, want, alt, failures, match_types=None):
    out = find_student(r, query)
    got = out["student"]["name"] if out.get("found") else None
    if want is None:
        if out.get("found") or "available_range" not in out:
            failures.append((query, "not found + available_range", got or out))
        return
    if got != want:
        failures.append((query, want, got if got else {k: out.get(k) for k in ("found", "candidates")}))
        return
    if match_types and out["match_type"] not in match_types:
        failures.append((query, f"match_type∈{match_types}", out["match_type"]))
    has_alt = "alternative" in out
    if alt is None and has_alt:
        failures.append((query, "alternative 없음", out["alternative"]))
    if alt is not None and (not has_alt or out["alternative"]["name"] != alt):
        failures.append((query, f"alternative={alt}", out.get("alternative")))


NUMBER_TEMPLATES = {
    "k번 학생": "{k}번 학생",
    "학생 k": "학생 {k}",
    "k": "{k}",
    "k번": "{k}번",
    "문장형 몇 점": "{k}번 학생 몇 점이야?",
    "문장형 알려줘": "{k}번 학생 점수 알려줘",
}


@pytest.mark.parametrize("template", list(NUMBER_TEMPLATES))
def test_find_student_number_queries(P, template):
    fmt, n, _ = P["ds"]
    failures = []
    for k in _probe_numbers(fmt, n):
        want, alt = _number_expectation(fmt, P["names"], k)
        _check(P["result"], NUMBER_TEMPLATES[template].format(k=k), want, alt, failures)
    assert not failures, failures[:10]


SENTENCE_VARIANTS = ["{k}번 학생 점수 줘", "{k}번 학생 어때?", "{k}번 학생은 몇 점 받았어?", "{k}번 학생 성적 좀 알려줘",
                     "{k}번 학생 등수는?", "학생 {k}번 점수 확인해줘", "{k}번 학생 틀린 문제 뭐야?", "{k}번 학생의 총점",
                     "{k}번은 몇 점?", "No.{k} 학생", "#{k}", "{k} 번 학생 결과 보여줘"]


def test_find_student_number_sentence_variants(P):
    """다양한 문장형 숫자 질의 — 잔여 단어가 실명에 우연히 포함되어 이름 검색으로 새는지 확인."""
    fmt, n, _ = P["ds"]
    failures = []
    for k in sorted({1, 2, 10, 57, n} & set(range(1, n + 1))):
        want, alt = _number_expectation(fmt, P["names"], k)
        for t in SENTENCE_VARIANTS:
            _check(P["result"], t.format(k=k), want, alt, failures)
    assert not failures, failures[:10]


def test_find_student_ordinal(P):
    fmt, n, _ = P["ds"]
    failures = []
    for k in _probe_numbers(fmt, n):
        want = P["names"][k - 1] if 1 <= k <= n else None
        _check(P["result"], f"{k}번째 학생", want, None, failures, match_types={"list_position"})
    assert not failures, failures[:10]


def test_find_student_exact_names(P):
    failures = []
    for name in P["names"]:
        _check(P["result"], name, name, None, failures, match_types={"exact_name"})
    assert not failures, failures[:10]


def test_find_student_name_with_suffix(P):
    """실명 형식: '김민준 학생', '김민준 학생 몇 점이야?' 처럼 말해도 찾는지."""
    fmt = P["ds"][0]
    if fmt != "실명":
        pytest.skip("실명 형식 전용")
    failures = []
    for name in P["names"]:
        _check(P["result"], f"{name} 학생", name, None, failures)
    assert not failures, f"{len(failures)}/{len(P['names'])}건 실패: {failures[:8]}"


def test_find_student_name_sentence(P):
    fmt = P["ds"][0]
    if fmt != "실명":
        pytest.skip("실명 형식 전용")
    failures = []
    for name in P["names"]:
        _check(P["result"], f"{name} 학생 몇 점이야?", name, None, failures)
    assert not failures, f"{len(failures)}/{len(P['names'])}건 실패: {failures[:5]}"
