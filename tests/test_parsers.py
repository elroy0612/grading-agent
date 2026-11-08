"""파서 구조 검증. 정답 리터럴은 테스트에도 두지 않는다(하드코딩 오해 방지)."""
import io

import pandas as pd

from core.parsers.classify import ANSWER, OMR, PAPER, classify_file
from core.parsers.omr import parse_omr
from tests.conftest import ANSWER_PDF, OMR_XLSX, PAPER_PDF


def test_answer_key_structure(answer_key):
    """구조만 검증한다(정답·배점 값을 테스트에도 적지 않음 → 하드코딩 오해 방지)."""
    assert answer_key.numbers == list(range(1, len(answer_key.items) + 1))
    assert len(answer_key.items) == 45
    assert all(1 <= i.answer <= 5 for i in answer_key.items.values())
    assert all(isinstance(i.points, int) and i.points > 0 for i in answer_key.items.values())
    assert answer_key.total_points == sum(i.points for i in answer_key.items.values())
    assert not answer_key.warnings


def test_omr_structure(omr):
    assert len(omr.names) == 10
    assert omr.responses.shape == (10, 45)
    assert list(omr.responses.index) == list(range(1, 11))
    assert omr.responses.isin([0, 1, 2, 3, 4, 5]).all().all()


def test_omr_variants_blank_circled_csv():
    df = pd.DataFrame({"이름": ["홍길동", "김철수"], "1번": ["③", None], "2번": [2, "x"], "문항3": [5, 1],
                       "Q4": ["6", "3.0"], "5": [" 4 ", "1,3"]})
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    omr = parse_omr(buf.getvalue().encode("cp949"), "omr.csv")
    assert omr.names == ["홍길동", "김철수"]
    assert omr.responses.loc[1].tolist() == [3, 2, 5, 0, 4]
    assert omr.responses.loc[2].tolist() == [0, 0, 1, 3, 0]
    assert any("무응답" in w for w in omr.warnings)


def test_question_paper_sets_and_noise(questions):
    assert sorted(questions) == list(range(1, 46))
    sets = {q.set_range for q in questions.values() if q.set_range}
    assert sets == {(16, 17), (31, 34), (36, 37), (38, 39), (41, 42), (43, 45)}
    # 세트 공통 지문이 소속 문항에 연결되고 앞 문항에 섞이지 않는다
    assert len(questions[41].set_passage) > 500 and len(questions[43].set_passage) > 500
    assert "[41" not in questions[40].body and "Mia" not in questions[42].body
    text = " ".join(q.stem + q.body + q.set_passage for q in questions.values())
    for noise in ("저작권", "홀수형", "확인 사항", "듣기 문제가 끝"):
        assert noise not in text


def test_question_types(questions):
    assert all(questions[n].is_listening for n in range(1, 18))
    assert not any(questions[n].is_listening for n in range(18, 46))
    assert all(questions[n].qtype == "빈칸 추론" for n in range(31, 35))
    assert questions[21].qtype == "함축 의미 추론"
    assert questions[29].qtype == "어법"
    assert questions[36].qtype == questions[37].qtype == "글의 순서"
    assert questions[38].qtype == "문장 삽입"
    assert questions[40].qtype == "요약문 완성"
    assert questions[42].qtype == "어휘(낱말 쓰임)"
    assert questions[44].qtype == "지칭 대상"
    assert all(q.qtype != "기타" for q in questions.values())


def test_classify():
    assert classify_file(ANSWER_PDF.name, ANSWER_PDF.read_bytes()) == ANSWER
    assert classify_file(PAPER_PDF.name, PAPER_PDF.read_bytes()) == PAPER
    assert classify_file(OMR_XLSX.name, b"") == OMR
    # 파일명이 단서가 없어도 내용으로 구분
    assert classify_file("a.pdf", ANSWER_PDF.read_bytes()) == ANSWER
    assert classify_file("b.pdf", PAPER_PDF.read_bytes()) == PAPER
