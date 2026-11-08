import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.grader import grade  # noqa: E402
from core.parsers.answer_key import parse_answer_key  # noqa: E402
from core.parsers.omr import parse_omr  # noqa: E402
from core.parsers.question_paper import parse_question_paper  # noqa: E402

ANSWER_PDF = ROOT / "2026학년도-대학수학능력시험-영어-정답.pdf"
PAPER_PDF = ROOT / "2026학년도-대학수학능력시험-영어-문제.pdf"
OMR_XLSX = ROOT / "sample_data.xlsx"


@pytest.fixture(scope="session")
def answer_key():
    return parse_answer_key(ANSWER_PDF.read_bytes())


@pytest.fixture(scope="session")
def omr():
    return parse_omr(OMR_XLSX.read_bytes(), OMR_XLSX.name)


@pytest.fixture(scope="session")
def questions():
    return parse_question_paper(PAPER_PDF.read_bytes())


@pytest.fixture(scope="session")
def result(answer_key, omr, questions):
    return grade(answer_key, omr, questions)
