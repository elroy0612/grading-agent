"""테스트용 OMR 데이터 생성기 (시드 고정, 문항별 난이도 + 학생 능력 모델).

프로젝트 tests/ 로 옮겨도 동작하도록 프로젝트 루트를 자동 탐색한다.
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import numpy as np
from openpyxl import Workbook

sys.dont_write_bytecode = True


def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "core" / "grader.py").exists():
            return p
    env = os.environ.get("GRADER_PROJECT_ROOT")
    if env:
        return Path(env)
    raise RuntimeError("프로젝트 루트(core/grader.py)를 찾지 못했습니다. GRADER_PROJECT_ROOT를 지정하세요.")


ROOT = find_project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ANSWER_PDF = ROOT / "2026학년도-대학수학능력시험-영어-정답.pdf"
PAPER_PDF = ROOT / "2026학년도-대학수학능력시험-영어-문제.pdf"
SAMPLE_XLSX = ROOT / "sample_data.xlsx"
SHEET = "OMR_응답"

# ── 학생명 형식 ─────────────────────────────────────────────
SURNAMES = ["김", "이", "박", "최", "정", "강", "조", "윤", "장", "임", "한", "오", "서", "신", "권", "황", "안", "송", "류", "전"]
GIVEN = ["민준", "서연", "도윤", "하은", "시우", "지호", "서윤", "하준", "지우", "수아", "예준", "지민", "주원", "채원",
         "건우", "다은", "우진", "예린", "선우", "유나", "현우", "소율", "지훈", "하린", "준서", "윤서", "연우", "가은",
         "은우", "수빈"]


def korean_names(n: int, seed: int = 0) -> list[str]:
    combos = [s + g for s in SURNAMES for g in GIVEN]  # 600개, 모두 고유
    rng = np.random.default_rng(1000 + seed)
    idx = rng.permutation(len(combos))[:n]
    return [combos[i] for i in idx]


NAME_FORMATS = {
    "n학생": lambda n, seed: [f"{i}학생" for i in range(1, n + 1)],
    "00n학생": lambda n, seed: [f"{i:03d}학생" for i in range(1, n + 1)],
    "학생n": lambda n, seed: [f"학생{i}" for i in range(1, n + 1)],
    "900학생": lambda n, seed: [f"{900 + i}학생" for i in range(n)],
    "실명": korean_names,
    "0학생": lambda n, seed: [f"{i}학생" for i in range(n)],
}


# ── 응답 생성 ───────────────────────────────────────────────
def gen_responses(key: dict[int, int], n: int, seed: int, blank_rate: float = 0.01) -> np.ndarray:
    """(n, len(key)) 정수 행렬. 0 = 무응답, 1~5 = 마킹. 열 순서 = sorted(key)."""
    rng = np.random.default_rng(seed)
    qs = sorted(key)
    p_item = rng.uniform(0.30, 0.95, len(qs))              # 문항별 기본 정답률(난이도)
    ability = rng.normal(0.0, 1.0, n)                       # 학생 능력
    logit = np.log(p_item / (1 - p_item))[None, :] + 1.3 * ability[:, None]
    p = 1.0 / (1.0 + np.exp(-logit))
    correct = rng.random((n, len(qs))) < p
    resp = np.zeros((n, len(qs)), dtype=int)
    for j, q in enumerate(qs):
        k = key[q]
        distractors = [c for c in range(1, 6) if c != k]
        w = rng.dirichlet(np.full(4, 0.8))                  # 매력적인 오답지 편중
        wrong = rng.choice(distractors, size=n, p=w)
        resp[:, j] = np.where(correct[:, j], k, wrong)
    if blank_rate:
        resp[rng.random((n, len(qs))) < blank_rate] = 0
    return resp


# ── 파일 작성 ───────────────────────────────────────────────
def omr_rows(names, resp: np.ndarray, qnums: list[int], header_fmt="Q{:02d}", name_header="학생명",
             blank=None) -> list[list]:
    rows = [[name_header] + [header_fmt.format(q) for q in qnums]]
    for name, r in zip(names, resp):
        rows.append([name] + [int(v) if v else blank for v in r])
    return rows


def write_xlsx(sheets: dict[str, list[list]]) -> bytes:
    """시트명 → 행 목록. '='로 시작하는 문자열은 수식이 아닌 문자열 셀로 저장한다."""
    wb = Workbook()
    for i, (title, rows) in enumerate(sheets.items()):
        ws = wb.active if i == 0 else wb.create_sheet()
        ws.title = title
        for row in rows:
            ws.append(row)
        for row in ws.iter_rows():
            for c in row:
                if isinstance(c.value, str) and c.value.startswith("="):
                    c.data_type = "s"
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def write_csv(rows: list[list], encoding: str = "utf-8-sig") -> bytes:
    import csv

    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    for r in rows:
        w.writerow(["" if v is None else v for v in r])
    return buf.getvalue().encode(encoding)


def make_dataset(fmt: str, n: int, seed: int, key: dict[int, int], blank_rate: float = 0.01):
    names = NAME_FORMATS[fmt](n, seed)
    resp = gen_responses(key, n, seed, blank_rate)
    qnums = sorted(key)
    data = write_xlsx({SHEET: omr_rows(names, resp, qnums)})
    return names, resp, qnums, data
