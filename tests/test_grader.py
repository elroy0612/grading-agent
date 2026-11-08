from openpyxl import load_workbook
import io

from core.exporter import build_excel


def test_summary_matches_hand_check(result):
    s = result.summary
    assert s["n_students"] == 10 and s["max_score"] == 100
    assert (s["mean"], s["median"], s["std"]) == (65.2, 65.5, 20.64)
    assert s["highest"] == {"score": 92, "students": ["901학생"]}
    assert s["lowest"] == {"score": 31, "students": ["909학생"]}
    assert [t["number"] for t in s["top_wrong"]] == [21, 33, 42]
    assert [t["wrong_count"] for t in s["top_wrong"]] == [9, 8, 7]


def test_scores_consistent(result):
    st = result.students
    points = result.items["배점"]
    recomputed = result.correct.astype(int).mul(points, axis="columns").sum(axis=1)
    assert (recomputed == st["총점"]).all()
    assert st.loc[st["총점"].idxmax(), "순위"] == 1


def test_excel_layout(result):
    wb = load_workbook(io.BytesIO(build_excel(result)))
    assert wb.sheetnames == ["채점 결과", "문항 분석"]
    ws = wb["채점 결과"]
    labels = [ws.cell(row=r, column=1).value for r in range(1, 12)]
    for needed in ("평균", "중앙값", "표준편차 (표본)", "가장 많이 틀린 문항 TOP3"):
        assert needed in labels
    mean_cell = ws.cell(row=labels.index("평균") + 1, column=2)
    assert mean_cell.value == 65.2 and "점" in mean_cell.number_format  # 숫자 셀(엑셀에서 계산 가능)
    header_row = next(r for r in range(1, 30) if ws.cell(row=r, column=1).value == "번호")
    assert ws.cell(row=header_row, column=2).value == "학생명"
    assert ws.cell(row=header_row, column=6).value == "1번"
    assert ws.cell(row=header_row + 1, column=2).value == "900학생"
    assert ws.cell(row=header_row + 10, column=2).value == "909학생"
    assert {ws.cell(row=header_row + 1, column=c).value for c in range(6, 51)} <= {"O", "X"}
