"""GradingResult → 엑셀(bytes). README p8 레이아웃: 한 시트에 개요(위) + 학생별 결과(아래)."""
from __future__ import annotations

import io

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from core.grader import fmt_num
from core.models import GradingResult

TITLE_FONT = Font(bold=True, size=13)
BOLD = Font(bold=True)
HEADER_FILL = PatternFill("solid", fgColor="DCE6F1")
LABEL_FILL = PatternFill("solid", fgColor="F2F2F2")
WRONG_FILL = PatternFill("solid", fgColor="F8CBAD")
WRONG_FONT = Font(color="9C0006", bold=True)
CENTER = Alignment(horizontal="center", vertical="center")
THIN = Side(style="thin", color="BFBFBF")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
CIRCLED = {1: "①", 2: "②", 3: "③", 4: "④", 5: "⑤"}


def _names(students: list[str]) -> str:
    return ", ".join(students)


POINT_FORMAT = 'General"점"'  # 숫자 셀로 저장(엑셀에서 계산 가능) + '65.2점'처럼 표시


def _summary_rows(result: GradingResult) -> list[tuple[str, object]]:
    s = result.summary
    top = s["top_wrong"]
    rows = [
        ("응시자 수", f"{s['n_students']}명"),
        ("문항 수 / 만점", f"{s['n_questions']}문항 / {s['max_score']}점"),
        ("평균", s["mean"]),
        ("중앙값", s["median"]),
        ("표준편차 (표본)", s["std"]),
        ("최고점", f"{s['highest']['score']}점 ({_names(s['highest']['students'])})"),
        ("최저점", f"{s['lowest']['score']}점 ({_names(s['lowest']['students'])})"),
        ("가장 많이 틀린 문항 TOP3", ", ".join(f"{t['number']}번" for t in top) or "없음 (전원이 모든 문항을 맞힘)"),
    ]
    for i, t in enumerate(top, 1):
        rows.append((f"  {i}위", f"{t['number']}번 - 오답 {t['wrong_count']}명 ({fmt_num(t['wrong_rate'])}%)"
                     + (f", {t['type']}" if t.get("type") else "")))
    if s["top_wrong_ties_excluded"]:
        rows.append(("  (3위 동점 문항)", ", ".join(f"{q}번" for q in s["top_wrong_ties_excluded"])))
    return rows


def _write_main_sheet(ws, result: GradingResult) -> None:
    q_nums = list(result.correct.columns)
    ws.title = "채점 결과"

    ws.cell(row=1, column=1, value="전체 학생 성적 개요").font = TITLE_FONT
    row = 2
    for label, value in _summary_rows(result):
        a = ws.cell(row=row, column=1, value=label)
        a.font, a.fill, a.border = BOLD, LABEL_FILL, BOX
        b = ws.cell(row=row, column=2, value=value)
        b.border = BOX
        if isinstance(value, float):
            b.number_format = POINT_FORMAT
            b.alignment = Alignment(horizontal="left")
        ws.merge_cells(start_row=row, start_column=2, end_row=row, end_column=8)
        row += 1

    row += 1
    ws.cell(row=row, column=1, value="학생별 점수 및 문항별 정답 여부 (O: 정답, X: 오답)").font = TITLE_FONT
    row += 1
    header_row = row
    has_id = result.students["학번"].notna().any()
    headers = ["번호", "학생명"] + (["학번"] if has_id else []) + ["총점", "정답수", "순위"] + [f"{q}번" for q in q_nums]
    for col, h in enumerate(headers, 1):
        c = ws.cell(row=row, column=col, value=h)
        c.font, c.fill, c.alignment, c.border = BOLD, HEADER_FILL, CENTER, BOX

    first_q_col = len(headers) - len(q_nums) + 1
    for pos, st in result.students.iterrows():
        row += 1
        values = [int(pos), st["학생명"]] + ([st["학번"]] if has_id else []) + [int(st["총점"]), int(st["정답수"]), int(st["순위"])]
        for col, v in enumerate(values, 1):
            c = ws.cell(row=row, column=col, value=v)
            c.alignment, c.border = CENTER, BOX
        for j, q in enumerate(q_nums):
            ok = bool(result.correct.at[pos, q])
            c = ws.cell(row=row, column=first_q_col + j, value="O" if ok else "X")
            c.alignment, c.border = CENTER, BOX
            if not ok:
                c.fill, c.font = WRONG_FILL, WRONG_FONT

    ws.freeze_panes = ws.cell(row=header_row + 1, column=3)
    ws.column_dimensions["A"].width = 24
    ws.column_dimensions["B"].width = 14
    for col in range(3, first_q_col):
        ws.column_dimensions[get_column_letter(col)].width = 9
    for col in range(first_q_col, first_q_col + len(q_nums)):
        ws.column_dimensions[get_column_letter(col)].width = 5.5


def _write_item_sheet(ws, result: GradingResult) -> None:
    ws.title = "문항 분석"
    items = result.items
    headers = ["문항", "유형", "정답", "배점", "정답률(%)", "정답자 수", "오답자 수", "①", "②", "③", "④", "⑤", "무응답"]
    for col, h in enumerate(headers, 1):
        c = ws.cell(row=1, column=col, value=h)
        c.font, c.fill, c.alignment, c.border = BOLD, HEADER_FILL, CENTER, BOX
    for i, (q, it) in enumerate(items.iterrows(), start=2):
        values = [int(q), it["유형"] or "", CIRCLED[int(it["정답"])], int(it["배점"]), float(it["정답률"]),
                  int(it["정답자수"]), int(it["오답자수"])] + [int(it[f"선택_{k}"]) for k in range(1, 6)] + [int(it["무응답"])]
        for col, v in enumerate(values, 1):
            c = ws.cell(row=i, column=col, value=v)
            c.alignment, c.border = CENTER, BOX
    ws.freeze_panes = "B2"
    ws.column_dimensions["B"].width = 26


def _neutralize_formulas(wb: Workbook) -> None:
    """학생명 등에서 '='로 시작하는 문자열이 엑셀 수식으로 저장되지 않게 한다(수식 주입 방지)."""
    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for c in row:
                if c.data_type == "f":
                    c.data_type = "s"


def build_excel(result: GradingResult) -> bytes:
    wb = Workbook()
    _write_main_sheet(wb.active, result)
    _write_item_sheet(wb.create_sheet(), result)
    _neutralize_formulas(wb)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
