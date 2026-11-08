# OO 학원 평가 도우미

모의고사 **문제지 PDF · OMR 인식 결과 · 정답 PDF**를 올리면 자동으로 채점하고, 결과를 엑셀로 추출하며,
채점 결과에 대해 질문하거나 오답이 많은 유형의 유사 문제를 만들어 주는 AI 에이전트 웹 서비스입니다.

| 항목 | 내용 |
|---|---|
| 학번 | `202235854` |
| 이름 | `권현석` |
| 배포 URL | `https://(배포 후 입력).streamlit.app` |

## 주요 기능

- **페이지 1 – 랜딩**: 서비스 소개, 가천대 로고, 상단 페이지 이동 버튼
- **페이지 2 – 채점 · 분석**
  - 사이드바에 파일 3개를 한 번에 Drag & Drop → 파일 종류 자동 인식(수동 변경 가능)
  - **분석 시작**: 정답 PDF에서 문항 번호·정답·배점을 추출해 자동 채점 (정답·배점 하드코딩 없음)
  - **엑셀 추출**: 전체 개요(평균·중앙값·표준편차·최다 오답 TOP3) + 학생별 점수·문항별 O/X, 문항 분석 시트
  - **채팅**: 채점이 끝나면 활성화. 세션 내 대화 기억, 평균/중앙값·학생별 점수·최하위 학생·유사 문제 생성,
    데이터에 없는 질문 회피, 프롬프트 인젝션 방어

## 기술 스택

Streamlit 1.67 · pandas · pdfplumber · openpyxl · LangChain 1.x (`create_agent`) · LangGraph · OpenAI `gpt-5.4-mini`

설계 상세는 [ARCHITECTURE.md](ARCHITECTURE.md)를 참고하세요.

## 로컬 실행

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env            # OPENAI_API_KEY 입력
streamlit run app.py
```

## 배포 (Streamlit Community Cloud)

1. 이 저장소를 GitHub에 push (`.env`는 `.gitignore`로 제외됨)
2. share.streamlit.io → New app → Main file `app.py`, **Python 3.12**
3. Advanced settings → Secrets
   ```toml
   OPENAI_API_KEY = "sk-..."
   OPENAI_MODEL = "gpt-5.4-mini"
   OPENAI_REASONING_EFFORT = "low"
   OPENAI_GENERATION_EFFORT = "medium"
   ```

## 테스트

```bash
python -m pytest -q                       # 전체 916개 (약 2.5분, API 키 불필요)
python -m pytest -q --ignore=tests/data_quality   # 빠른 테스트만 (약 20초)
python -m tests.chat_regression           # 실제 OpenAI 호출로 채팅 평가 시나리오 + 응답 시간 측정
```

- `tests/test_parsers.py`, `test_grader.py`: 파서·채점·엑셀 레이아웃
- `tests/test_agent_logic.py`, `test_robustness.py`: 학생 매칭, Tool 퍼징, 인젝션 가드, 동적 채점 증명, 엣지 케이스
- `tests/test_agent_stream.py`: 가짜 LLM으로 화면 출력 필터·오류·시간 제한·출력 유출 필터 검증
- `tests/data_quality/`: 학생명 6개 형식 × 120명 속성 테스트 + OMR 엣지 케이스(앱과 독립적인 재계산과 비교)

## 프로젝트 구조

```
app.py                  # 엔트리포인트 (st.navigation)
config.py               # 설정·비밀키 (.env → st.secrets)
views/                  # home.py(랜딩), analysis.py(업로드·채점·엑셀·채팅)
core/parsers/           # 정답 PDF, OMR, 문제지 PDF 파서 + 파일 자동 분류
core/grader.py          # 채점·통계
core/exporter.py        # 엑셀 생성
agent/                  # tools(조회 Tool), prompts(시스템 프롬프트), guard(인젝션 방어), agent(실행·스트리밍)
tests/                  # pytest + 채팅 회귀 스크립트
```
