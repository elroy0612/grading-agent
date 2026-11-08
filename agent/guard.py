"""프롬프트 인젝션 방어: 입력 가드(LLM 호출 전) + 출력 유출 필터(답변 후).

입력 가드는 '정밀도 우선'이다. 영어 학원에서 흔한 정상 질문(function word, writing prompt,
'이 툴로 뭘 할 수 있어?')은 통과시키고, 놓친 공격은 시스템 프롬프트 보안 규칙(2층)과 출력 필터(3층)가 막는다.
"""
from __future__ import annotations

import re
import unicodedata

from agent.tools import TOOL_NAMES

REFUSAL = ("죄송하지만 시스템 내부 구성이나 설정 정보는 보안상 알려드릴 수 없습니다. "
           "채점 결과나 문제 생성에 관해 궁금한 점을 물어봐 주세요.")


def _norm(s: str) -> str:
    """NFKC + 소문자 + 공백·구두점 제거 → 'T o o l' 같은 띄어쓰기 우회 차단."""
    s = unicodedata.normalize("NFKC", s).lower()
    return re.sub(r"[\s.,!?'\"`\-_~:;/\\()\[\]{}<>*#]+", "", s)


# 내부 구성 그 자체를 가리키는 말 → 넓은 요청어와 함께 나오면 차단
STRONG = (r"(시스템프롬프트|systemprompt|프롬프트|시스템메시지|api키|apikey|langchain|langgraph|openai|gpt|llm"
          r"|(?:your|system|너의|네|니)(?:prompts?|instructions?|functions?|tools?|rules?|규칙|지시|지침|설정))")
ASK = (r"(알려|보여|말해|출력|공개|나열|리스트|목록|뭐|뭔|무엇|어떤|무슨|있어|있나|연결|쓰고|써|복사|그대로|전부|모두|요약|번역"
       r"|print|show|list|reveal|tell|what|which|dump|repeat|translate|summari[sz]e)")
# 일상 어휘와 겹치는 말('도구', 'function word', '함수 그래프') → 좁은 요청어와 함께일 때만 차단.
# '이 툴/이 도구'(= 이 앱)는 기능 질문이므로 제외.
TOOLISH = r"((?<!이)(?<!엑셀)(?<!추출)(?:tools?|툴|도구|함수|functions?|플러그인|plugins?))"
ASK_NARROW = (r"(연결|목록|리스트|어떤|무슨|뭐뭐|뭐있|이름|호출|쓰고|사용하고|사용중|있나|붙어"
              r"|list|which|what|available|connected|using|have)")
GEN = r"(설정|모델|지시|지침|구조|코드|명령|model|settings?|config|rules|instructions)"
YOU = r"(너(?!무|비|머|희)|니가|네가|당신|챗봇|(?<!로)봇|에이전트|you|your)"

R_STRONG = re.compile(f"{STRONG}.{{0,12}}{ASK}|{ASK}.{{0,12}}{STRONG}")
R_TOOL = re.compile(f"{TOOLISH}.{{0,10}}{ASK_NARROW}|{ASK_NARROW}.{{0,10}}{TOOLISH}")
R_YOU = re.compile(f"{YOU}.{{0,10}}{GEN}.{{0,12}}{ASK}|{YOU}.{{0,6}}{ASK}.{{0,6}}{GEN}|{ASK}.{{0,6}}{GEN}.{{0,8}}{YOU}")
R_OVR = re.compile(
    r"(이전|앞|위|기존|모든|원래)(의|에)?(지시|명령|규칙|지침|설정|텍스트|프롬프트).{0,8}(무시|잊|버려|출력|반복|그대로)"
    r"|ignore(all|previous|the|above|your)|disregard|forget(all|previous|your)|jailbreak|탈옥|개발자모드|developermode"
    r"|디버그모드|debugmode|관리자모드|adminmode|repeat(the)?(text|words|everything|above)|above(text|instructions)"
    r"|print(the)?above|(너|you).{0,6}(이제|now).{0,15}(제한|규칙|필터|검열).{0,4}(없|no)"
)

LEAK = re.compile("|".join(TOOL_NAMES + [r"build_tools", r"create_agent", r"SYSTEM_PROMPT", r"langchain",
                                         r"langgraph", r"openai", r"chatgpt", r"gpt-?\d", r"reasoning_effort",
                                         r"CANARY-"]), re.I)


def is_injection(text: str) -> bool:
    t = _norm(text)
    return bool(R_OVR.search(t) or R_STRONG.search(t) or R_TOOL.search(t) or R_YOU.search(t))


def leaks_internal(answer: str, canary: str | None = None) -> bool:
    return bool(LEAK.search(answer) or (canary and canary in answer))
