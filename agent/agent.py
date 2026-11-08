"""LLM + Tool 조립, 스트리밍 실행, 히스토리 관리."""
from __future__ import annotations

import secrets
import time
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache

import httpx
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call
from langchain_core.messages import AIMessageChunk, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError

import config
from agent.guard import REFUSAL, is_injection, leaks_internal
from agent.prompts import build_system_prompt
from agent.tools import build_tools
from core.models import GradingResult

RECURSION_LIMIT = 12
DEADLINE_S = 18
HISTORY_TURNS = 10
MSG_TIMEOUT = "답변 생성이 지연되고 있습니다. 질문을 조금 더 간단히 해서 다시 시도해 주세요."
MSG_RECURSION = "요청을 처리하는 데 시간이 너무 걸렸습니다. 질문을 조금 더 구체적으로 해 주세요."
MSG_ERROR = "일시적인 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
MSG_INTERRUPTED = "\n\n(응답이 중간에 중단되었습니다. 다시 질문해 주세요.)"


class MissingAPIKey(RuntimeError):
    pass


@lru_cache(maxsize=4)
def get_llm(model: str, reasoning_effort: str, api_key: str) -> ChatOpenAI:
    kwargs = dict(
        model=model,
        use_responses_api=True,
        max_completion_tokens=3000,
        timeout=httpx.Timeout(12, connect=3),
        max_retries=1,
        api_key=api_key,
    )
    if reasoning_effort and reasoning_effort.lower() != "default":
        kwargs["reasoning_effort"] = reasoning_effort
    return ChatOpenAI(**kwargs)


@dataclass
class ChatSession:
    agent: object
    canary: str


def generation_middleware(gen_llm: ChatOpenAI):
    """유사 문제를 '작성하는' 모델 호출에만 더 높은 추론 강도를 쓴다.

    직전 메시지가 prepare_similar_question의 결과일 때만 모델을 바꾼다. 일반 조회 질문은 빠른 설정 그대로.
    (실측: low 설정에서 어법 유형은 정답이 없는 문제를 만드는 경우가 있었음)
    """

    @wrap_model_call
    def stronger_reasoning_for_generation(request, handler):
        last = request.messages[-1] if request.messages else None
        if isinstance(last, ToolMessage) and last.name == "prepare_similar_question":
            return handler(request.override(model=gen_llm))
        return handler(request)

    return stronger_reasoning_for_generation


def build_session(result: GradingResult) -> ChatSession:
    api_key = config.get_secret("OPENAI_API_KEY")
    if not api_key:
        raise MissingAPIKey("OPENAI_API_KEY가 설정되지 않았습니다.")
    canary = f"CANARY-{secrets.token_hex(6)}"
    llm = get_llm(config.OPENAI_MODEL, config.OPENAI_REASONING_EFFORT or "", api_key)
    middleware = []
    if config.OPENAI_GENERATION_EFFORT and config.OPENAI_GENERATION_EFFORT != config.OPENAI_REASONING_EFFORT:
        middleware.append(generation_middleware(get_llm(config.OPENAI_MODEL, config.OPENAI_GENERATION_EFFORT, api_key)))
    agent = create_agent(
        model=llm,
        tools=build_tools(result),
        system_prompt=build_system_prompt(result, canary, config.APP_TITLE),
        middleware=middleware,
    )
    return ChatSession(agent=agent, canary=canary)


def trim_history(messages: list[dict], turns: int = HISTORY_TURNS) -> list[dict]:
    """role/content 텍스트만, 최근 N턴, 항상 user 메시지로 시작."""
    msgs = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] in ("user", "assistant")]
    msgs = msgs[-turns * 2:]
    while msgs and msgs[0]["role"] != "user":
        msgs.pop(0)
    return msgs


def answer_stream(session: ChatSession, history: list[dict]) -> Iterator[str]:
    """화면에 내보낼 텍스트 조각만 yield (Tool 호출/결과 청크는 제외)."""
    if history and is_injection(history[-1]["content"]):
        yield REFUSAL
        return
    start, got, last_step = time.monotonic(), False, None
    try:
        stream = session.agent.stream(
            {"messages": trim_history(history)},
            config={"recursion_limit": RECURSION_LIMIT},
            stream_mode="messages",
        )
        for chunk, meta in stream:
            # 시간 검사는 필터보다 먼저: Tool 단계 청크가 오갈 때도 마감 시간을 확인한다
            if not got and time.monotonic() - start > DEADLINE_S - 1:
                yield MSG_TIMEOUT
                return
            if meta.get("langgraph_node") != "model" or not isinstance(chunk, AIMessageChunk):
                continue
            text = chunk.text
            if not text:
                continue
            step = meta.get("langgraph_step")
            if got and step != last_step:  # 서로 다른 모델 호출의 텍스트는 문단으로 구분
                yield "\n\n"
            last_step, got = step, True
            yield text
    except GraphRecursionError:
        yield MSG_INTERRUPTED if got else MSG_RECURSION
    except Exception:  # noqa: BLE001  (네트워크/API 오류 — 내부 메시지 노출 금지)
        yield MSG_INTERRUPTED if got else MSG_ERROR


def finalize_answer(session: ChatSession, answer: str) -> str:
    """출력 유출 필터: 내부 정보가 보이면 거절 문구로 대체."""
    return REFUSAL if leaks_internal(answer, session.canary) else answer


def ask(session: ChatSession, history: list[dict]) -> str:
    """비스트리밍 헬퍼(테스트·회귀용)."""
    return finalize_answer(session, "".join(answer_stream(session, history)))
