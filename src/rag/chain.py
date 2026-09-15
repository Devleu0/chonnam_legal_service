"""
RAG 체인 구성
-------------
검색된 법령 근거(구조적 청킹 + 하이브리드 검색)를 프롬프트에 주입하여
환각을 억제한 답변을 생성하고, 지역 법률 인프라 연계 정보를 결과 하단에 덧붙인다.
"""
from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from src.rag.infra_mapping import format_infra_markdown, recommend_infra


def build_llm(provider: str, model_name: str, temperature: float = 0.0):
    """provider(openai/gemini/claude)에 맞는 LangChain Chat 모델 인스턴스를 생성한다.

    2026-09 기준 실제 사용 가능한 최신 모델로 검증된 값만 config.PROVIDER_MODELS 에 등록되어 있다.
    """
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model_name, temperature=temperature)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    if provider == "claude":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model_name, temperature=temperature)
    raise ValueError(f"지원하지 않는 provider: {provider}")

SYSTEM_TEMPLATE = """
당신은 광주/전남 지역의 스타트업 및 중소기업을 위한 B2B 맞춤형 법률·규정 검토 전문 AI 에이전트입니다.
반드시 [참고 법령/규정]에 제시된 내용만을 근거로 답변하세요. 근거에 없는 내용은 "제공된 자료에서 확인되지 않음"이라고 명시하고 추측하지 마세요. (환각 현상 방지)

[참고 법령/규정]
{context}

[사용자 질의]
{question}

[답변 형식]
1. 규정 스캐닝 결과: 위반 소지 및 독소조항 여부 (있음/없음/판단불가)
2. 상세 근거: 조·항·호 번호를 명시하여 설명 (예: OO조례 제5조 제2항)
3. 실무 권고사항: 사용자가 다음에 취해야 할 조치
"""

PROMPT = ChatPromptTemplate.from_template(SYSTEM_TEMPLATE)


def format_docs(docs) -> str:
    lines = []
    for d in docs:
        lines.append(f"({d.metadata.get('law_name')} {d.metadata.get('article_no')}) {d.page_content}")
    return "\n\n".join(lines)


def build_rag_chain(retriever, provider: str = "openai", model_name: str = "gpt-5.5", temperature: float = 0.0):
    """검색기 + LLM + 프롬프트를 결합한 RAG 체인을 반환한다.

    provider: "openai" | "gemini" | "claude"

    체인은 (answer, source_docs) 형태가 아닌, 지역 인프라 연계 정보까지
    포함된 최종 마크다운 문자열을 반환하는 하나의 함수(invoke(question)->str)로 노출한다.
    """
    llm = build_llm(provider, model_name, temperature)

    base_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )

    def _invoke_with_infra(question: str) -> str:
        docs = retriever.invoke(question)
        answer = base_chain.invoke(question)
        categories = [d.metadata.get("category", "") for d in docs]
        infra = recommend_infra(question, categories)
        return answer + "\n\n" + format_infra_markdown(infra)

    return RunnableLambda(_invoke_with_infra)


def build_naive_llm_chain(provider: str = "openai", model_name: str = "gpt-5.5", temperature: float = 0.0):
    """검색(RAG) 미적용 순수 LLM 체인. 평가 시 환각 억제 효과의 대조군으로 사용."""
    llm = build_llm(provider, model_name, temperature)
    naive_prompt = ChatPromptTemplate.from_template(
        "당신은 법률 상담 AI입니다. 다음 질문에 답하세요.\n\n[질문]\n{question}"
    )
    return naive_prompt | llm | StrOutputParser()
