"""
⚖️ 광주/전남 지역 B2B 맞춤형 법률 검토 에이전트 - Streamlit 프로토타입

제안서(5조 진흥) 기반 구현:
- 조·항·호 구조를 보존하는 구조적 청킹 (src/ingestion/chunker.py)
- BM25 + 임베딩 하이브리드 검색 (src/rag/retriever.py)
- 환각 억제 프롬프트 + 지역 법률 인프라 자동 연계 (src/rag/chain.py, infra_mapping.py)
"""
from __future__ import annotations

import os

import streamlit as st

import config
from src.rag.chain import build_rag_chain
from src.rag.retriever import build_hybrid_retriever

st.set_page_config(page_title="지역 법률 상담 서비스", page_icon="⚖️", layout="wide")

# ---------------------------------------------------------
# 1. 사이드바: API 키 및 검색 옵션
# ---------------------------------------------------------
st.sidebar.title("🔑 5조 진흥 설정 패널")
api_key = st.sidebar.text_input("OpenAI API Key를 입력하세요", type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

st.sidebar.markdown("---")
st.sidebar.subheader("검색 옵션")
region_filter = st.sidebar.selectbox("지역 필터", ["전체", "광주", "전남"], index=0)
top_k = st.sidebar.slider("검색 문서 수 (k)", min_value=1, max_value=5, value=config.TOP_K)
model_name = st.sidebar.selectbox("LLM 모델", ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o"], index=0)

st.sidebar.markdown("---")
st.sidebar.caption("본 프로토타입은 5조 진흥의 '『LLM을 활용한 지역 법률 상담 서비스 시스템 개발』' 제안서를 바탕으로 제작되었습니다.")


# ---------------------------------------------------------
# 2. 벡터스토어 및 하이브리드 검색기 초기화 (캐시)
# ---------------------------------------------------------
@st.cache_resource(show_spinner="법령 데이터셋 로드 및 벡터 DB 구축 중...")
def get_retriever(k: int):
    return build_hybrid_retriever(config.DATA_PATH, config.PERSIST_DIR, k=k)


# ---------------------------------------------------------
# 3. 프론트엔드 UI/UX
# ---------------------------------------------------------
st.title("⚖️ 광주/전남 지역 B2B 맞춤형 법률 검토 에이전트")
st.markdown(
    """
이 서비스는 별도의 법무팀이 없는 지역 내 스타트업 및 중소기업을 위해
**지자체 조례, 산업단지 규정, 표준 계약서** 등을 검토해 드리는 AI 챗봇입니다.

- 📚 데이터: 광주/전남 자치법규, 산업단지 입주규정, 표준 NDA/근로계약서 (조·항·호 구조 보존)
- 🔍 검색: BM25 + 임베딩 하이브리드 검색으로 환각 억제
- 📍 결과 하단에 관련 **지역 법률 인프라**(연락처/링크) 자동 연계
"""
)

with st.expander("💬 예시 질문 보기"):
    st.markdown(
        "- 신재생에너지 스타트업이 타 지자체 보조금을 이미 받았는데 광주시 보조금도 받을 수 있나요?\n"
        "- NDA 계약서에서 손해배상 범위가 명시되지 않은 조항은 독소조항인가요?\n"
        "- 산업단지 임대료 감면을 받은 후 3년 이내에 사업장을 이전하면 어떻게 되나요?\n"
        "- 근로계약서에 연차유급휴가 조건을 명시하지 않아도 되나요?"
    )

if not api_key:
    st.warning("👈 왼쪽 패널에 OpenAI API Key를 입력해야 서비스를 시작할 수 있습니다.")
    st.stop()

retriever = get_retriever(top_k)
rag_chain = build_rag_chain(retriever, model_name=model_name)

# ---------------------------------------------------------
# 4. 채팅 인터페이스
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_query = st.chat_input("검토가 필요한 계약서 조항이나 보조금 규정에 대해 질문해주세요.")

if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        with st.spinner("관련 광주/전남 조례 및 규정을 하이브리드 검색 중입니다..."):
            try:
                query = user_query
                if region_filter != "전체":
                    query = f"[{region_filter} 지역 한정] {user_query}"
                response = rag_chain.invoke(query)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:  # noqa: BLE001
                st.error(f"오류가 발생했습니다: {e}")
