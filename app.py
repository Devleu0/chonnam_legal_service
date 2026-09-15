import streamlit as st
import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# ---------------------------------------------------------
# 1. 초기 설정 및 API 키 입력
# ---------------------------------------------------------
st.set_page_config(page_title="지역 법률 상담 서비스", page_icon="⚖️", layout="wide")

st.sidebar.title("🔑 5조 진흥 설정 패널") # 제안서의 팀명 반영
api_key = st.sidebar.text_input("OpenAI API Key를 입력하세요", type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

# ---------------------------------------------------------
# 2. 가상 데이터셋 및 벡터 DB 구축 (RAG)
# 제안서의 조/항/호 단위 데이터 처리 및 지역(광주/전남) 규정 반영
# ---------------------------------------------------------
@st.cache_resource
def initialize_vector_db():
    # 실제로는 XML/JSON을 파싱해야 하지만, 프로토타입을 위해 더미 데이터를 사용합니다.
    # 제안서에 명시된 '광주광역시 자치법규', 'NDA', '보조금법' 데이터의 예시입니다.
    dummy_laws = [
        "제1조(목적) 이 조례는 광주광역시 중소기업의 육성 및 지원에 관한 사항을 규정함을 목적으로 한다.",
        "제5조(보조금 지원) 광주광역시장은 지역 산업단지 입주 기업 중 신재생에너지 관련 기술을 개발하는 스타트업에 대하여 예산의 범위에서 보조금을 지원할 수 있다. 단, 타 지자체와 중복 수급은 불가하다.",
        "NDA 제3조(비밀유지 의무) '을'은 '갑'으로부터 제공받은 기술 자료를 제3자에게 누설해서는 안 되며, 위반 시 민형사상 책임을 진다. 광주테크노파크 표준 계약 기준에 따른다.",
        "근로기준법 제17조(근로조건의 명시) 사용자는 근로계약을 체결할 때에 근로자에게 임금, 소정근로시간, 제55조에 따른 휴일, 제60조에 따른 연차유급휴가, 그 밖에 대통령령으로 정하는 근로조건을 명시하여야 한다."
    ]
    
    # 임베딩 모델 로드 (한국어 처리에 좋은 오픈소스 모델 사용)
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    
    # 구조적 청킹이 완료된 텍스트라고 가정하고 Chroma DB에 저장
    vectorstore = Chroma.from_texts(
        texts=dummy_laws, 
        embedding=embeddings,
        metadatas=[{"source": "광주조례"}, {"source": "보조금규정"}, {"source": "표준NDA"}, {"source": "근로기준법"}]
    )
    return vectorstore.as_retriever(search_kwargs={"k": 2})

# ---------------------------------------------------------
# 3. LLM 프롬프트 엔지니어링 (시스템 프롬프트)
# ---------------------------------------------------------
# 제안서의 '지역 연계 측면' 및 '환각 억제' 목표를 달성하기 위한 프롬프트
template = """
당신은 광주/전남 지역의 스타트업 및 중소기업을 위한 B2B 맞춤형 법률·규정 검토 전문 AI 에이전트입니다.
반드시 제공된 [참고 법령/규정]만을 바탕으로 객관적인 위험 요소를 분석하여 답변하세요. (환각 현상 방지)

[참고 법령/규정]
{context}

[사용자 질의]
{question}

[답변 형식]
1. 규정 스캐닝 결과: 위반 소지 및 독소조항 여부
2. 상세 근거: 조/항/호를 기반으로 한 설명
3. 💡 지역 법률 인프라 연계: 답변 하단에 다음의 지역 지원 기관 정보를 반드시 포함하여 실질적인 문제 해결을 보조하세요.
   - 광주테크노파크 법률지원단 (연락처: 062-123-4567, 링크: https://www.gjtp.or.kr)
   - 광주지방고용노동청 (노무 상담)
   - 광주광역시 무료 법률상담소
"""
prompt = ChatPromptTemplate.from_template(template)

# ---------------------------------------------------------
# 4. Streamlit 프론트엔드 UI/UX
# ---------------------------------------------------------
st.title("⚖️ 광주/전남 지역 B2B 맞춤형 법률 검토 에이전트")
st.markdown("""
이 서비스는 별도의 법무팀이 없는 지역 내 스타트업 및 중소기업을 위해 
**지자체 조례, 산업단지 규정, 표준 계약서** 등을 검토해 드리는 AI 챗봇입니다.
*(본 프로토타입은 5조 진흥의 제안서를 바탕으로 제작되었습니다)*
""")

if not api_key:
    st.warning("👈 왼쪽 패널에 OpenAI API Key를 입력해야 서비스를 시작할 수 있습니다.")
    st.stop()

# 파이프라인 초기화
retriever = initialize_vector_db()
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0) # 객관성을 위해 temperature=0 설정

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG 체인 구성 (LangChain)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 채팅 인터페이스 구축
if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 대화 기록 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력
user_query = st.chat_input("검토가 필요한 계약서 조항이나 보조금 규정에 대해 질문해주세요.")

if user_query:
    # 1. 사용자 질문 출력
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # 2. AI 답변 생성 (RAG 파이프라인)
    with st.chat_message("assistant"):
        with st.spinner("관련 광주광역시 조례 및 판례를 검색 중입니다..."):
            try:
                response = rag_chain.invoke(user_query)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")