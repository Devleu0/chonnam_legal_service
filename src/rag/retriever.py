"""
하이브리드 검색기 (Hybrid Retriever)
------------------------------------
제안서 (b)/(평가 전략) "단순 벡터 검색(Naive RAG) vs 하이브리드 검색" 비교를 지원하기 위해
BM25(키워드 기반 희소 검색) + 임베딩 벡터 검색(밀집 검색)을 EnsembleRetriever로 결합한다.
"""
from __future__ import annotations

from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.ingestion.chunker import load_and_chunk

EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-multitask"


def load_documents(data_path: str) -> list[Document]:
    chunks = load_and_chunk(data_path)
    return [Document(page_content=c.text, metadata=c.metadata) for c in chunks]


def build_naive_retriever(persist_dir: str, k: int = 3):
    """단순 벡터 검색만 사용하는 Naive RAG 검색기 (평가 비교군)."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name="regional_legal_docs",
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})


def build_hybrid_retriever(data_path: str, persist_dir: str, k: int = 3, vector_weight: float = 0.5):
    """BM25 + 벡터 검색을 결합한 하이브리드 검색기 (본 시스템의 기본 검색기)."""
    documents = load_documents(data_path)

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = k

    vector_retriever = build_naive_retriever(persist_dir, k=k)

    hybrid = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[1 - vector_weight, vector_weight],
    )
    return hybrid


def filter_by_region(docs: list[Document], region: str | None) -> list[Document]:
    """메타데이터 필터링: 특정 지역(광주/전남)이 지정되면 '전국' 문서와 함께만 반환."""
    if not region:
        return docs
    return [d for d in docs if d.metadata.get("region") in (region, "전국")]
