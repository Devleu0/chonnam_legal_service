"""
벡터 DB(ChromaDB) 구축 스크립트
------------------------------
구조적으로 청킹된 법령 데이터를 임베딩하여 Chroma 벡터스토어로 영속화한다.
한국어 처리 성능이 검증된 오픈소스 임베딩 모델(jhgan/ko-sroberta-multitask)을 사용한다.

실행:
    python -m src.ingestion.build_vectorstore --data data/laws.json --persist ./chroma_db
"""
from __future__ import annotations

import argparse
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.ingestion.chunker import load_and_chunk

EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-multitask"


def build_vectorstore(data_path: str, persist_dir: str) -> Chroma:
    chunks = load_and_chunk(data_path)
    texts = [c.text for c in chunks]
    metadatas = [c.metadata for c in chunks]

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=persist_dir,
        collection_name="regional_legal_docs",
    )
    return vectorstore


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="법령 데이터 벡터 DB 구축")
    parser.add_argument("--data", default="data/laws.json")
    parser.add_argument("--persist", default="./chroma_db")
    args = parser.parse_args()

    Path(args.persist).mkdir(parents=True, exist_ok=True)
    vs = build_vectorstore(args.data, args.persist)
    print(f"벡터 DB 구축 완료 -> {args.persist} (컬렉션: regional_legal_docs)")
