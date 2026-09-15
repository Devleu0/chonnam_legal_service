"""
RAGAS 기반 평가 스크립트
------------------------
제안서 (b)-4) 평가 전략 구현:
  1) 일반 LLM(검색 미적용) vs 본 RAG 시스템 비교 -> 환각 억제 효과 검증
  2) Naive RAG(단순 벡터 검색) vs Hybrid RAG(BM25+벡터) 비교
     -> 구조적 청킹 + 하이브리드 검색 도입에 따른 정확도/근거 기반성 향상 폭 검증

평가지표(RAGAS): faithfulness(근거 충실도), answer_relevancy(답변 관련성),
context_precision(문맥 정밀도), context_recall(문맥 재현율)

실행:
    export OPENAI_API_KEY=sk-...
    python -m eval.evaluate_ragas
"""
from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

import config
from src.rag.chain import build_naive_llm_chain, build_rag_chain
from src.rag.retriever import build_hybrid_retriever, build_naive_retriever

# 평가용 골드 질의셋 (실무 빈발 질문 + 정답 근거 요약)
EVAL_SET = [
    {
        "question": "신재생에너지 스타트업이 타 지자체 보조금을 이미 받았는데 광주시 보조금도 받을 수 있나요?",
        "ground_truth": "광주광역시 중소기업 육성 및 지원 조례 제5조 제2항에 따라 타 지자체로부터 동일 목적의 보조금을 중복 지원받은 경우 지급하지 않으므로 받을 수 없습니다.",
    },
    {
        "question": "산업단지 임대료 감면을 받은 후 3년 이내에 사업장을 이전하면 어떻게 되나요?",
        "ground_truth": "광주광역시 산업단지 입주기업 지원 규정 제12조 제2항에 따라 감면 기간 중 정당한 사유 없이 이전하면 감면액 전부를 반환해야 합니다.",
    },
    {
        "question": "NDA 계약서에서 손해배상 범위가 명시되지 않은 조항은 독소조항인가요?",
        "ground_truth": "표준 비밀유지계약서 제3조 제2항은 배상 범위의 상한이 명시되지 않으면 '을'에게 불리한 독소조항으로 분류될 수 있습니다.",
    },
    {
        "question": "근로계약서에 연차유급휴가 조건을 명시하지 않아도 되나요?",
        "ground_truth": "근로기준법 제17조 제1항에 따라 사용자는 연차유급휴가를 포함한 근로조건을 근로계약 체결 시 반드시 명시해야 합니다.",
    },
]


def _run_pipeline(chain_type: str) -> Dataset:
    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    if chain_type == "naive_llm":
        chain = build_naive_llm_chain()
        for item in EVAL_SET:
            rows["question"].append(item["question"])
            rows["answer"].append(chain.invoke({"question": item["question"]}))
            rows["contexts"].append([""])  # 검색 미적용
            rows["ground_truth"].append(item["ground_truth"])

    elif chain_type == "naive_rag":
        retriever = build_naive_retriever(config.PERSIST_DIR, k=config.TOP_K)
        chain = build_rag_chain(retriever)
        for item in EVAL_SET:
            docs = retriever.invoke(item["question"])
            rows["question"].append(item["question"])
            rows["answer"].append(chain.invoke(item["question"]))
            rows["contexts"].append([d.page_content for d in docs])
            rows["ground_truth"].append(item["ground_truth"])

    elif chain_type == "hybrid_rag":
        retriever = build_hybrid_retriever(config.DATA_PATH, config.PERSIST_DIR, k=config.TOP_K)
        chain = build_rag_chain(retriever)
        for item in EVAL_SET:
            docs = retriever.invoke(item["question"])
            rows["question"].append(item["question"])
            rows["answer"].append(chain.invoke(item["question"]))
            rows["contexts"].append([d.page_content for d in docs])
            rows["ground_truth"].append(item["ground_truth"])
    else:
        raise ValueError(chain_type)

    return Dataset.from_dict(rows)


def main() -> None:
    results = {}
    for chain_type in ["naive_llm", "naive_rag", "hybrid_rag"]:
        print(f"=== {chain_type} 평가 실행 중 ===")
        dataset = _run_pipeline(chain_type)
        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
        score = evaluate(dataset, metrics=metrics)
        results[chain_type] = score.to_pandas().mean(numeric_only=True).to_dict()

    Path("eval").mkdir(exist_ok=True)
    out_path = Path("eval/ragas_results.json")
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"결과 저장 완료 -> {out_path}")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
