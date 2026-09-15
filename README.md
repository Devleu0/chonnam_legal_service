# ⚖️ LLM을 활용한 지역 법률 상담 서비스 시스템 (5조 진흥)

산학협력프로젝트(캡스톤디자인) 제안서를 바탕으로 구현한
**광주/전남 지역 스타트업·중소기업을 위한 B2B 맞춤형 법률·규정 검토 AI 에이전트**입니다.

에이전틱 AI와 하이브리드 검색 기반 RAG(검색 증강 생성) 기술로 지자체 조례, 산업단지 규정,
표준 계약서(NDA/근로계약서)를 검토하고, 결과 하단에 실제 지역 법률 지원 기관 정보를 연계합니다.

## 팀 구성원 (5조 진흥)

| 이름 | 학과 | 역할 |
|---|---|---|
| 위정우 | 인공지능학부 | 팀장 - 프로젝트 총괄, 파이프라인 설계 |
| 김한백 | 인공지능학부 | RAG 아키텍처 구축 및 벡터 DB 최적화 |
| 강준서 | 인공지능학부 | 법률 데이터 크롤링 및 텍스트 전처리 |
| 김태규 | 자율전공 | 프론트엔드 UI/UX 기획 및 웹 개발 |
| 정우진 | 소프트웨어공학과 | LLM API 연동, 서비스 아키텍처 설계 |

## 아키텍처

```
data/laws.json (조·항·호 구조 원시 데이터)
        │
        ▼
src/ingestion/chunker.py       구조적 청킹 (조 단위, 계층 보존 + 메타데이터 부여)
        │
        ▼
src/ingestion/build_vectorstore.py   ko-sroberta 임베딩 → Chroma 벡터 DB 영속화
        │
        ▼
src/rag/retriever.py           BM25(희소) + 임베딩(밀집) 하이브리드 검색 (EnsembleRetriever)
        │                       + 지역(광주/전남) 메타데이터 필터링
        ▼
src/rag/chain.py               환각 억제 프롬프트 + LLM 응답 생성
        │
        ▼
src/rag/infra_mapping.py       질의/카테고리 기반 지역 법률 인프라 자동 매핑
        │
        ▼
app.py (Streamlit)             채팅 UI, 지역/모델/k 값 옵션 제공
```

평가는 `eval/evaluate_ragas.py`에서 RAGAS 프레임워크로 다음을 비교합니다.

1. 일반 LLM(검색 미적용) vs 본 하이브리드 RAG → 환각 억제 효과 검증
2. Naive RAG(단순 벡터 검색) vs Hybrid RAG(BM25+벡터) → 구조적 청킹·하이브리드 검색 도입 효과 검증

## 폴더 구조

```
.
├── app.py                       # Streamlit 프로토타입 엔트리포인트
├── config.py                    # 공통 설정값
├── data/laws.json                # 조·항·호 구조 예시 법령/계약서 데이터셋
├── src/
│   ├── ingestion/
│   │   ├── chunker.py            # 구조적 청킹
│   │   └── build_vectorstore.py  # Chroma 벡터 DB 구축 스크립트
│   └── rag/
│       ├── retriever.py          # BM25+임베딩 하이브리드 검색기
│       ├── chain.py               # RAG 체인 (프롬프트+LLM)
│       └── infra_mapping.py       # 지역 법률 인프라 연계 로직
├── eval/
│   └── evaluate_ragas.py         # RAGAS 정량 평가 스크립트
└── tests/
    └── test_chunker.py           # 구조적 청킹 단위 테스트
```

## 실행 방법

```bash
pip install -r requirements.txt

# 1) 벡터 DB 사전 구축 (선택, 최초 1회 - app.py 실행 시 자동으로도 생성됨)
python -m src.ingestion.build_vectorstore --data data/laws.json --persist ./chroma_db

# 2) 웹 서비스 실행
streamlit run app.py
```

실행 후 사이드바에 OpenAI API Key를 입력하면 사용할 수 있습니다.

## 평가 실행

```bash
export OPENAI_API_KEY=sk-...
python -m eval.evaluate_ragas
```

## 향후 계획 (제안서 (d) 프로젝트 계획 반영)

- [x] 법률 데이터 구조 설계 (조·항·호) 및 구조적 청킹 구현
- [x] 하이브리드 검색(BM25+벡터) 기반 RAG 파이프라인 구축
- [x] 프롬프트 엔지니어링(환각 억제) 및 지역 인프라 연계 로직
- [x] 웹 서비스 프로토타입(Streamlit) 구현
- [x] RAGAS 기반 정량 평가 스크립트
- [ ] 국가법령정보센터 Open API 연동을 통한 실데이터 자동 수집·갱신
- [ ] 메타데이터 필터링 UI 고도화 및 대법원 판례 데이터 추가
