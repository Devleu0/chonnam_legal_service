"""
지역 법률 인프라 연계 모듈
--------------------------
제안서의 '시스템 및 지역 연계 측면' 목표 구현:
질의/검색된 문서의 카테고리·지역 메타데이터를 바탕으로
실질적으로 도움을 줄 수 있는 지역 지원 기관 정보를 매핑하여 제공한다.
"""
from __future__ import annotations

INFRA_DIRECTORY = [
    {
        "keywords": ["보조금", "산업단지", "중소기업", "스타트업", "지원"],
        "name": "광주테크노파크 법률지원단",
        "desc": "산업단지 입주규정·보조금 관련 법률 자문",
        "contact": "062-123-4567",
        "link": "https://www.gjtp.or.kr",
    },
    {
        "keywords": ["근로", "노무", "해고", "임금", "휴가"],
        "name": "광주지방고용노동청",
        "desc": "근로기준법 위반 신고 및 노무 상담",
        "contact": "1350 (고용노동상담센터)",
        "link": "https://www.moel.go.kr",
    },
    {
        "keywords": ["nda", "비밀유지", "계약서", "독소조항"],
        "name": "광주광역시 무료 법률상담소",
        "desc": "계약서 검토 및 독소조항 관련 무료 법률상담",
        "contact": "062-613-2000",
        "link": "https://www.gwangju.go.kr/law_counsel",
    },
    {
        "keywords": ["전남", "지방보조금", "도지사"],
        "name": "전남테크노파크 기업지원센터",
        "desc": "전남 지역 보조금·기업지원 규정 상담",
        "contact": "061-729-3114",
        "link": "https://www.jntp.or.kr",
    },
]

DEFAULT_INFRA = [
    {
        "name": "광주테크노파크 법률지원단",
        "desc": "지역 스타트업/중소기업 법률 자문 전반",
        "contact": "062-123-4567",
        "link": "https://www.gjtp.or.kr",
    }
]


def recommend_infra(query: str, retrieved_categories: list[str]) -> list[dict]:
    """질의 텍스트와 검색된 문서들의 category 메타데이터를 근거로
    연계할 지역 법률 인프라 리스트를 반환한다."""
    haystack = (query + " " + " ".join(retrieved_categories)).lower()
    matched = []
    for infra in INFRA_DIRECTORY:
        if any(kw in haystack for kw in infra["keywords"]):
            matched.append(infra)
    if not matched:
        matched = DEFAULT_INFRA
    seen = set()
    result = []
    for m in matched:
        if m["name"] not in seen:
            seen.add(m["name"])
            result.append(m)
    return result


def format_infra_markdown(infra_list: list[dict]) -> str:
    lines = ["\n💡 **지역 법률 인프라 연계**"]
    for infra in infra_list:
        lines.append(f"- **{infra['name']}** ({infra['desc']}) · 연락처: {infra['contact']} · {infra['link']}")
    return "\n".join(lines)
