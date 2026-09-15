"""
구조적 청킹(Structural Chunking) 모듈
------------------------------------
법률/조례/계약서 원문(JSON, 조·항·호 계층 구조)을 입력받아
문맥 훼손 없이 '조' 단위(필요 시 항 단위)로 청크를 생성하고,
검색 정확도를 높이기 위한 메타데이터(법령명, 지역, 카테고리, 조번호)를 함께 부여한다.

제안서 (a) 데이터 처리 방안: "조·항·호의 계층 구조를 보존하는 구조적 청킹" 구현체.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LawChunk:
    """하나의 검색 단위(청크)."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "metadata": self.metadata}


def _flatten_article(article: dict[str, Any], law_name: str) -> str:
    """조 하나(항/호 포함)를 사람이 읽는 원문 형태의 하나의 텍스트로 합친다.

    예)
    제5조(보조금 지원)
    ① 광주광역시장은 ... 지원할 수 있다.
    ② 제1항에 따른 보조금은 ...
       1. 국가 또는 다른 지방자치단체로부터...
       2. 허위 또는 부정한 방법으로...
    """
    lines = [f"{article['조']}({article.get('title', '')})".rstrip("()")]
    if article.get("title"):
        lines = [f"{article['조']}({article['title']})"]
    else:
        lines = [f"{article['조']}"]

    for para in article.get("항", []):
        prefix = f"{para['항']} " if para.get("항") else ""
        lines.append(f"{prefix}{para['내용']}".strip())
        for ho in para.get("호", []):
            lines.append(f"    {ho['호']} {ho['내용']}")

    return "\n".join(lines)


def load_and_chunk(json_path: str | Path) -> list[LawChunk]:
    """법령 JSON 파일을 로드하여 조 단위 구조적 청크 리스트로 변환한다."""
    json_path = Path(json_path)
    data = json.loads(json_path.read_text(encoding="utf-8"))

    chunks: list[LawChunk] = []
    for law in data:
        law_name = law["law_name"]
        for article in law.get("articles", []):
            text = _flatten_article(article, law_name)
            metadata = {
                "law_name": law_name,
                "law_type": law.get("law_type", ""),
                "region": law.get("region", "전국"),
                "category": ", ".join(law.get("category", [])),
                "article_no": article.get("조", ""),
                "article_title": article.get("title", ""),
                "source_url": law.get("source_url", ""),
                "source": law_name,
            }
            chunks.append(LawChunk(text=f"[{law_name} {article.get('조','')}({article.get('title','')})]\n{text}", metadata=metadata))
    return chunks


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/laws.json"
    for c in load_and_chunk(path):
        print("-" * 40)
        print(c.text)
        print(c.metadata)
