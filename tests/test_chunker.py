"""구조적 청킹 모듈에 대한 간단한 단위 테스트."""
from src.ingestion.chunker import load_and_chunk


def test_load_and_chunk_preserves_article_structure():
    chunks = load_and_chunk("data/laws.json")
    assert len(chunks) > 0

    subsidy_chunk = next(c for c in chunks if c.metadata["article_no"] == "제5조")
    assert "①" in subsidy_chunk.text
    assert "②" in subsidy_chunk.text
    assert "1." in subsidy_chunk.text
    assert subsidy_chunk.metadata["region"] == "광주"
    assert subsidy_chunk.metadata["law_name"] == "광주광역시 중소기업 육성 및 지원 조례"


def test_all_chunks_have_required_metadata():
    chunks = load_and_chunk("data/laws.json")
    for c in chunks:
        for key in ["law_name", "law_type", "region", "category", "article_no"]:
            assert key in c.metadata
