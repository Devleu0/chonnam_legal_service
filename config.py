"""전역 설정 값."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = str(BASE_DIR / "data" / "laws.json")
PERSIST_DIR = str(BASE_DIR / "chroma_db")

EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-multitask"
DEFAULT_LLM_MODEL = "gpt-4o-mini"

TOP_K = 3
VECTOR_WEIGHT = 0.5  # 하이브리드 검색: BM25 0.5 + 벡터 0.5
