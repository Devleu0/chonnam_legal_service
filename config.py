"""전역 설정 값."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = str(BASE_DIR / "data" / "laws.json")
PERSIST_DIR = str(BASE_DIR / "chroma_db")

EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-multitask"

TOP_K = 3
VECTOR_WEIGHT = 0.5  # 하이브리드 검색: BM25 0.5 + 벡터 0.5

# ---------------------------------------------------------
# 멀티 LLM 프로바이더 설정 (2026-09 기준 실제 사용 가능한 모델로 검증)
# ---------------------------------------------------------
DEFAULT_PROVIDER = "openai"
DEFAULT_LLM_MODEL = "gpt-5.5"

PROVIDER_MODELS = {
    "openai": ["gpt-5.5", "gpt-5.1", "gpt-4o-mini"],
    "gemini": ["gemini-3.5-flash", "gemini-2.5-pro", "gemini-2.5-flash"],
    "claude": ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"],
}

PROVIDER_LABELS = {
    "openai": "OpenAI",
    "gemini": "Google Gemini",
    "claude": "Anthropic Claude",
}

# 각 프로바이더가 API 키를 찾는 환경변수명
PROVIDER_ENV_KEY = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
}
