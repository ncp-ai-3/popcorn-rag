import os
from functools import lru_cache
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv_files() -> None:
    """레포 루트 `.env` → 현재 작업 디렉터리 `.env` (이미 설정된 환경 변수는 덮어쓰지 않음)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(_REPO_ROOT / ".env", override=False)
    load_dotenv(override=False)


_load_dotenv_files()


@lru_cache
def get_embedding_dimension() -> int:
    return int(os.getenv("EMBEDDING_DIMENSION", "768"))


@lru_cache
def get_embedding_model_id() -> str:
    """Hugging Face 모델 ID. 로컬 스냅샷이 없으면 이 ID로 cache_folder에 받음."""
    return os.getenv("EMBEDDING_MODEL_ID", "intfloat/multilingual-e5-base")


@lru_cache
def get_model_cache_dir() -> str:
    """캐시·로컬 스냅샷 루트. Docker에서는 보통 MODEL_CACHE_DIR=/model_cache."""
    return os.getenv("MODEL_CACHE_DIR", str(_REPO_ROOT / "model_cache"))
