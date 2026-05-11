from sentence_transformers import SentenceTransformer

from app.config import get_model_cache_dir

_model: SentenceTransformer | None = None


def load_model() -> None:
    global _model
    path = get_model_cache_dir()
    _model = SentenceTransformer(path)


def get_model() -> SentenceTransformer:
    if _model is None:
        raise RuntimeError("모델이 아직 로드되지 않았습니다. 앱 lifespan에서 load_model()을 호출했는지 확인하세요.")
    return _model
