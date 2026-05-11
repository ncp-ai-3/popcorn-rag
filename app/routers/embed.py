from fastapi import APIRouter, HTTPException

from app.config import get_embedding_dimension
from app.model_loader import get_model
from app.schemas import EmbedRequest, EmbedResponse

router = APIRouter(prefix="/api/v1", tags=["embed"])


@router.post("/embed", response_model=EmbedResponse)
def embed_sync(body: EmbedRequest) -> EmbedResponse:
    expected_dim = get_embedding_dimension()
    vec = get_model().encode(body.text or "").tolist()
    vec_f = [float(x) for x in vec]
    if len(vec_f) != expected_dim:
        raise HTTPException(
            status_code=500,
            detail=(
                f"embedding dimension mismatch: expected {expected_dim}, got {len(vec_f)}"
            ),
        )
    return EmbedResponse(embedding=vec_f, dimension=expected_dim)
