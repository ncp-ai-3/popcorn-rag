from typing import List

from pydantic import BaseModel, Field


class EmbedRequest(BaseModel):
    text: str = Field(
        ...,
        description="PopupEmbeddingService.buildEmbeddingContent 와 동일한 문자열",
    )


class EmbedResponse(BaseModel):
    embedding: List[float]
    dimension: int
