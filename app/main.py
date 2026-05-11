from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.model_loader import load_model
from app.routers import embed


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="Popup Embedding API",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(embed.router)
