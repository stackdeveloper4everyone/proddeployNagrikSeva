from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.app.api.routes import router as api_router
from backend.app.core.config import get_settings
from backend.app.core.security import InMemoryRateLimiter
from backend.app.services.memory_service import MemoryService
from backend.app.services.policy_agent import PolicyAgent
from backend.app.services.sarvam_client import SarvamClient
from backend.app.services.search_service import SearchService
from backend.app.services.session_store import SessionStore
from backend.app.services.vector_store import PolicyVectorStore


settings = get_settings()
app = FastAPI(title=settings.app_name, debug=settings.debug)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dir = Path(settings.frontend_dir)
app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

sarvam_client = SarvamClient(settings)
search_service = SearchService(settings)
vector_store = PolicyVectorStore(settings)
memory_service = MemoryService(settings)
session_store = SessionStore()
policy_agent = PolicyAgent(
    sarvam=sarvam_client,
    search_service=search_service,
    vector_store=vector_store,
    memory_service=memory_service,
)

app.state.rate_limiter = InMemoryRateLimiter(settings.rate_limit_per_minute)
app.state.session_store = session_store
app.state.policy_agent = policy_agent
app.state.memory_service = memory_service

app.include_router(api_router, prefix=settings.api_prefix)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(frontend_dir / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

