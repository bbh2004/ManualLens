from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import CORS_ORIGINS
from backend.database import init_db
from backend.utils.logging_config import setup_logging, get_logger
from backend.api import upload, status, query, documents, pages

setup_logging()
logger = get_logger("manuallens.main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup actions
    logger.info("ManualLens Backend starting up...")
    init_db()
    yield
    # Shutdown actions
    logger.info("ManualLens Backend shutting down...")

app = FastAPI(
    title="ManualLens Backend API",
    description="Multimodal RAG Application for Technical PDF Manuals",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Route registration
app.include_router(upload.router)
app.include_router(status.router)
app.include_router(query.router)
app.include_router(documents.router)
app.include_router(pages.router)

@app.get("/health", tags=["health"])
def health_check():
    """Health check endpoint for server and deployment monitoring."""
    return {"status": "ok"}

@app.get("/api/warmup", tags=["warmup"])
@app.post("/api/warmup", tags=["warmup"])
def warmup_services():
    """Predictive pre-warming endpoint for Modal serverless GPU containers."""
    from backend.config import USE_MODAL
    if not USE_MODAL:
        return {"status": "skipped", "message": "USE_MODAL is false. Running locally."}
    try:
        from backend.modal_worker import get_modal_worker
        worker = get_modal_worker()
        res = worker.ping.remote()
        return {"status": "ok", "modal_response": res, "message": "GPU worker is warm and ready."}
    except Exception as e:
        logger.warning(f"Modal warmup ping failed: {e}")
        return {"status": "error", "message": str(e)}

