import asyncio
import os
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from trainer.endpoints import factory_router
from trainer.utils.cleanup_loop import periodically_cleanup_tasks_and_cache
from validator.utils.logging import get_logger


load_dotenv(".trainer.env")

logger = get_logger(__name__)

_cleanup_task: asyncio.Task | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background cleanup loop on startup; cancel gracefully on shutdown."""
    global _cleanup_task
    logger.info("Starting async cleanup loop as a background task.")
    _cleanup_task = asyncio.create_task(periodically_cleanup_tasks_and_cache())

    yield  # application runs here

    if _cleanup_task and not _cleanup_task.done():
        _cleanup_task.cancel()
        try:
            await _cleanup_task
        except asyncio.CancelledError:
            logger.info("Cleanup loop stopped.")


def factory() -> FastAPI:
    logger.debug("Entering factory function")
    app = FastAPI(lifespan=lifespan)
    app.include_router(factory_router())

    # CORS — restrict to configured origins in production.
    # Set the ALLOWED_ORIGINS env-var to a comma-separated list of origins,
    # e.g. "https://orchestrator.example.com".  Defaults to "*" (open) which
    # is still gated by the orchestrator IP check on every endpoint.
    allowed_origins_raw = os.getenv("ALLOWED_ORIGINS", "*")
    allowed_origins = [o.strip() for o in allowed_origins_raw.split(",") if o.strip()]

    # allow_credentials=True is incompatible with wildcard origins in browsers;
    # only enable it when specific origins are configured.
    allow_credentials = allowed_origins != ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


app = factory()

if __name__ == "__main__":
    logger.info("Starting trainer")
    uvicorn.run(app, host="0.0.0.0", port=8001)
