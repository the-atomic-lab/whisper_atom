from typing import Callable
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from app.engine import STT
from app.config import EnvVar

import logging

logger = logging.getLogger(__name__)


def start_app_handler(app: FastAPI) -> Callable:
    def startup() -> None:
        logger.info("Running app start handler.")
        app.state.asr = STT.load(
            model="resources/large-v3",
            diarizator="resources/pyannote/speaker-diarization-3.1.yml",
            device="cuda",
            compute_type=EnvVar.COMPUTE_TYPE,
            stt_infer_batch=EnvVar.STT_INFER_BATCH,
        )

    return startup


def stop_app_handler(app: FastAPI) -> Callable:
    def shutdown() -> None:
        logger.info("Running app shutdown handler.")
        app.state.searcher = None
        app.state.uploader = None

    return shutdown


async def exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": str(exc)},
    )
