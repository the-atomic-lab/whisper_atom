from fastapi import APIRouter

from app.api.routes import heartbeat, detect

router = APIRouter()
router.include_router(heartbeat.router, tags=["health"], prefix="/v1/health")
router.include_router(detect.router, tags=["asr"], prefix="/v1/apply")
