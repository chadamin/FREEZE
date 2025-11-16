"""WebSocket 라우터 통합"""
from fastapi import APIRouter
from .esp32_ws import router as esp32_router
from .app_ws import router as app_router
from .audio_ws import router as audio_router
from .camera_ws import router as camera_router

# 메인 라우터 생성
router = APIRouter(tags=["websocket"])

# 각 WebSocket 라우터 포함
router.include_router(esp32_router)
router.include_router(app_router)
router.include_router(audio_router)
router.include_router(camera_router)

__all__ = ["router"]
