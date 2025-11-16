"""라우터 통합"""
from fastapi import APIRouter
from .routes_http import router as http_router
from .routes_ws import router as ws_router

# 메인 라우터
router = APIRouter()
router.include_router(http_router)
router.include_router(ws_router)

__all__ = ["router"]
