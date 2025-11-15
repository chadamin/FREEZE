# app.py
import os
from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.routes_ws import router as ws_router
from routes.routes_http import router as http_router

# .env 로드
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

app = FastAPI()

# CORS (필요하면 origin 나중에 좁히기)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(ws_router)
app.include_router(http_router)
