# freeze/app.py

import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from .routes.ws_app import router as ws_app_router
from .routes.ws_esp32 import router as ws_esp32_router
from .routes.ws_audio import router as ws_audio_router
from .routes.ws_camera import router as ws_camera_router
from .routes_http import router as http_router
from .routes.ws_pi import router as ws_pi_router
from .routes.routes_numbers import router as numbers_router
from freeze.routes.route_whisper_inject import router as whisper_inject_router
from freeze.routes.route_number_send import router as number_router
from .routes.ui import router as ui_router
from .routes.route_debug_push import router as debug_router


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(http_router)
app.include_router(ws_app_router)
app.include_router(ws_esp32_router)
app.include_router(ws_audio_router)
app.include_router(ws_camera_router)
app.include_router(ws_pi_router)
app.include_router(numbers_router)
app.include_router(whisper_inject_router)
app.include_router(number_router)
app.include_router(ui_router)
app.include_router(debug_router)

