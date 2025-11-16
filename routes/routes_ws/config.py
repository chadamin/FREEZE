"""설정값 및 환경변수"""
import os

# Base URL
BASE_URL = os.getenv("BASE_URL", "").rstrip("/")
_BASE_URL_DYNAMIC = ""  # auto-detected base url cache

# YOLO
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8n.pt")

# 출력/저장 설정
OUTPUT_DIR    = os.getenv("OUTPUT_DIR", "./clips")
SNAP_QUALITY  = int(os.getenv("SNAP_QUALITY", "90"))
PRE_ROLL_SEC  = float(os.getenv("PRE_ROLL_SEC", "5.0"))
POST_ROLL_SEC = float(os.getenv("POST_ROLL_SEC", "10.0"))
FALLBACK_FPS  = float(os.getenv("FALLBACK_FPS", "15.0"))
VIDEO_CODEC   = os.getenv("VIDEO_CODEC", "mp4v")
MAX_WIDTH     = int(os.getenv("MAX_WIDTH", "1920"))

# 스트리밍 설정
INLINE_SNAPSHOT_B64 = os.getenv("INLINE_SNAPSHOT_B64", "1") == "1"
WS_VIDEO_STREAM     = os.getenv("WS_VIDEO_STREAM", "0") == "1"
WS_VIDEO_CHUNK      = int(os.getenv("WS_VIDEO_CHUNK", "65536"))

# 오디오 저장 설정
SAVE_AUDIO      = os.getenv("SAVE_AUDIO", "1") == "1"
SAVE_AUDIO_TS   = os.getenv("SAVE_AUDIO_TS", "0") == "1"
SAVE_AUDIO_LEN_SEC = float(os.getenv("SAVE_AUDIO_LEN_SEC", "3.0"))
AUDIO_OUT_DIR   = OUTPUT_DIR

# 오디오 기본 모드: env(환경음만) | speech(Whisper만) | both(둘다)
AUDIO_WS_DEFAULT_MODE = os.getenv("AUDIO_WS_DEFAULT_MODE", "env").lower()

# 오디오 RAW 프레임 설정 (RAW fallback)
RAW_SR = 16000
RAW_FRAME_MS = 200
RAW_SAMPLES = RAW_SR * RAW_FRAME_MS // 1000     # 3200
RAW_FRAME_BYTES = RAW_SAMPLES * 2               # int16 → 6400

# Whisper 포커스
FOCUS_HOTWORDS = [s.strip() for s in os.getenv("FOCUS_HOTWORDS", "").split(",") if s.strip()]
FOCUS_WINDOW_MS = int(os.getenv("FOCUS_WINDOW_MS", "5000"))

# 디렉토리 자동 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)
