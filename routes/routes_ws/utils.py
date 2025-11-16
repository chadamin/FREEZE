"""공통 유틸 함수 및 헬퍼"""
import json
import struct
import uuid
import traceback
import base64
import io
import wave
from typing import Dict, Set, Optional
import asyncio
from fastapi import WebSocket

from .config import BASE_URL

# ====== Base URL 관리 ======
_BASE_URL_DYNAMIC = ""

def effective_base_url() -> str:
    """BASE_URL 또는 동적으로 감지된 URL 반환"""
    global _BASE_URL_DYNAMIC
    return BASE_URL or _BASE_URL_DYNAMIC or ""

def file_url(name: str) -> str:
    """파일 URL 생성"""
    rel = f"/clips/{name}"
    base = effective_base_url()
    return (base + rel) if base else rel

def maybe_set_base_url_from_ws(ws: WebSocket):
    """BASE_URL 미설정 시, WebSocket 요청 헤더로 절대 URL 베이스 자동 감지"""
    global _BASE_URL_DYNAMIC
    try:
        if BASE_URL:
            return
        headers = dict(ws.headers or {})
        norm = {k.lower(): v for k, v in headers.items()}
        scheme = (norm.get("x-forwarded-proto") or "http").split(",")[0].strip()
        host = (norm.get("x-forwarded-host") or norm.get("host") or "").split(",")[0].strip()
        if host:
            detected = f"{scheme}://{host}"
            if detected != _BASE_URL_DYNAMIC:
                _BASE_URL_DYNAMIC = detected.rstrip("/")
                print(f"[BASE_URL] detected from WS headers -> {_BASE_URL_DYNAMIC}")
    except Exception as e:
        log_exc("[BASE_URL detect]", e)

# ====== JSON 유틸 ======
def safe_json(data: dict) -> str:
    """안전한 JSON 직렬화"""
    try:
        return json.dumps(data, ensure_ascii=False)
    except Exception as e:
        print("[JSON ERROR]", e)
        return "{}"

def log_exc(prefix: str, err: Exception):
    """예외 로깅"""
    print(f"{prefix}: {err.__class__.__name__}: {err}")
    print(traceback.format_exc())

# ====== 바이너리 프로토콜 ======
_BIN_MAGIC = b"MED0"

def pack_media_header(transfer_id: uuid.UUID, seq: int, kind: int) -> bytes:
    """미디어 프레임 헤더 패킹 (b"MED0" + uuid + seq + kind)"""
    try:
        return _BIN_MAGIC + transfer_id.bytes + struct.pack(">IB", seq, kind)
    except Exception as e:
        log_exc("[_pack_media_header]", e)
        return _BIN_MAGIC + (b"\x00" * 16) + struct.pack(">IB", 0, kind)

# ====== App 클라이언트 허브 ======
Topic = str
_app_clients: Dict[Topic, Set[WebSocket]] = {}
_app_lock = asyncio.Lock()

async def app_add(topic: Topic, ws: WebSocket):
    """앱 클라이언트 추가"""
    async with _app_lock:
        _app_clients.setdefault(topic, set()).add(ws)

async def app_remove(topic: Topic, ws: WebSocket):
    """앱 클라이언트 제거"""
    async with _app_lock:
        if topic in _app_clients and ws in _app_clients[topic]:
            _app_clients[topic].remove(ws)
            if not _app_clients[topic]:
                _app_clients.pop(topic, None)

async def _safe_send_text(ws: WebSocket, data: str) -> bool:
    """안전한 텍스트 전송"""
    try:
        await ws.send_text(data)
        return True
    except Exception as e:
        log_exc("[app_broadcast_json/send_text]", e)
        return False

async def _safe_send_bytes(ws: WebSocket, b: bytes) -> bool:
    """안전한 바이너리 전송"""
    try:
        await ws.send_bytes(b)
        return True
    except Exception as e:
        log_exc("[app_broadcast_binary/send_bytes]", e)
        return False

async def app_broadcast_json(topic: Topic, payload: dict):
    """토픽의 모든 클라이언트에게 JSON 브로드캐스트"""
    try:
        async with _app_lock:
            targets = list(_app_clients.get(topic, set()))
        data = safe_json(payload)
        dead = []
        for ws in targets:
            if not await _safe_send_text(ws, data):
                dead.append(ws)
        for ws in dead:
            await app_remove(topic, ws)
    except Exception as e:
        log_exc("[app_broadcast_json]", e)

async def app_broadcast_binary(topic: Topic, b: bytes):
    """토픽의 모든 클라이언트에게 바이너리 브로드캐스트"""
    try:
        async with _app_lock:
            targets = list(_app_clients.get(topic, set()))
        dead = []
        for ws in targets:
            if not await _safe_send_bytes(ws, b):
                dead.append(ws)
        for ws in dead:
            await app_remove(topic, ws)
    except Exception as e:
        log_exc("[app_broadcast_binary]", e)

# ====== 파일 관련 유틸 ======
def wav_dur_sec(b: bytes) -> float:
    """WAV 파일의 길이(초) 계산"""
    try:
        with wave.open(io.BytesIO(b), "rb") as wf:
            frames = wf.getnframes()
            sr = wf.getframerate()
            return (frames / float(sr)) if sr > 0 else 0.0
    except Exception:
        return 0.0

def b64_decode_safe(b64_str: str) -> Optional[bytes]:
    """Base64 안전 디코딩"""
    try:
        # 분리자 제거 (예: "data:image/jpeg;base64,..." -> "...")
        b64 = b64_str.split(",", 1)[-1]
        # 패딩 추가
        return base64.b64decode(b64 + ("=" * ((-len(b64)) % 4)))
    except Exception as e:
        log_exc("[b64_decode_safe]", e)
        return None
