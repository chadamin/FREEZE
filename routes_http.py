# routes_http.py
import os, glob, time, datetime
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field
from typing import Optional
from fastapi import Query

from . import runtime as RT
from .routes.ws_app import app_broadcast_json, app_broadcast_binary
from .danger_check import get_vibration_pin

router = APIRouter()

# -----------------------------
# Files (snapshots/clips)
# -----------------------------
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./clips")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def _latest_file(pattern: str) -> Optional[str]:
    paths = glob.glob(os.path.join(OUTPUT_DIR, pattern))
    return max(paths, key=lambda p: os.path.getmtime(p)) if paths else None

# -----------------------------
# Home / last snapshot / clips / status
# -----------------------------
@router.get("/", response_class=HTMLResponse)
def home():
    return """<!doctype html>
<html><head>
  <meta charset="utf-8"/>
  <title>Detect → Snapshot & Video</title>
  <style>
    body{margin:0;background:#111;color:#eee;font-family:system-ui}
    .wrap{padding:24px;line-height:1.6}
    a{color:#8cf;text-decoration:none}
    a:hover{text-decoration:underline}
  </style>
</head>
<body><div class="wrap">
  <h2>Detect → Snapshot & Video (no live streaming)</h2>
  <ul>
    <li><a href="/last_snap.jpg" target="_blank">Last snapshot</a></li>
    <li><a href="/clips" target="_blank">Clips</a></li>
    <li><a href="/status" target="_blank">Status</a></li>
  </ul>
</div></body></html>"""

@router.get("/last_snap.jpg")
def last_snap():
    p = _latest_file("snap_*.jpg")
    if not p or not os.path.isfile(p):
        return Response(status_code=404)
    return FileResponse(p, media_type="image/jpeg", filename=os.path.basename(p))

@router.get("/clips", response_class=HTMLResponse)
def list_clips():
    files = [f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith((".mp4", ".avi", ".jpg"))]
    files.sort(key=lambda n: os.path.getmtime(os.path.join(OUTPUT_DIR, n)))
    rows = []
    for name in files:
        p = os.path.join(OUTPUT_DIR, name)
        sz = os.path.getsize(p)
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(p)))
        rows.append(
            f"<li><a href='/clips/{name}' download>{name}</a> "
            f"<small>({sz/1024/1024:.2f} MB, {ts})</small></li>"
        )
    html = (
        f"<html><body style='background:#111;color:#eee;font-family:system-ui'>"
        f"<div style='padding:24px'><h3>Files in {os.path.abspath(OUTPUT_DIR)}</h3>"
        f"<ul>{''.join(rows) or '<li>No files yet</li>'}</ul></div></body></html>"
    )
    return HTMLResponse(html)

@router.get("/clips/{name}")
def serve_clip(name: str):
    safe = os.path.basename(name)
    path = os.path.join(OUTPUT_DIR, safe)
    if not os.path.isfile(path):
        return Response(status_code=404)
    mt = "video/mp4" if safe.lower().endswith(".mp4") else (
        "video/x-msvideo" if safe.lower().endswith(".avi") else "image/jpeg"
    )
    return FileResponse(path, media_type=mt, filename=safe)

@router.get("/status")
async def status():
    latest_snap = _latest_file("snap_*.jpg")
    files = [os.path.join(OUTPUT_DIR, f) for f in os.listdir(OUTPUT_DIR)]
    clip_count = len([f for f in files if os.path.isfile(f) and f.lower().endswith((".mp4", ".avi"))])
    size_mb = sum(os.path.getsize(f) for f in files if os.path.isfile(f)) / (1024 * 1024)

    async with RT.state_lock:
        return {
            "direction": RT.last_direction, "label": RT.last_group_label,
            "confidence": RT.last_group_conf, "raw_idx": RT.last_raw_idx,
            "raw_label": RT.last_raw_label, "raw_conf": RT.last_raw_conf,
            "rms": RT.last_energy_rms, "dbfs": RT.last_dbfs,
            "transcript": RT.last_transcript, "updated_ms": RT.last_updated_ms,
            "output_dir": os.path.abspath(OUTPUT_DIR),
            "clips_count": clip_count,
            "latest_snapshot": os.path.basename(latest_snap) if latest_snap else None,
            "disk_usage_mb": round(size_mb, 2),
        }

# -----------------------------
# REST: direction / notify
# -----------------------------
class DirectionBody(BaseModel):
    direction: int

class NotifyBody(BaseModel):
    event: str = Field(..., description="'danger' | 'info' | 'heartbeat'")
    direction: int = -1          # optional override from client
    db: float = 0.0
    ms: int = RT.VIBRATE_MS
    pin: Optional[int] = None
    token: Optional[str] = None
    esp_id: Optional[str] = None
    # optional heartbeat status
    device_id: Optional[str] = None
    fsr_value: Optional[int] = None
    fsr_pressed: Optional[bool] = None

def _sanitize_dir(v: int) -> int:
    try:
        x = int(v)
        return x % 360 if x >= 0 else -1
    except Exception:
        return -1

DANGER_WINDOW_MS = getattr(RT, "DANGER_WINDOW_MS", int(os.getenv("DANGER_WINDOW_MS", "3500")))

@router.post("/notify")
async def notify(
    body: NotifyBody,
    verbose: int = Query(0, description="1이면 예전처럼 전체 필드 반환, 기본은 최소 필드(direction,danger,ms)"),
):
    override_dir = _sanitize_dir(body.direction)
    ev = (body.event or "info").lower()

    # ===== HEARTBEAT (응답만, broadcast 없음) =====
    if ev == "heartbeat":
        now = RT.now_ms()
        async with RT.state_lock:
            if override_dir >= 0:
                RT.last_direction = override_dir
                RT.last_updated_ms = now

            is_recent = (now - RT.last_updated_ms) <= DANGER_WINDOW_MS
            is_danger = (RT.last_group_label not in ("safe", "no-audio", "unknown")) and (float(RT.last_group_conf) >= 0.30)

            resp_dir = RT.last_direction if (is_recent and is_danger) else -1
            resp_ms  = RT.VIBRATE_MS if (is_recent and is_danger and resp_dir >= 0) else 0

        if verbose:
            return {"ok": True, "direction": resp_dir, "danger": bool(resp_ms > 0), "ms": resp_ms,
                    "recent": is_recent, "label": RT.last_group_label, "conf": RT.last_group_conf}
        return {"direction": resp_dir, "danger": bool(resp_ms > 0), "ms": resp_ms}

    # ===== REGULAR NOTIFY (danger/info) =====
    async with RT.state_lock:
        if override_dir >= 0:
            RT.last_direction = override_dir
            RT.last_updated_ms = RT.now_ms()

        computed_danger = (
            True if ev == "danger"
            else (RT.last_group_label not in ("safe", "no-audio", "unknown")) and (float(RT.last_group_conf) >= 0.30)
        )

        direction = RT.last_direction
        group_label = RT.last_group_label
        group_conf = RT.last_group_conf
        raw = {"idx": RT.last_raw_idx, "label": RT.last_raw_label, "conf": RT.last_raw_conf}
        dbfs = RT.last_dbfs

    # --- ms=0은 “정지”로 존중 ---
    if body.ms is None:
        desired_ms = RT.VIBRATE_MS
    else:
        desired_ms = int(body.ms)
    ms_to_send = desired_ms if (computed_danger and direction >= 0 and desired_ms > 0) else 0

    pin = body.pin if body.pin is not None else get_vibration_pin(direction)

    # (A) 앱/대시보드 등에는 기존 전체 페이로드 브로드캐스트
    try:
        await RT.broadcast_info(
            direction=direction, group_label=group_label, group_conf=group_conf,
            dbfs=dbfs, ms=ms_to_send, raw=raw, event=body.event, pin=pin,
        )
    except TypeError:
        # runtime.broadcast_info 시그니처가 다를 경우 무시
        pass

    # (B) ESP32로는 최소 JSON만 직접 전송
    try:
        import json
        esp_msg = {
            "t": ev,                                     # "danger" | "info" | "heartbeat"
            "dir": int(direction if direction >= 0 else -1),
            "ms": int(ms_to_send),                       # 0이면 진동 없음
            "pin": int(pin if pin is not None else -1),  # -1이면 기기측 기본 로직
            "dang": bool(ms_to_send > 0 and direction >= 0),
        }
        esp_text = json.dumps(esp_msg, separators=(",", ":"))
        for ws in getattr(RT, "esp32_conns", {}).values():
            try:
                await ws.send_text(esp_text)
            except Exception as e:
                print("ESP32 send error:", e)
    except Exception as e:
        print("ESP32 min send build error:", e)

    # HTTP 응답(ESP32의 heartbeat pull)이 읽는 최소 응답도 유지
    minimal = {"direction": int(direction if direction >= 0 else -1),
               "danger": bool(ms_to_send > 0 and direction >= 0),
               "ms": int(ms_to_send)}
    if verbose:
        return {"ok": True,
                "direction": direction, "group_label": group_label, "group_conf": group_conf,
                "raw": raw, "dbfs": dbfs,
                "danger": minimal["danger"], "ms": minimal["ms"], "pin": pin}
    return minimal


# -----------------------------
# Upload APIs
# -----------------------------
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
os.makedirs(os.path.join(UPLOAD_DIR, "snapshots"), exist_ok=True)
os.makedirs(os.path.join(UPLOAD_DIR, "videos"), exist_ok=True)

@router.post("/api/upload/snapshot")
async def upload_snapshot(
    file: UploadFile = File(...),
    device_id: str = Form("unknown"),
    ts: int = Form(None),
    det_count: int = Form(None),
    classes: str = Form(""),
    conf: str = Form(""),
    trigger_by: str = Form(""),
):
    ts_str = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    save_to = os.path.join(UPLOAD_DIR, "snapshots", f"{device_id}_{ts_str}.jpg")
    with open(save_to, "wb") as f:
        f.write(await file.read())
    return {"ok": True, "path": save_to, "det_count": det_count,
            "classes": classes, "conf": conf, "trigger_by": trigger_by}

@router.post("/api/upload/video")
async def upload_video(
    file: UploadFile = File(...),
    device_id: str = Form("unknown"),
    ts: int = Form(None),
):
    ts_str = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    save_to = os.path.join(UPLOAD_DIR, "videos", f"{device_id}_{ts_str}.mp4")
    with open(save_to, "wb") as f:
        f.write(await file.read())
    return {"ok": True, "path": save_to}
