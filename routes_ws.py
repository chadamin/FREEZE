# routes_ws.py — WS for audio(YAMNet+Whisper), camera(YOLO), and app media push
# (robust, no 'file'/'time' args to broadcast_info, auto-absolute BASE_URL from WS headers)
from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect
import asyncio, json, numpy as np, os, base64, time, cv2, datetime, threading, re, io, wave, uuid, struct, traceback
from collections import deque
from typing import Deque, Tuple, Optional, List, Dict, Set

from yamnet_module import classify_sound_with_confidence
from whisper_module import transcribe_audio_wav_bytes
from danger_check import is_significant_group, get_vibration_pin
from camera_module import CameraEventProcessor
from runtime import (
    DOMAIN_HINT, NAMES_HINTS, WHISPER_ASYNC,
    state_lock, esp32_conns, DEFAULT_ESP32_ID, now_ms,
    broadcast_info, VIBRATE_MS,
    rms_and_dbfs, decode_from_canonical_payload, parse_binary_frame,
    vad_is_speech_int16, WhisperAccumulator, gate_is_speech,
    last_direction, last_group_label, last_group_conf,
    last_raw_idx, last_raw_label, last_raw_conf,
    last_energy_rms, last_dbfs, last_transcript, last_updated_ms,
    set_latest_jpg, pulse, is_pulsing,
)

router = APIRouter()

# ==============================
# Configs
# ==============================
BASE_URL = os.getenv("BASE_URL", "").rstrip("/")
_BASE_URL_DYNAMIC = ""  # auto-detected base url cache
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8n.pt")
OUTPUT_DIR    = os.getenv("OUTPUT_DIR", "./clips")
SNAP_QUALITY  = int(os.getenv("SNAP_QUALITY", "90"))
PRE_ROLL_SEC  = float(os.getenv("PRE_ROLL_SEC", "5.0"))
POST_ROLL_SEC = float(os.getenv("POST_ROLL_SEC", "10.0"))
FALLBACK_FPS  = float(os.getenv("FALLBACK_FPS", "15.0"))
VIDEO_CODEC   = os.getenv("VIDEO_CODEC", "mp4v")
MAX_WIDTH     = int(os.getenv("MAX_WIDTH", "1920"))
INLINE_SNAPSHOT_B64 = os.getenv("INLINE_SNAPSHOT_B64", "1") == "1"
WS_VIDEO_STREAM     = os.getenv("WS_VIDEO_STREAM", "0") == "1"
WS_VIDEO_CHUNK      = int(os.getenv("WS_VIDEO_CHUNK", "65536"))
os.makedirs(OUTPUT_DIR, exist_ok=True)
SAVE_AUDIO   = os.getenv("SAVE_AUDIO", "1") == "1"   # 1이면 저장
SAVE_AUDIO_TS = os.getenv("SAVE_AUDIO_TS", "0") == "1"  # 1이면 매 세그먼트 별도 파일 저장
AUDIO_OUT_DIR = OUTPUT_DIR  # clips 폴더 재사용
SAVE_AUDIO_LEN_SEC = float(os.getenv("SAVE_AUDIO_LEN_SEC", "3.0"))  # 저장할 길이(초)

# 오디오 WS 기본 모드: env(환경음만) | speech(Whisper만) | both(둘다)
AUDIO_WS_DEFAULT_MODE = os.getenv("AUDIO_WS_DEFAULT_MODE", "env").lower()


# ===== Audio RAW frame settings (RAW fallback) =====
RAW_SR = 16000
RAW_FRAME_MS = 200
RAW_SAMPLES = RAW_SR * RAW_FRAME_MS // 1000     # 3200
RAW_FRAME_BYTES = RAW_SAMPLES * 2               # int16 → 6400

SAVE_AUDIO   = os.getenv("SAVE_AUDIO", "1") == "1"
SAVE_AUDIO_TS = os.getenv("SAVE_AUDIO_TS", "0") == "1"
AUDIO_OUT_DIR = OUTPUT_DIR
SAVE_AUDIO_LEN_SEC = float(os.getenv("SAVE_AUDIO_LEN_SEC", "3.0"))

_audio_ring = bytearray()
_audio_ring_sr = RAW_SR  # ✅ 이제 정상
def _ring_cap_bytes(sr: int) -> int:
    return int(SAVE_AUDIO_LEN_SEC * sr) * 2

# ==============================
# Utils
# ==============================
def _effective_base_url() -> str:
    return BASE_URL or _BASE_URL_DYNAMIC or ""

def _file_url(name: str) -> str:
    rel = f"/clips/{name}"
    base = _effective_base_url()
    return (base + rel) if base else rel

def _safe_json(data: dict) -> str:
    try: return json.dumps(data, ensure_ascii=False)
    except Exception as e:
        print("[JSON ERROR]", e); return "{}"

def _log_exc(prefix: str, err: Exception):
    print(f"{prefix}: {err.__class__.__name__}: {err}")
    print(traceback.format_exc())

def draw_border(bgr: np.ndarray, show: bool) -> np.ndarray:
    if not show: return bgr
    out = bgr.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w-1, h-1), (0, 0, 255), 8)
    return out

def _maybe_set_base_url_from_ws(ws: WebSocket):
    """BASE_URL 미설정 시, WebSocket 요청 헤더로 절대 URL 베이스 자동 감지."""
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
        _log_exc("[BASE_URL detect]", e)

processor = CameraEventProcessor(YOLO_MODEL)

# ==============================
# App WebSocket hub (topic → set(ws))
# ==============================
Topic = str
_app_clients: Dict[Topic, Set[WebSocket]] = {}
_app_lock = asyncio.Lock()

async def app_add(topic: Topic, ws: WebSocket):
    async with _app_lock:
        _app_clients.setdefault(topic, set()).add(ws)

async def app_remove(topic: Topic, ws: WebSocket):
    async with _app_lock:
        if topic in _app_clients and ws in _app_clients[topic]:
            _app_clients[topic].remove(ws)
            if not _app_clients[topic]:
                _app_clients.pop(topic, None)

async def _safe_send_text(ws: WebSocket, data: str) -> bool:
    try:
        await ws.send_text(data); return True
    except Exception as e:
        _log_exc("[app_broadcast_json/send_text]", e); return False

async def _safe_send_bytes(ws: WebSocket, b: bytes) -> bool:
    try:
        await ws.send_bytes(b); return True
    except Exception as e:
        _log_exc("[app_broadcast_binary/send_bytes]", e); return False

async def app_broadcast_json(topic: Topic, payload: dict):
    try:
        async with _app_lock:
            targets = list(_app_clients.get(topic, set()))
        data = _safe_json(payload)
        dead = []
        for ws in targets:
            if not await _safe_send_text(ws, data):
                dead.append(ws)
        for ws in dead:
            await app_remove(topic, ws)
    except Exception as e:
        _log_exc("[app_broadcast_json]", e)

async def app_broadcast_binary(topic: Topic, b: bytes):
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
        _log_exc("[app_broadcast_binary]", e)

# 바이너리 프레임 헤더: b"MED0" + 16바이트 uuid + uint32 seq + uint8 kind(1=jpeg,2=mp4)
_BIN_MAGIC = b"MED0"
def _pack_media_header(transfer_id: uuid.UUID, seq: int, kind: int) -> bytes:
    try:
        return _BIN_MAGIC + transfer_id.bytes + struct.pack(">IB", seq, kind)
    except Exception as e:
        _log_exc("[_pack_media_header]", e)
        return _BIN_MAGIC + (b"\x00"*16) + struct.pack(">IB", 0, kind)

# ==============================
# Pre-roll buffer (camera)
# ==============================
_prebuf: Deque[Tuple[int, bytes]] = deque()
_prelock = threading.Lock()

def _push_preroll(ts_ms: int, jpg: bytes):
    try:
        with _prelock:
            _prebuf.append((ts_ms, jpg))
            cutoff = ts_ms - int(PRE_ROLL_SEC * 1000)
            while _prebuf and _prebuf[0][0] < cutoff:
                _prebuf.popleft()
    except Exception as e:
        _log_exc("[_push_preroll]", e)

# ==============================
# Snapshot & Recording helpers
# ==============================
def _save_snapshot(jpg_annotated: bytes) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    name = f"snap_{ts}.jpg"
    path = os.path.join(OUTPUT_DIR, name)
    try:
        with open(path, "wb") as f:
            f.write(jpg_annotated)
        print(f"[SNAP] {path}")
        return path, name, int(time.time())
    except Exception as e:
        _log_exc("[SNAP] save error", e)
        return None, None, None

async def _publish_snapshot(topic: Topic, jpg_bytes: bytes,
                            file_path: Optional[str], file_name: Optional[str], ts_sec: Optional[int]):
    try:
        files = {}
        if file_name:
            files["snapshot_url"] = _file_url(file_name)

        payload = {
            "type": "snapshot",
            "source": "yolo",
            "ts": now_ms(),
            "file": (_file_url(file_name) if file_name else None),
            "time": (ts_sec if ts_sec is not None else int(time.time())),
            "direction": last_direction,
            "group_label": last_group_label,
            "group_conf": last_group_conf,
            "dbfs": last_dbfs,
            "raw": {"idx": last_raw_idx, "label": last_raw_label, "conf": last_raw_conf},
            "transcript": last_transcript,
            "files": files,
        }
        await app_broadcast_json(topic, payload)

        if INLINE_SNAPSHOT_B64 and jpg_bytes:
            b64 = base64.b64encode(jpg_bytes).decode("ascii")
            await app_broadcast_json(topic, {
                "type": "snapshot_inline",
                "mime": "image/jpeg",
                "data": f"data:image/jpeg;base64,{b64}",
                "ts": now_ms(),
                "transfer_id": str(uuid.uuid4()),
            })
    except Exception as e:
        _log_exc("[_publish_snapshot]", e)

class Recorder:
    def __init__(self, out_dir: str, topic: Topic):
        self.out_dir = out_dir
        self.topic = topic
        self.recording = False
        self._writer: Optional[cv2.VideoWriter] = None
        self._size: Optional[Tuple[int, int]] = None
        self._fps: float = FALLBACK_FPS
        self._deadline: float = 0.0
        self._path: Optional[str] = None
        self._lock = threading.Lock()

    def _estimate_fps(self, ts_list: List[int]) -> float:
        if len(ts_list) < 2: return FALLBACK_FPS
        dt = (ts_list[-1] - ts_list[0]) / max(1, (len(ts_list) - 1))
        fps = 1000.0 / max(dt, 1.0)
        return float(max(5.0, min(30.0, fps)))

    def _ensure_writer(self, w: int, h: int):
        if self._writer is not None: return
        fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
        self._writer = cv2.VideoWriter(self._path, fourcc, self._fps, (w, h))
        self._size = (w, h)

    def start(self, preroll: List[Tuple[int, bytes]]):
        with self._lock:
            if self.recording: return
            ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            self._path = os.path.join(self.out_dir, f"clip_{ts}.mp4")
            ts_ms = [t for (t, _) in preroll][-30:]
            self._fps = self._estimate_fps(ts_ms)
            self._writer = None
            self._size = None
            self._deadline = time.time() + POST_ROLL_SEC
            self.recording = True
            print(f"[REC] start {self._path}")

            name = os.path.basename(self._path)
            video_url = _file_url(name)
            epoch_sec = int(time.time())

            asyncio.create_task(broadcast_info(
                direction=last_direction, group_label=last_group_label,
                group_conf=last_group_conf, dbfs=last_dbfs,
                raw={"idx": last_raw_idx, "label": last_raw_label, "conf": last_raw_conf},
                transcript=last_transcript,
                event="recording_started", source="yolo",
                files={"video_url": video_url}
            ))

            asyncio.create_task(app_broadcast_json(self.topic, {
                "type": "recording_started", "source": "yolo", "ts": now_ms(),
                "file": video_url, "time": epoch_sec,
                "files": {"video_url": video_url},
                "direction": last_direction, "group_label": last_group_label,
                "group_conf": last_group_conf, "dbfs": last_dbfs,
                "raw": {"idx": last_raw_idx, "label": last_raw_label, "conf": last_raw_conf},
                "transcript": last_transcript
            }))

            for _, jpg in preroll:
                try:
                    frm = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                    if frm is None: continue
                    h, w = frm.shape[:2]
                    if w > MAX_WIDTH:
                        scale = MAX_WIDTH / float(w)
                        frm = cv2.resize(frm, (int(w*scale), int(h*scale)))
                        h, w = frm.shape[:2]
                    self._ensure_writer(w, h)
                    if self._size != (w, h):
                        frm = cv2.resize(frm, self._size)
                    self._writer.write(frm)
                except Exception as e:
                    _log_exc("[REC preroll write]", e)

    def write(self, bgr: np.ndarray):
        with self._lock:
            if not self.recording: return
            try:
                h, w = bgr.shape[:2]
                if w > MAX_WIDTH:
                    scale = MAX_WIDTH / float(w)
                    bgr = cv2.resize(bgr, (int(w*scale), int(h*scale)))
                    h, w = bgr.shape[:2]
                self._ensure_writer(w, h)
                if self._size != (w, h):
                    bgr = cv2.resize(bgr, self._size)
                self._writer.write(bgr)
            except Exception as e:
                _log_exc("[REC write]", e)

    def maybe_stop(self):
        path = None
        with self._lock:
            if self.recording and time.time() > self._deadline:
                path = self._path
                try:
                    if self._writer is not None: self._writer.release()
                except Exception as e:
                    _log_exc("[REC release]", e)
                finally:
                    print(f"[REC] stop  {self._path}")
                    self._writer = None; self._size = None
                    self._path = None; self.recording = False
        if path:
            try:
                name = os.path.basename(path)
                video_url = _file_url(name)
                epoch_sec = int(time.time())

                asyncio.create_task(broadcast_info(
                    direction=last_direction, group_label="yolo_recording_done",
                    group_conf=1.0, dbfs=last_dbfs, ms=0, event="info",
                    files={"video_url": video_url}, source="yolo"
                ))

                asyncio.create_task(app_broadcast_json(self.topic, {
                    "type": "recording_done", "source": "yolo", "ts": now_ms(),
                    "file": video_url, "time": epoch_sec,
                    "files": {"video_url": video_url}
                }))

                if WS_VIDEO_STREAM:
                    asyncio.create_task(_stream_video_to_app(self.topic, os.path.join(OUTPUT_DIR, name)))
            except Exception as e:
                _log_exc("[REC maybe_stop notify]", e)

async def _stream_video_to_app(topic: Topic, path: str):
    tid = uuid.uuid4()
    try:
        await app_broadcast_json(topic, {
            "type": "media_start", "transfer_id": str(tid),
            "kind": "video", "mime": "video/mp4", "n_chunks": -1
        })
        seq = 0
        with open(path, "rb") as f:
            while True:
                chunk = f.read(WS_VIDEO_CHUNK)
                if not chunk: break
                hdr = _pack_media_header(tid, seq, 2)
                await app_broadcast_binary(topic, hdr + chunk)
                seq += 1
        await app_broadcast_json(topic, {
            "type": "media_end", "transfer_id": str(tid), "received_chunks": seq
        })
    except Exception as e:
        _log_exc("[_stream_video_to_app]", e)
        await app_broadcast_json(topic, {
            "type": "media_error", "transfer_id": str(tid), "error": str(e)
        })

# ==============================
# Focus window for Whisper
# ==============================
FOCUS_HOTWORDS = [s.strip() for s in os.getenv("FOCUS_HOTWORDS", "").split(",") if s.strip()]
FOCUS_WINDOW_MS = int(os.getenv("FOCUS_WINDOW_MS", "5000"))
_focus_until_ms = 0
_focus_lock = asyncio.Lock()

def _hit_hotword(text: str) -> bool:
    if not text or not FOCUS_HOTWORDS: return False
    for hw in FOCUS_HOTWORDS:
        if hw and re.search(re.escape(hw), text, re.IGNORECASE): return True
    return False

def _wav_dur_sec(b: bytes) -> float:
    try:
        with wave.open(io.BytesIO(b), "rb") as wf:
            frames = wf.getnframes(); sr = wf.getframerate()
            return (frames / float(sr)) if sr > 0 else 0.0
    except Exception:
        return 0.0

# ==============================
# ESP32 WebSocket (heartbeat)
# ==============================
@router.websocket("/ws/esp32")
async def ws_esp32(websocket: WebSocket):
    try:
        await websocket.accept()
    except Exception as e:
        _log_exc("[ESP32 accept]", e); return

    _maybe_set_base_url_from_ws(websocket)
    esp_id = websocket.query_params.get("id", DEFAULT_ESP32_ID)
    esp32_conns[esp_id] = websocket
    print(f"🔌 ESP32 connected: {esp_id} (total={len(esp32_conns)})")

    async def _recv_loop():
        while True:
            try:
                _ = await asyncio.wait_for(websocket.receive_text(), timeout=60)
            except asyncio.TimeoutError:
                continue
            except Exception:
                break

    async def _ping_loop():
        while True:
            try:
                await asyncio.sleep(15)
                await websocket.send_text('{"t":"ping"}')
            except Exception:
                break

    recv_task = asyncio.create_task(_recv_loop())
    ping_task = asyncio.create_task(_ping_loop())
    try:
        await asyncio.wait({recv_task, ping_task}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for t in (recv_task, ping_task):
            try: t.cancel()
            except: pass
        try:
            if esp32_conns.get(esp_id) is websocket:
                esp32_conns.pop(esp_id, None)
            await websocket.close()
        except Exception:
            pass
        print(f"🔌 ESP32 disconnected: {esp_id} (total={len(esp32_conns)})")

# ==============================
# App WebSocket (subscribe by topic)
# ==============================
@router.websocket("/ws/app")
async def ws_app(ws: WebSocket):
    try:
        await ws.accept()
    except Exception as e:
        _log_exc("[APP accept]", e); return

    _maybe_set_base_url_from_ws(ws)
    topic = ws.query_params.get("topic", "public")
    await app_add(topic, ws)
    print(f"📱 app connected topic={topic}")
    try:
        while True:
            try:
                _ = await asyncio.wait_for(ws.receive_text(), timeout=60)
            except asyncio.TimeoutError:
                continue
            except WebSocketDisconnect:
                break
            except Exception as e:
                _log_exc("[APP recv]", e); break
    finally:
        try:
            await app_remove(topic, ws)
            await ws.close()
        except Exception:
            pass
        print(f"📱 app disconnected topic={topic}")

# ==============================
# Audio WebSocket (Canonical + RAW fallback)
# ==============================
def _dbfs_int16(x: np.ndarray) -> float:
    if x.size == 0: return float("-inf")
    peak = np.max(np.abs(x))
    return float("-inf") if peak == 0 else 20.0 * np.log10(peak / 32767.0)

_last_ring_at_ms = 0

def _ring_append_int16(int16_mono: np.ndarray, sr: int, frame_ms: int = 200):
    """수신 지연이 크면 그만큼 무음(0)으로 패딩한 뒤 링버퍼에 붙인다."""
    global _audio_ring, _audio_ring_sr, _last_ring_at_ms
    if int16_mono is None or int16_mono.size == 0:
        return
    _audio_ring_sr = int(sr or _audio_ring_sr or RAW_SR)

    now_ms = int(time.time() * 1000)

    # 직전 append 이후 지연이 프레임 길이의 1.5배 이상이면 무음으로 메꿈
    if _last_ring_at_ms > 0:
        dt = now_ms - _last_ring_at_ms
        expect = frame_ms
        if dt > int(expect * 1.5):
            missed = max(1, int(round(dt / expect)) - 1)
            pad_bytes = missed * (_audio_ring_sr * frame_ms // 1000) * 2  # int16 mono
            _audio_ring.extend(b"\x00" * pad_bytes)

    # 실제 프레임 추가
    _audio_ring.extend(int16_mono.astype(np.int16).tobytes())
    _last_ring_at_ms = now_ms

    # 링 용량 유지
    cap = _ring_cap_bytes(_audio_ring_sr)
    if len(_audio_ring) > cap:
        _audio_ring = _audio_ring[-cap:]

@router.websocket("/ws/audio")
async def audio_ws(websocket: WebSocket):
    global _focus_until_ms
    try:
        await websocket.accept()
        mode = websocket.query_params.get("mode", AUDIO_WS_DEFAULT_MODE).lower()
        DO_WHISPER = mode in ("speech", "both")
        DO_YAMNET  = mode in ("env", "both")
        print(f"🎧 /ws/audio connected mode={mode} (whisper={DO_WHISPER}, yamnet={DO_YAMNET})")

    except Exception as e:
        _log_exc("[AUDIO accept]", e); return

    _maybe_set_base_url_from_ws(websocket)
    print("🎧 /ws/audio connected")

    acc = WhisperAccumulator()
    bbuf = bytearray()  # RAW 모드 수신 시 6400B로 자르기

    async def run_whisper_once(wav_bytes: bytes):
        global _focus_until_ms, last_transcript, last_updated_ms
        try:
            phrase_boost = list(set(NAMES_HINTS + FOCUS_HOTWORDS))
            res = await transcribe_audio_wav_bytes(
                wav_bytes, lang="ko", initial_prompt=DOMAIN_HINT, phrase_hints=phrase_boost
            ) or {"text": "", "hits": []}
            text = res.get("text", "") if isinstance(res, dict) else str(res or "")
            hits = list(res.get("hits", []) or []) if isinstance(res, dict) else []
        except Exception as e:
            _log_exc("[whisper error]", e); text, hits = "", []

        if hits: print(f"[WHISPER][HIT] {hits}")

        try:
            now = now_ms(); opened = False
            if text and _hit_hotword(text):
                async with _focus_lock:
                    _focus_until_ms = max(_focus_until_ms, now + FOCUS_WINDOW_MS)
                    opened = True

            allow = now <= _focus_until_ms
            if text and allow:
                async with state_lock:
                    last_transcript = text; last_updated_ms = now
                await broadcast_info(
                    direction=last_direction, group_label=last_group_label,
                    group_conf=last_group_conf, dbfs=last_dbfs,
                    raw={"idx": last_raw_idx, "label": last_raw_label, "conf": last_raw_conf},
                    transcript=text, event=("focus_open" if opened else "transcript"),
                    source="whisper"
                )
            if hits:
                await broadcast_info(
                    direction=last_direction, group_label=last_group_label,
                    group_conf=last_group_conf, dbfs=last_dbfs,
                    raw={"idx": last_raw_idx, "label": last_raw_label, "conf": last_raw_conf},
                    transcript=text, event="whisper_hit", source="whisper"
                )
        except Exception as e:
            _log_exc("[whisper broadcast]", e)

    async def _process_waveform(waveform: np.ndarray, sr: int, dir_in: int = -1):
        
        # === ADD END ===
        _ring_append_int16(waveform, int(sr or RAW_SR), frame_ms=RAW_FRAME_MS)

        dbfs = -120.0; rms = 0.0
        group_label = "no-audio"; group_conf = 0.0
        raw_idx = -1; raw_label = ""; raw_conf = 0.0

        if waveform is not None and getattr(waveform, "size", 0) > 0:
            try:
                rms, dbfs = rms_and_dbfs(waveform)
                result = await asyncio.to_thread(classify_sound_with_confidence, waveform, sr)
                group_label = result.get("group_label", "unknown")
                group_conf = float(result.get("group_conf", 0.0))
                raw_idx = int(result.get("raw_idx", -1))
                raw_label = str(result.get("raw_label", ""))
                raw_conf = float(result.get("raw_conf", 0.0))
            except Exception as e:
                _log_exc("[YAMNET error]", e)

        try:
            dir_norm = (int(dir_in) % 360) if int(dir_in) >= 0 else -1
        except Exception:
            dir_norm = -1

        # 상태 업데이트
        global last_direction, last_group_label, last_group_conf, last_raw_idx
        global last_raw_label, last_raw_conf, last_energy_rms, last_dbfs, last_updated_ms
        try:
            async with state_lock:
                if 0 <= dir_norm < 360: last_direction = dir_norm
                last_group_label = group_label; last_group_conf = group_conf
                last_raw_idx = raw_idx; last_raw_label = raw_label; last_raw_conf = raw_conf
                last_energy_rms = rms; last_dbfs = dbfs; last_updated_ms = now_ms()
        except Exception as e:
            _log_exc("[AUDIO state_lock]", e)

        # VAD & gate & Whisper
        try:
            vad_ok = False
            if waveform is not None and getattr(waveform, "size", 0) > 0:
                vad_ok = vad_is_speech_int16(waveform.astype(np.int16).tobytes(), sr)

            if vad_ok and gate_is_speech(dbfs, raw_label, raw_conf, group_label, group_conf):
                acc.add(waveform, sr)
                if acc.ready():
                    wav_for_whisper = acc.flush_wav()
                    dur = _wav_dur_sec(wav_for_whisper) if wav_for_whisper else 0.0

                    # === CHANGE: 링버퍼에서 SAVE_AUDIO_LEN_SEC(기본 3s) 길이로 저장 ===
                    if SAVE_AUDIO:
                        try:
                            # 링버퍼가 비었으면(초기 등) 기존 세그먼트라도 저장
                            if len(_audio_ring) == 0 and wav_for_whisper:
                                out_bytes = wav_for_whisper
                                out_sr = int(sr or _audio_ring_sr or RAW_SR)
                            else:
                                out_sr = int(_audio_ring_sr or RAW_SR)
                                cap = _ring_cap_bytes(out_sr)
                                ring = bytes(_audio_ring[-cap:]) if len(_audio_ring) >= cap else bytes(_audio_ring)
                                bio = io.BytesIO()
                                with wave.open(bio, "wb") as w:
                                    w.setnchannels(1)
                                    w.setsampwidth(2)
                                    w.setframerate(out_sr)
                                    w.writeframes(ring)
                                out_bytes = bio.getvalue()

                            last_path = os.path.join(AUDIO_OUT_DIR, "last_in.wav")
                            with open(last_path, "wb") as f:
                                f.write(out_bytes)

                            if SAVE_AUDIO_TS:
                                ts_name = f"in_{int(time.time()*1000)}.wav"
                                with open(os.path.join(AUDIO_OUT_DIR, ts_name), "wb") as f:
                                    f.write(out_bytes)

                            print(f"[AUDIO][SAVE] {last_path} (target≈{SAVE_AUDIO_LEN_SEC:.1f}s, sr={out_sr})")
                        except Exception as e:
                            _log_exc("[AUDIO save wav via ring]", e)
                    # === CHANGE END ===


                    if wav_for_whisper and dur >= 0.8:
                        if WHISPER_ASYNC: asyncio.create_task(run_whisper_once(wav_for_whisper))
                        else: await run_whisper_once(wav_for_whisper)

        except Exception as e:
            _log_exc("[AUDIO whisper path]", e)

        # 위험/정보 broadcast
        try:
            significant = is_significant_group(group_label, group_conf, dbfs)
            pin = get_vibration_pin(last_direction) if significant else None
            await broadcast_info(
                direction=last_direction, group_label=group_label,
                group_conf=group_conf, dbfs=dbfs,
                ms=(VIBRATE_MS if significant else 0),
                raw={"idx": raw_idx, "label": raw_label, "conf": raw_conf},
                event=("danger" if significant else "info"), pin=pin, source="yamnet"
            )
        except Exception as e:
            _log_exc("[AUDIO broadcast_info]", e)

    try:
        while True:
            try:
                msg = await websocket.receive()
            except WebSocketDisconnect:
                break
            except Exception as e:
                _log_exc("[AUDIO receive]", e); break

            waveform = None; sr = None; dir_in = -1

            # ─── 1) JSON(Canonical) ───
            if msg.get("text"):
                try:
                    data = json.loads(msg["text"])
                except Exception:
                    continue
                if "pcm_b64" in data or "audio_b64" in data:
                    try:
                        dir_val = data.get("direction", data.get("dir", -1))
                        try: dir_in = int(dir_val)
                        except: dir_in = -1
                        waveform, sr, ch, wav_bytes = decode_from_canonical_payload(data)
                        # canonical이 float32일 수도 있으니 int16 범위로 클램프
                        if waveform.dtype != np.int16:
                            wf = np.clip(waveform, -1.0, 1.0) if waveform.dtype != np.int16 else waveform
                            waveform = (wf * 32768.0).astype(np.int16) if wf.dtype != np.int16 else wf
                        print(f"[AUDIO][JSON] dir_in={dir_in} sr={sr}")
                    except Exception as e:
                        _log_exc("[AUDIO parse canonical JSON]", e)
                        waveform = None

            # ─── 2) Binary(Canonical with header) or RAW fallback ───
            elif msg.get("bytes"):
                b = msg["bytes"]
                # 우선 canonical 바이너리 헤더 파서 시도
                parsed = None
                try:
                    parsed = parse_binary_frame(b)  # (waveform, sr, ch, wav_bytes, seq, ts_ms, dir_in, flags)
                except Exception:
                    parsed = None

                if parsed:
                    try:
                        (wf, sr, ch, wav_bytes, seq, ts_ms, dir_in, flags) = parsed
                        if wf.dtype != np.int16:
                            wf = np.clip(wf, -1.0, 1.0) if wf.dtype != np.int16 else wf
                            wf = (wf * 32768.0).astype(np.int16) if wf.dtype != np.int16 else wf
                        waveform = wf
                        print(f"[AUDIO][BIN] canonical dir_in={dir_in} sr={sr} ch={ch}")
                    except Exception as e:
                        _log_exc("[AUDIO canonical BIN adapt]", e)
                        waveform = None
                else:
                    # ---- RAW fallback: 16k/mono/int16, 200ms=6400B 고정 프레임 ----
                    bbuf.extend(b)
                    while len(bbuf) >= RAW_FRAME_BYTES:
                        frame_bytes = bbuf[:RAW_FRAME_BYTES]
                        del bbuf[:RAW_FRAME_BYTES]

                        if len(frame_bytes) != RAW_FRAME_BYTES or (len(frame_bytes) % 2) != 0:
                            print(f"[AUDIO][RAW] drop (unaligned) len={len(frame_bytes)}")
                            continue

                        wf = np.frombuffer(frame_bytes, dtype=np.int16)

                        # ➕ 여기 한 줄 추가 (지터 스무딩 + 링 누적)
                        _ring_append_int16(wf, RAW_SR, frame_ms=RAW_FRAME_MS)

                        # 기존 처리 유지
                        await _process_waveform(wf, RAW_SR, dir_in=-1)

                    # RAW는 위에서 모두 처리했으므로 다음 반복
                    continue


            # 최종 처리(1프레임/버퍼 기준)
            if waveform is not None and getattr(waveform, "size", 0) > 0:
                await _process_waveform(waveform, int(sr or RAW_SR), dir_in)

    finally:
        try: await websocket.close()
        except Exception: pass
        print("🎧 /ws/audio disconnected")

# ==============================
# Camera WebSocket
# ==============================
@router.websocket("/ws/camera")
async def ws_camera(ws: WebSocket):
    try:
        await ws.accept()
    except Exception as e:
        _log_exc("[CAMERA accept]", e); return

    _maybe_set_base_url_from_ws(ws)
    print("📷 /ws/camera connected")
    topic = ws.query_params.get("topic", "public")

    async def handle_frame(jpg_bytes: bytes) -> bool:
        ts_ms = now_ms()

        def work():
            try:
                bgr = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)
                if bgr is None: return None, False, None, None
                annotated, triggered = processor.process_frame_and_get_annotated(bgr)
                show_border = is_pulsing() or triggered
                out = draw_border(annotated, show_border)
                ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                snap_ok, snap_buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), SNAP_QUALITY])
                return (out if ok else annotated), triggered, (buf.tobytes() if ok else None), (snap_buf.tobytes() if snap_ok else None)
            except Exception as e:
                _log_exc("[CAM work]", e); return None, False, None, None

        out_bgr, triggered, latest_jpg, snap_jpg = await asyncio.to_thread(work)
        if out_bgr is None: return False

        try:
            if latest_jpg: set_latest_jpg(latest_jpg)
            _push_preroll(ts_ms, jpg_bytes)
        except Exception as e:
            _log_exc("[CAM post-work]", e)

        if triggered:
            try:
                pulse(1200.0)
            except Exception as e:
                _log_exc("[pulse]", e)
            # snapshot
            try:
                if snap_jpg:
                    snap_path, snap_name, ts_sec = _save_snapshot(snap_jpg)
                    await _publish_snapshot(topic, snap_jpg, snap_path, snap_name, ts_sec)
                    if snap_name and ts_sec:
                        snap_url = _file_url(snap_name)
                        asyncio.create_task(broadcast_info(
                            direction=last_direction, group_label=last_group_label,
                            group_conf=last_group_conf, dbfs=last_dbfs,
                            raw={"idx": last_raw_idx, "label": last_raw_label, "conf": last_raw_conf},
                            event="snapshot", source="yolo",
                            files={"snapshot_url": snap_url}
                        ))
            except Exception as e:
                _log_exc("[CAM snapshot/broadcast]", e)
            # start recording
            try:
                if not recorder.recording:
                    with _prelock:
                        preroll = list(_prebuf)
                    recorder.start(preroll)
            except Exception as e:
                _log_exc("[CAM recorder.start]", e)

        try:
            if recorder.recording:
                recorder.write(out_bgr)
                recorder.maybe_stop()
        except Exception as e:
            _log_exc("[CAM recorder.write/stop]", e)

        return True

    # lazy recorder
    global recorder
    try:
        recorder
    except NameError:
        recorder = Recorder(OUTPUT_DIR, topic)

    try:
        while True:
            try:
                msg = await ws.receive()
            except WebSocketDisconnect:
                break
            except Exception as e:
                _log_exc("[CAM receive]", e); break

            try:
                if msg.get("bytes"):
                    await handle_frame(msg["bytes"])
                elif msg.get("text"):
                    try:
                        data = json.loads(msg["text"])
                        if isinstance(data, dict) and isinstance(data.get("frame_b64"), str):
                            b64 = data["frame_b64"].split(",", 1)[-1]
                            jpg = base64.b64decode(b64 + ("=" * ((-len(b64)) % 4)))
                            await handle_frame(jpg)
                    except Exception as e:
                        _log_exc("[CAM parse text]", e); continue
            except Exception as e:
                _log_exc("[CAM handle_frame]", e); continue
    finally:
        try: await ws.close()
        except Exception: pass
        print("📷 /ws/camera disconnected")
