# freeze/runtime.py — ANGLE vibrate on danger (edge+cooldown guards), optional ESP mirroring
import os, io, wave, base64, struct, time, asyncio, json, threading, traceback
from typing import Optional, Tuple, Dict, List
import numpy as np
from fastapi import WebSocket

# ==============================
# 런타임 식별자
# ==============================
RUNTIME_BUILD = "angle-v1.6-guarded+topic+compat"
print(f"[runtime] loaded {RUNTIME_BUILD}")

# ==============================
# 설정(환경변수)
# ==============================
VIBRATE_MS            = int(os.getenv("VIBRATE_MS", "1500"))
VIBRATE_DEBOUNCE_MS   = int(os.getenv("VIBRATE_DEBOUNCE_MS", "1200"))
DISABLE_DEBOUNCE      = os.getenv("DISABLE_DEBOUNCE", "false").lower() == "true"

DEFAULT_ESP32_ID      = os.getenv("DEFAULT_ESP32_ID", "esp32-01")

# ✅ 기본은 이벤트 미러링 끔(원하면 1로 켜기)
MIRROR_EVENTS_TO_ESP  = os.getenv("MIRROR_EVENTS_TO_ESP", "0") in ("1","true","yes")

# 🔧 ESP 호환키(direction/deg) 동시 송신 여부(기본 끔)
ESP_COMPAT_KEYS       = os.getenv("ESP_COMPAT_KEYS", "0") in ("1","true","yes")

AUDIO_SLICE_SEC       = float(os.getenv("AUDIO_SLICE_SEC", "0.5"))
WHISPER_INTERVAL      = float(os.getenv("WHISPER_INTERVAL", "3.0"))
OVERLAP_SEC           = float(os.getenv("OVERLAP_SEC", "0.3"))
DBFS_GATE             = float(os.getenv("DBFS_GATE", "-60.0"))
SPEECH_GATE_CONF      = float(os.getenv("SPEECH_GATE_CONF", "0.20"))
SPEECH_RAW_GATE_CONF  = float(os.getenv("SPEECH_RAW_GATE_CONF", "0.30"))
WHISPER_ASYNC         = os.getenv("WHISPER_ASYNC", "true").lower() == "true"
DEFAULT_SR            = int(os.getenv("DEFAULT_SR", "16000"))
DOMAIN_HINT: Optional[str] = os.getenv("DOMAIN_HINT", None)
NAMES_HINTS = [s.strip() for s in os.getenv("NAMES_HINTS", "").split(",") if s.strip()]

# App hub topic (앱이 /ws/app?topic=... 과 동일)
WS_TOPIC              = os.getenv("WS_TOPIC", "public")

# 각도 미확정/비정상일 때 쓸 기본 각도(도)
ANGLE_FALLBACK        = int(os.getenv("ANGLE_FALLBACK", "0"))  # 0=정면 가정

# 🔒 danger 가드(연속 울림 방지)
DANGER_COOLDOWN_MS    = int(os.getenv("DANGER_COOLDOWN_MS", "3000"))
EDGE_TRIGGER_DANGER   = os.getenv("EDGE_TRIGGER_DANGER", "1").lower() in ("1","true","yes")
DANGER_MIN_CONF       = float(os.getenv("DANGER_MIN_CONF", "0.0"))
ANGLE_BUCKET_DEG      = int(os.getenv("ANGLE_BUCKET_DEG", "30"))

# 텐서플로우/야믿넷 로그 억제(선택)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ==============================
# 전역 상태/락
# ==============================
state_lock = asyncio.Lock()
send_lock  = asyncio.Lock()

esp32_conns: Dict[str, WebSocket] = {}  # 연결된 ESP32

# 최신 상태(오디오/환경/텍스트)
last_direction: int     = -1
last_group_label: str   = ""
last_group_conf: float  = 0.0
last_raw_label: str     = ""
last_raw_idx: int       = -1
last_raw_conf: float    = 0.0
last_energy_rms: float  = 0.0
last_dbfs: float        = -120.0
last_transcript: str    = ""
last_updated_ms: int    = 0

def now_ms() -> int:
    return int(time.time() * 1000)


# 숨길 로그 태그들(쉼표로 여러 개 가능). 기본값: [DEBUG] 만 숨김
SUPPRESS_LOG_TAGS = [s.strip() for s in os.getenv("SUPPRESS_LOG_TAGS", "[DEBUG]").split(",") if s.strip()]

# ==============================
# 카메라 스트리밍 상태
# ==============================
_latest_jpg: bytes = b""
_latest_lock = threading.Lock()
_pulse_until: float = 0.0

def set_latest_jpg(jpg: bytes) -> None:
    global _latest_jpg
    with _latest_lock:
        _latest_jpg = jpg

def get_latest_jpg() -> bytes:
    with _latest_lock:
        return _latest_jpg

def pulse(ms: float = 1200.0) -> None:
    global _pulse_until
    _pulse_until = time.time() + (ms / 1000.0)

def is_pulsing() -> bool:
    return time.time() < _pulse_until

# ==============================
# 유틸 (오디오/프로토콜)
# ==============================
def rms_and_dbfs(waveform: np.ndarray) -> Tuple[float, float]:
    if waveform.size == 0:
        return 0.0, -120.0
    rms = float(np.sqrt(np.mean(waveform.astype(np.float32) ** 2)))
    dbfs = -120.0 if rms == 0 else 20.0 * np.log10(rms / 32768.0)
    return rms, dbfs

def wrap_pcm16_to_wav(pcm_bytes: bytes, sr: int, channels: int = 1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()

BIN_MAGIC = b'AIOD'
BIN_HEADER_FMT = "<4s I I h B B"
BIN_HEADER_SIZE = struct.calcsize(BIN_HEADER_FMT)

def parse_binary_frame(b: bytes, sr: int = DEFAULT_SR):
    if len(b) < BIN_HEADER_SIZE:
        raise ValueError(f"short frame: {len(b)} < {BIN_HEADER_SIZE}")
    magic, seq, ts_ms, direction, flags, ch = struct.unpack(BIN_HEADER_FMT, b[:BIN_HEADER_SIZE])
    if magic != BIN_MAGIC:
        raise ValueError(f"bad magic: {magic!r}")
    pcm = b[BIN_HEADER_SIZE:]
    waveform = np.frombuffer(pcm, dtype=np.int16)
    if ch > 1 and waveform.size > 0:
        try:
            waveform = waveform.reshape(-1, ch)[:, 0]
        except Exception:
            wf2 = waveform[: (waveform.size // ch) * ch].reshape(-1, ch)
            waveform = wf2[:, 0]
    wav_bytes = wrap_pcm16_to_wav(waveform.tobytes(), sr, channels=1)
    return waveform, sr, ch, wav_bytes, int(seq), int(ts_ms), int(direction), int(flags)

def _b64fix(s: str) -> str:
    s = s.strip()
    if s.startswith("data:"):
        s = s.split(",", 1)[1]
    pad = (-len(s)) % 4
    if pad:
        s += "=" * pad
    return s

def decode_from_canonical_payload(data: dict):
    if isinstance(data.get("pcm_b64"), str) and data["pcm_b64"]:
        pcm = base64.b64decode(_b64fix(data["pcm_b64"]), validate=False)
        sr  = int(data.get("sr", DEFAULT_SR))
        ch  = int(data.get("ch", 1))
        audio = np.frombuffer(pcm, dtype=np.int16)
        if ch > 1 and audio.size > 0:
            audio = audio.reshape(-1, ch)[:, 0]
        return audio, sr, 1, wrap_pcm16_to_wav(audio.tobytes(), sr, channels=1)

    if isinstance(data.get("audio_b64"), str) and data["audio_b64"]:
        wav = base64.b64decode(_b64fix(data["audio_b64"]), validate=False)
        with wave.open(io.BytesIO(wav), "rb") as wf:
            sr = wf.getframerate(); ch = wf.getnchannels()
            frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16)
        if ch > 1 and audio.size > 0:
            audio = audio.reshape(-1, ch)[:, 0]
        return audio, sr, ch, wav

    raise ValueError("invalid payload: require 'pcm_b64' or 'audio_b64'")

# ==============================
# WebRTC VAD (16000Hz만 지원)
# ==============================
USE_WEBRTCVAD = os.getenv("USE_WEBRTCVAD", "true").lower() == "true"
VAD_MODE = int(os.getenv("VAD_MODE", "2"))
try:
    import webrtcvad
    _vad = webrtcvad.Vad(VAD_MODE)
except Exception:
    _vad = None
    USE_WEBRTCVAD = False

def vad_is_speech_int16(pcm: bytes, sr: int) -> bool:
    if not USE_WEBRTCVAD or _vad is None or sr != 16000:
        return True
    frame_ms = 20
    frame_bytes = int(sr * (frame_ms/1000.0)) * 2
    if len(pcm) < frame_bytes:
        return False
    speech_frames = 0
    total = 0
    for i in range(0, len(pcm) - frame_bytes + 1, frame_bytes):
        chunk = pcm[i:i+frame_bytes]
        if _vad.is_speech(chunk, sr):
            speech_frames += 1
        total += 1
    return (total > 0) and (speech_frames / total >= 0.5)

# ==============================
# Whisper 누적
# ==============================
class WhisperAccumulator:
    def __init__(self):
        self.buf = bytearray()
        self.sr = None
        self.sec = 0.0
    def add(self, wf: np.ndarray, sr: int):
        if self.sr is None:
            self.sr = sr
        self.buf.extend(wf.astype(np.int16).tobytes())
        self.sec += AUDIO_SLICE_SEC
    def ready(self) -> bool:
        return self.sec >= WHISPER_INTERVAL
    def flush_wav(self) -> bytes:
        if self.sr is None or not self.buf:
            return b""
        wav = wrap_pcm16_to_wav(bytes(self.buf), self.sr, 1)
        tail_bytes = int(OVERLAP_SEC * self.sr) * 2
        tail = self.buf[-tail_bytes:] if len(self.buf) > tail_bytes else self.buf[:]
        self.buf.clear()
        self.buf.extend(tail)
        self.sec = OVERLAP_SEC
        return wav

def gate_is_speech(dbfs: float, raw_label: str, raw_conf: float,
                   group_label: str, group_conf: float) -> bool:
    level_ok = (dbfs >= DBFS_GATE)
    by_raw   = raw_label.lower().startswith("speech") and (raw_conf  >= SPEECH_RAW_GATE_CONF)
    by_group = (group_label in ("Shout","Speech")) and (group_conf >= SPEECH_GATE_CONF)
    return level_ok and (by_raw or by_group)

# ==============================
# 내부 로깅
# ==============================
def clog(*args, **kwargs):
    # 맨 앞 문자열이 숨김 태그로 시작하면 출력하지 않음
    if args and isinstance(args[0], str):
        first = args[0].lstrip()
        for tag in SUPPRESS_LOG_TAGS:
            if tag and first.startswith(tag):
                return
    print(*args, flush=True, **kwargs)


def clog_exc(prefix: str = "[EXC]"):
    print(prefix, flush=True)
    traceback.print_exc()

RT_CLog = clog
RT_CLogExc = clog_exc

# ==============================
# JSON 직렬화 보조
# ==============================
def _to_plain(x):
    try:
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.ndarray,)):
            return x.tolist()
    except Exception:
        pass
    if isinstance(x, (float, int, str, bool)) or x is None:
        return x
    return str(x)

# ==============================
# 디버그 유틸
# ==============================
def esp32_list_ids() -> List[str]:
    try:
        return list(esp32_conns.keys())
    except Exception:
        return []

# ==============================
# ESP32 통신 헬퍼
# ==============================
async def _send_ws_safe(ws: WebSocket, payload: dict) -> bool:
    try:
        async with send_lock:
            await ws.send_text(json.dumps(payload, ensure_ascii=False))
        return True
    except Exception as e:
        clog("[ESP32 send error]", e)
        return False

async def esp32_send_json(payload: dict, esp_id: Optional[str] = None) -> bool:
    eid = esp_id or DEFAULT_ESP32_ID
    ws = esp32_conns.get(eid)
    if ws is None:
        clog(f"[ESP32] not connected: {eid} (connected={esp32_list_ids()})")
        return False
    return await _send_ws_safe(ws, payload)

async def esp32_broadcast_all(payload: dict):
    dead = []
    for eid, ws in list(esp32_conns.items()):
        ok = await _send_ws_safe(ws, payload)
        if not ok:
            dead.append(eid)
    for d in dead:
        esp32_conns.pop(d, None)

_last_vibrate_ts = 0  # debounce ts(ms)

def _norm_angle_for_esp(angle_deg: Optional[int]) -> int:
    try:
        if angle_deg is None or int(angle_deg) < 0:
            a = ANGLE_FALLBACK
        else:
            a = int(angle_deg)
        return a % 360
    except Exception:
        return ANGLE_FALLBACK % 360

async def esp32_vibrate_angle(ms: Optional[int] = None,
                              angle_deg: Optional[int] = None,
                              strength: int = 100,
                              esp_id: Optional[str] = None) -> bool:
    global _last_vibrate_ts
    now = now_ms()
    if not DISABLE_DEBOUNCE and (now - _last_vibrate_ts < VIBRATE_DEBOUNCE_MS):
        clog(f"[VIBRATE] blocked by debounce ({now - _last_vibrate_ts}ms < {VIBRATE_DEBOUNCE_MS}ms)")
        return True
    _last_vibrate_ts = now

    _ms = int(ms if ms is not None else VIBRATE_MS)
    _angle = _norm_angle_for_esp(angle_deg)
    payload = {"t": "vibrate", "ms": _ms, "angle": _angle, "strength": int(strength)}
    if ESP_COMPAT_KEYS:
        payload.update({"direction": _angle, "deg": _angle})  # ← 필요 시만 켜기
    clog(f"[ESP->] vibrate angle={_angle} ms={_ms} ids={esp32_list_ids()}")
    return await esp32_send_json(payload, esp_id=esp_id)

# ==============================
# 앱 허브 브로드캐스트
# ==============================
async def _broadcast_to_app_hub(payload: dict, topic: Optional[str] = None):
    """
    payload를 앱 허브(/ws/app)로 전송. topic이 주어지면 그 토픽으로, 없으면 WS_TOPIC.
    """
    # numpy → 기본형 변환
    for k, v in list(payload.items()):
        if isinstance(v, dict):
            payload[k] = {kk: _to_plain(vv) for kk, vv in v.items()}
        else:
            payload[k] = _to_plain(v)

    # 🔧 import 경로를 절대 경로 → 패키지 순으로 시도 (환경에 따라 다를 수 있어서)
    try:
        from routes.ws_app import app_broadcast_json  # FREEZE/app.py 기준
    except Exception:
        try:
            from .routes.ws_app import app_broadcast_json  # freeze/runtime.py 기준 패키지
        except Exception:
            from freeze.routes.ws_app import app_broadcast_json  # 모듈 이름이 freeze인 경우

    try:
        eff_topic = str(topic or payload.get("topic") or WS_TOPIC)
        await app_broadcast_json(eff_topic, payload)
    except Exception as e:
        clog("[APP broadcast error]", e)

def _norm_direction(direction: Optional[int]) -> int:
    try:
        if direction is None:
            return -1
        ang = int(direction) % 360
        if 0 <= ang < 360:
            return ang
    except Exception:
        pass
    return -1

# ===== danger 엣지/쿨다운 상태 =====
_last_danger_ts = 0
_last_danger_key = None  # (angle_bucket, label, source)

def _danger_bucket(angle: int, bucket: int = None) -> int:
    if bucket is None:
        bucket = ANGLE_BUCKET_DEG
    try:
        if angle is None or angle < 0:
            return -1
        b = int(bucket) if int(bucket) > 0 else 30
        return (int(angle) % 360) // b
    except Exception:
        return -1

# ==============================
# 메인 브로드캐스트: 앱 허브 + (옵션) ESP 미러 + danger 진동
# ==============================
async def broadcast_info(
    direction: int,
    group_label: str,
    group_conf: float,
    dbfs: float,
    ms: int = 0,
    raw: Optional[dict] = None,
    transcript: Optional[str] = None,
    yolo: Optional[dict] = None,
    event: str = "info",
    files: Optional[dict] = None,   # ex) {"snapshot_url": "...", "video_url": "..."}
    source: Optional[str] = None,   # "yolo" | "yamnet" | "whisper"
    strength: int = 100,
    topic: Optional[str] = None,
):
    # ★ 전역 상태는 함수 시작부에서 한 번에 선언 (읽기/쓰기 모두 커버)
    global _last_danger_ts, _last_danger_key

    # ---- 방향 보정 ----
    ang = _norm_direction(direction)
    if ang < 0:
        async with state_lock:
            ang = _norm_direction(last_direction)

    ev = (event or "info").lower()

    # conf 안전 포맷
    try:
        conf_val = float(group_conf)
    except Exception:
        conf_val = 0.0

    # 진단 로그
    clog(f"[BI] enter event={ev} dir={ang} ms={ms} lbl={group_label} conf={conf_val:.2f}")

    # ---- 공통 payload ----
    eff_topic = str(topic or WS_TOPIC)
    app_payload = {
        "type": "event",
        "topic": eff_topic,
        "ts": now_ms(),
        "event": ev,
        "source": source,
        "direction": ang if ang is not None else -1,
        "label": group_label,
        "confidence": conf_val,
        "dbfs": float(dbfs),
        "ms": int(ms or 0),                   # 대시보드 표시용
        "angle": int(ang if ang is not None else -1),
        "strength": int(strength),
    }
    if raw is not None:
        app_payload["raw"] = {
            "idx": int(raw.get("idx", -1)),
            "label": str(raw.get("label", "")),
            "conf": float(raw.get("conf", 0.0)),
        }
    if transcript is not None:
        app_payload["transcript"] = str(transcript)
    if yolo is not None:
        app_payload["yolo"] = yolo
    if files is not None:
        app_payload["files"] = files

    # 1) 앱 허브로 전파 (topic 반영)
    await _broadcast_to_app_hub(dict(app_payload), topic=eff_topic)

    # 2) (옵션) ESP로 미러링(디버그): 모든 이벤트
    if MIRROR_EVENTS_TO_ESP and esp32_conns:
        esp_payload = dict(app_payload)
        esp_payload["t"] = "event"  # 일반 이벤트임을 명시
        try:
            clog(f"[ESP->] mirror event={ev} angle={esp_payload['angle']} ids={esp32_list_ids()}")
            await esp32_broadcast_all(esp_payload)
        except Exception as e:
            clog("[ESP mirror error]", e)

    # 3) danger만 진동(엣지 + 쿨다운 + 최소 conf)
    if ev == "danger":
        # 너무 낮은 conf면 스킵
        if conf_val < DANGER_MIN_CONF:
            clog(f"[VIBRATE] skip: conf<{DANGER_MIN_CONF} (conf={conf_val:.2f})")
            return

        final_ms = int(ms or VIBRATE_MS)
        final_angle = ang if ang is not None and ang >= 0 else ANGLE_FALLBACK

        now = now_ms()
        key = (_danger_bucket(final_angle), str(group_label or ""), str(source or ""))

        allow_by_edge = (key != _last_danger_key) if EDGE_TRIGGER_DANGER else True
        allow_by_cooldown = (now - _last_danger_ts >= DANGER_COOLDOWN_MS)

        if allow_by_edge or allow_by_cooldown:
            clog(
                f"[VIBRATE] event=danger angle={final_angle} ms={final_ms} key={key} "
                f"edge={allow_by_edge} cool={allow_by_cooldown} ids={esp32_list_ids()}"
            )
            ok = await esp32_vibrate_angle(ms=final_ms, angle_deg=final_angle, strength=strength)
            if not ok:
                clog("[ESP32] vibrate send failed (not connected?)")
            _last_danger_ts  = now
            _last_danger_key = key
        else:
            clog(
                f"[VIBRATE] skipped: cooldown key={key} "
                f"dt={now - _last_danger_ts}ms<{DANGER_COOLDOWN_MS}ms"
            )
