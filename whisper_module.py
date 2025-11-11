# whisper_module.py
# ------------------------------------------------------------
# Real-time partials + one-shot final:
# - Quiet chunks ignored by VAD
# - On trigger, stream partial transcripts every ~0.5s
# - After ~3s window, emit one final segment
# ------------------------------------------------------------

import asyncio
import io
import os
import re
import tempfile
import wave
import time
from dataclasses import dataclass, field
from hashlib import blake2s
from typing import Optional, List, Dict, Tuple
from collections import deque

import numpy as np
import torch

# === faster-whisper ===
from faster_whisper import WhisperModel

# ========= Config (env-overridable) =========
MODEL_NAME       = os.getenv("WHISPER_MODEL", "small")       # 최종 전사용
FAST_MODEL_NAME  = os.getenv("WHISPER_FAST_MODEL", "tiny")   # 부분 전사용(저지연)
# faster-whisper 연산 타입 (CPU: int8_float16 권장 / GPU: float16 권장)
COMPUTE_TYPE     = os.getenv("WHISPER_COMPUTE", "int8_float16")
COMPUTE_TYPE_FAST= os.getenv("WHISPER_COMPUTE_FAST", COMPUTE_TYPE)

NO_SPEECH_THR    = float(os.getenv("WHISPER_NO_SPEECH_THR", "0.6"))   # (참고: openai-whisper 전용이었음)
COMP_RATIO_THR   = float(os.getenv("WHISPER_COMP_RATIO_THR", "2.6"))  # (참고: openai-whisper 전용이었음)
LOGPROB_THR      = float(os.getenv("WHISPER_LOGPROB_THR", "-1.0"))    # (참고: openai-whisper 전용이었음)
MIN_DUR_SECS     = float(os.getenv("WHISPER_MIN_DUR_SECS", "0.08"))
LANG_DEFAULT     = os.getenv("WHISPER_LANG", "ko")
PHRASE_HINTS_ENV = [s.strip() for s in os.getenv("PHRASE_HINTS", "").split(",") if s.strip()]

# 키워드 히트(간단 매칭)
DEFAULT_WHISPER_KEYWORDS: List[str] = [
    "help","fire","danger","sos","emergency",
    "불","불이야","도와줘","위험","살려줘","대피","비상","경보",
    "수현아", "광민아", "두형아", "진묵아", "다민아",
    "수현","광민","두형","진묵","다민", "야야"
]
WHISPER_KEYWORDS: List[str] = [
    s.strip() for s in os.getenv("WHISPER_KEYWORDS","").split(",") if s.strip()
] or DEFAULT_WHISPER_KEYWORDS

# 트리거 단어(명시)
CAPTURE_TRIGGERS: List[str] = [
    s.strip() for s in os.getenv("WHISPER_CAPTURE_TRIGGERS","").split(",") if s.strip()
]

# ---- One-shot event capture window ----
TRIGGER_ONLY        = os.getenv("WHISPER_TRIGGER_ONLY", "true").lower() == "true"
EVENT_TOTAL_SEC     = float(os.getenv("WHISPER_EVENT_TOTAL_SEC", "3.0"))   # 총 3초
EVENT_PRE_SEC       = float(os.getenv("WHISPER_EVENT_PRE_SEC",   "1.0"))   # 프리롤 1초
VAD_DBFS_GATE       = float(os.getenv("WHISPER_DBFS_GATE",       "-55.0")) # 이하면 무음

# ---- Real-time partial streaming ----
LIVE_PARTIAL            = os.getenv("WHISPER_LIVE_PARTIAL", "true").lower() == "true"
LIVE_STEP_SEC           = float(os.getenv("WHISPER_LIVE_STEP_SEC", "0.5"))   # 부분 전사 간격
LIVE_MIN_INTERVAL       = float(os.getenv("WHISPER_LIVE_MIN_INTERVAL", "0.25"))
LIVE_BEAM_SIZE          = int(os.getenv("WHISPER_LIVE_BEAM", "1"))          # 속도 위해 1 권장

# 같은 문장 디바운스(비트리거 모드에서만 의미)
WHISPER_DEBOUNCE_SEC = float(os.getenv("WHISPER_DEBOUNCE_SEC", "2"))

# 도메인 힌트 예시(대피 방송 문구 등)
EVAC_HINTS: List[str] = [
    "화재 발생","지금 즉시 대피하십시오","대피 안내 방송","비상방송","경보음","비상구",
    "계단을 이용하십시오","엘리베이터 사용 금지","소방 안내","집결지","유도등","불이야",
    "Fire alarm","Evacuation","Emergency exit","Do not use the elevator"
]

# 간단 후처리 교정
POST_CORRECT_MAP: Dict[str,str] = {
    "지금 즉시 대피하십시요":"지금 즉시 대피하십시오",
    "계단을 이용 하십시오":"계단을 이용하십시오",
    "엘리베이터 사용 금지 입니다":"엘리베이터 사용 금지입니다",
    "유도 등":"유도등",
    r"\besp\s*32\b":"ESP32",
    r"\b라즈베리\s*파이\b":"Raspberry Pi",
}

# ========= Model (lazy, single instance) =========
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_model_final = None
_model_live  = None

def _get_model_final():
    """최종 품질 모델 (싱글톤)"""
    global _model_final
    if _model_final is None:
        _model_final = WhisperModel(MODEL_NAME, compute_type=COMPUTE_TYPE)
        print(f"[WHISPER] final model loaded: {MODEL_NAME} ({COMPUTE_TYPE}) on {_DEVICE}")
    return _model_final

def _get_model_live():
    """저지연 라이브 모델 (싱글톤)"""
    global _model_live
    if _model_live is None:
        name = FAST_MODEL_NAME if FAST_MODEL_NAME else MODEL_NAME
        ctype = COMPUTE_TYPE_FAST
        _model_live = WhisperModel(name, compute_type=ctype)
        print(f"[WHISPER] live  model loaded: {name} ({ctype}) on {_DEVICE}")
    return _model_live

# ========= WAV / PCM helpers =========
def _wav_duration_seconds(b: bytes) -> float:
    try:
        with wave.open(io.BytesIO(b), "rb") as wf:
            return wf.getnframes() / float(max(1, wf.getframerate()))
    except:
        return 0.0

def _is_effectively_empty(b: bytes) -> bool:
    return (not b) or (_wav_duration_seconds(b) < MIN_DUR_SECS)

def _wav_info(b: bytes) -> Tuple[Optional[int], Optional[int], Optional[int], bytes]:
    try:
        with wave.open(io.BytesIO(b), "rb") as wf:
            sr = wf.getframerate()
            ch = wf.getnchannels()
            sw = wf.getsampwidth()
            pcm = wf.readframes(wf.getnframes())
        return sr, ch, sw, pcm
    except Exception:
        return None, None, None, b""

def _pcm16_to_wav(pcm: bytes, sr: int) -> bytes:
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm)
    return bio.getvalue()

def _pcm_dbfs(pcm: bytes) -> float:
    if not pcm:
        return -120.0
    a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    if a.size == 0:
        return -120.0
    rms = np.sqrt(np.mean(np.square(a)))
    if rms <= 1e-7:
        return -120.0
    return 20.0 * np.log10((rms / 32768.0) + 1e-12)

# ========= Hints / Post-process =========
def _merge_hints(runtime_hints: Optional[List[str]]) -> List[str]:
    hints = list(EVAC_HINTS)
    if PHRASE_HINTS_ENV: hints.extend(PHRASE_HINTS_ENV)
    if runtime_hints: hints.extend(runtime_hints)
    hints = [h.strip() for h in hints if h and h.strip()]
    return hints[:200]

def _build_boost_prompt(phrases: List[str], repeat: int = 2) -> str:
    if not phrases:
        return ""
    glossary = ", ".join(sorted(set(phrases), key=str.lower))
    # 지시문 제거(bleed 방지) — 용어 나열만
    return glossary

def _apply_post(text: str) -> str:
    if not text: return text
    for k,v in POST_CORRECT_MAP.items():
        if not any(ch in k for ch in ".*+?^$[](){}|\\"):
            text = text.replace(k, v)
    for k,v in POST_CORRECT_MAP.items():
        if any(ch in k for ch in ".*+?^$[](){}|\\") or k.startswith(r"\b"):
            try: text = re.sub(k, v, text, flags=re.IGNORECASE)
            except: pass
    return text

_repeat_pat = re.compile(r'(?:\b([\w가-힣]{2,12})\b[\s,·]*)(?:\1[\s,·]*){2,}')
def _clean_transcript(text: str) -> str:
    if not text: return ""
    s = re.sub(r'\s+', ' ', text).strip()
    def _collapse(m): return f"{m.group(1)} {m.group(1)}"
    s = _repeat_pat.sub(_collapse, s)
    s = re.sub(r'((?:[\w가-힣]{2,6})(?:\s|,|·)?)(?:\1){2,}', r'\1\1', s)
    s = re.sub(r'[!！]{3,}', '!!', s)
    s = re.sub(r'[?？]{3,}', '??', s)
    s = re.sub(r'[,.]{3,}', '..', s)
    MAX_LEN = 240
    if len(s) > MAX_LEN:
        s = s[:MAX_LEN].rstrip() + "…"
    return s

_LAST_HASH = ""
_LAST_AT   = 0.0
def _debounce_same_text(text: str) -> bool:
    global _LAST_HASH, _LAST_AT
    if not text: return False
    h = blake2s(text.encode("utf-8"), digest_size=8).hexdigest()
    now = time.time()
    if h == _LAST_HASH and (now - _LAST_AT) < WHISPER_DEBOUNCE_SEC:
        return True
    _LAST_HASH, _LAST_AT = h, now
    return False

def _find_hits(text: str, keywords: List[str]) -> List[str]:
    if not text: return []
    t = text.casefold()
    hits = set()
    for kw in keywords:
        k = kw.casefold()
        if re.match(r"^[a-z0-9 _-]+$", k):
            if re.search(r"\b"+re.escape(k)+r"\b", t): hits.add(kw)
        else:
            if k in t: hits.add(kw)
    return sorted(hits)

def _has_trigger(hits: List[str], text: str) -> bool:
    if CAPTURE_TRIGGERS:
        t = (text or "").casefold()
        for w in CAPTURE_TRIGGERS:
            if not w: continue
            if w.casefold() in t: return True
            if w in hits: return True
        return False
    return bool(hits)

def _is_prompt_bleed(text: str, phrases: List[str]) -> bool:
    if not text: return False
    if re.search(r"(다음 용어는|아래 단어들은|전문 용어 표기)", text): return True
    if not phrases: return False
    toks = [t for t in re.split(r"[\s,·]+", text.strip()) if t]
    if not toks: return False
    vocab = set(phrases)
    hits = sum(1 for t in toks if t in vocab)
    return hits >= max(3, int(len(toks) * 0.7))

# ========= Real-time Event Capture =========
@dataclass
class _EventCapture:
    sr: int = 16000
    pre_sec: float = EVENT_PRE_SEC
    total_sec: float = EVENT_TOTAL_SEC
    pre_buf: deque = field(default_factory=lambda: deque(maxlen=1))  # int16
    post_buf: bytearray = field(default_factory=bytearray)
    armed: bool = False
    deadline: float = 0.0
    pre_bytes: int = 0
    post_target: int = 0
    # live partial
    step_bytes: int = 0
    last_live_len: int = 0
    last_live_at: float = 0.0

    def _reconf(self, sr: int):
        self.sr = sr
        self.pre_bytes   = int(self.sr * self.pre_sec) * 2
        self.post_target = int(self.sr * max(0.0, self.total_sec - self.pre_sec)) * 2
        self.step_bytes  = int(self.sr * max(0.05, LIVE_STEP_SEC)) * 2
        self.pre_buf     = deque(maxlen=max(1, self.pre_bytes // 2))
        self.post_buf.clear()
        self.last_live_len = 0
        self.last_live_at  = 0.0

    def feed_pre(self, wav_bytes: bytes):
        sr, ch, sw, pcm = _wav_info(wav_bytes)
        if not sr or ch != 1 or sw != 2:
            return
        if sr != self.sr or self.pre_bytes == 0:
            self._reconf(sr)
        self.pre_buf.extend(np.frombuffer(pcm, dtype=np.int16).tolist())

    def arm(self, sr: int):
        if sr != self.sr or self.pre_bytes == 0:
            self._reconf(sr)
        self.armed = True
        self.deadline = time.time() + max(0.0, self.total_sec - self.pre_sec)
        self.post_buf.clear()
        self.last_live_len = 0
        self.last_live_at  = 0.0
        print(f"[WHISPER][EVENT] armed {self.total_sec:.2f}s (pre {self.pre_sec:.2f}s) sr={sr}")

    def feed_and_maybe_emit_final(self, wav_bytes: bytes) -> Optional[bytes]:
        # 항상 pre 갱신
        self.feed_pre(wav_bytes)
        if not self.armed:
            return None
        sr, ch, sw, pcm = _wav_info(wav_bytes)
        if not sr or ch != 1 or sw != 2:
            return None
        self.post_buf.extend(pcm)
        done_by_len = len(self.post_buf) >= self.post_target
        done_by_time = time.time() >= self.deadline
        if done_by_len or done_by_time:
            pre_samples = np.array(list(self.pre_buf), dtype=np.int16)
            pre_tail = pre_samples[-(self.pre_bytes // 2):] if self.pre_sec > 0 and pre_samples.size else np.array([], dtype=np.int16)
            post = np.frombuffer(bytes(self.post_buf), dtype=np.int16)
            mix = np.concatenate([pre_tail, post])
            max_samples = int(self.total_sec * self.sr)
            if mix.size > max_samples:
                mix = mix[:max_samples]
            out_wav = _pcm16_to_wav(mix.tobytes(), self.sr)
            # reset
            self.armed = False
            self.post_buf.clear()
            print(f"[WHISPER][EVENT] emitted {mix.size / self.sr:.2f}s")
            return out_wav
        return None

    def snapshot_live_if_due(self) -> Optional[bytes]:
        """Return pre+current-post snapshot for live partial, without consuming buffers."""
        if not (self.armed and LIVE_PARTIAL):
            return None
        now = time.time()
        cur_len = len(self.post_buf)
        if cur_len - self.last_live_len < self.step_bytes:
            return None
        if self.last_live_at and (now - self.last_live_at) < LIVE_MIN_INTERVAL:
            return None
        # build snapshot
        pre_samples = np.array(list(self.pre_buf), dtype=np.int16)
        pre_tail = pre_samples[-(self.pre_bytes // 2):] if self.pre_sec > 0 and pre_samples.size else np.array([], dtype=np.int16)
        post = np.frombuffer(bytes(self.post_buf), dtype=np.int16)
        mix = np.concatenate([pre_tail, post])
        max_samples = int(self.total_sec * self.sr)  # cap
        if mix.size > max_samples:
            mix = mix[-max_samples:]
        out_wav = _pcm16_to_wav(mix.tobytes(), self.sr)
        self.last_live_len = cur_len
        self.last_live_at  = now
        return out_wav

_EVENT = _EventCapture()

# ========= Core transcription =========
def _fw_transcribe_to_text(model: WhisperModel, wav_bytes: bytes, lang: str, beam_size: int, initial_prompt: Optional[str]) -> str:
    """faster-whisper 호출을 공통화: temp 파일에 쓰지 않고 메모리 배열로 직접 호출"""
    try:
        # WAV → float32 [-1,1]
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            sr = wf.getframerate()
            ch = wf.getnchannels()
            pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
            if ch == 2:
                pcm = pcm.reshape(-1, 2).mean(axis=1).astype(np.int16)
        audio = (pcm.astype(np.float32) / 32768.0)
        segments, info = model.transcribe(
            audio,
            language=lang,
            task="transcribe",
            beam_size=beam_size,
            initial_prompt=initial_prompt,
            condition_on_previous_text=False,
        )
        text = "".join(seg.text for seg in segments).strip()
        return text
    except Exception as e:
        print("[WHISPER][FW] transcribe error:", e)
        return ""

def _transcribe_sync(audio_wav_bytes: bytes,
                     lang: Optional[str],
                     initial_prompt: Optional[str],
                     phrase_hints: Optional[List[str]]) -> Dict[str, object]:
    # 빈/짧은 오디오는 무시
    if _is_effectively_empty(audio_wav_bytes):
        _EVENT.feed_pre(audio_wav_bytes)
        return {"text":"", "hits":[], "segments":[]}

    sr, ch, sw, pcm = _wav_info(audio_wav_bytes)
    if not sr:
        return {"text":"", "hits":[], "segments":[]}

    # 프리롤 갱신
    _EVENT.feed_pre(audio_wav_bytes)

    # 무음이면(이벤트 미암) skip
    if (not _EVENT.armed) and (ch == 1 and sw == 2):
        dbfs = _pcm_dbfs(pcm)
        if dbfs < VAD_DBFS_GATE:
            return {"text":"", "hits":[], "segments":[]}

    m_final = _get_model_final()
    m_live  = _get_model_live()

    hints = _merge_hints(phrase_hints)
    boost = _build_boost_prompt(hints, repeat=3 if len(hints)>3 else 2)
    prompt = " ".join([p for p in [boost, initial_prompt] if p])

    # ===== 1) 메인 전사(트리거 감지용; live 모델로 가볍게) =====
    text = _fw_transcribe_to_text(
        m_live, audio_wav_bytes, (lang or LANG_DEFAULT), LIVE_BEAM_SIZE, initial_prompt=None
    )
    text = _apply_post(text)
    text = _clean_transcript(text)
    if _is_prompt_bleed(text, hints):
        text = ""

    # 디바운스(비트리거 전파 억제용)
    _ = _debounce_same_text(text) if not TRIGGER_ONLY else True

    hits = _find_hits(text, WHISPER_KEYWORDS)
    if hits:
        print(f"[WHISPER][HIT] {hits} :: {text}")

    # ===== 2) 트리거 감지 → 이벤트 ARM =====
    if _has_trigger(hits, text) and sr:
        _EVENT.arm(sr)
        # 현재 청크도 즉시 반영
        _EVENT.feed_pre(audio_wav_bytes)

    segments: List[Dict[str, object]] = []

    # ===== 3) 진행 중 부분 전사 (live) =====
    live_wav = _EVENT.snapshot_live_if_due()
    if live_wav:
        sub_text = _fw_transcribe_to_text(
            m_live, live_wav, (lang or LANG_DEFAULT), LIVE_BEAM_SIZE, initial_prompt=None
        )
        sub_text = _apply_post(sub_text)
        sub_text = _clean_transcript(sub_text)
        if _is_prompt_bleed(sub_text, hints):
            sub_text = ""
        if sub_text:
            segments.append({"i": 0, "text": sub_text, "live": True})
            print(f"[WHISPER][LIVE] {sub_text}")

    # ===== 4) 이벤트 완료 시 최종 3초 전사 =====
    final_wav = _EVENT.feed_and_maybe_emit_final(audio_wav_bytes)
    if final_wav:
        sub_text = _fw_transcribe_to_text(
            m_final, final_wav, (lang or LANG_DEFAULT), beam_size=5, initial_prompt=None
        )
        sub_text = _apply_post(sub_text)
        sub_text = _clean_transcript(sub_text)
        if _is_prompt_bleed(sub_text, hints):
            sub_text = ""
        if sub_text:
            segments.append({"i": 1, "text": sub_text, "live": False})
            print(f"[WHISPER][EVENT] final(3s): {sub_text}")

    # TRIGGER_ONLY면 메인 text는 공백
    return {"text": ("" if TRIGGER_ONLY else text), "hits": hits, "segments": segments}

# ========= Public APIs =========
async def transcribe_audio_wav_bytes(audio_wav_bytes: bytes,
                                     lang: Optional[str] = LANG_DEFAULT,
                                     initial_prompt: Optional[str] = None,
                                     phrase_hints: Optional[List[str]] = None) -> Dict[str, object]:
    return await asyncio.to_thread(_transcribe_sync, audio_wav_bytes, lang, initial_prompt, phrase_hints)

def transcribe_audio(audio_bytes: bytes) -> Dict[str, object]:
    return _transcribe_sync(audio_bytes, LANG_DEFAULT, None, None)
