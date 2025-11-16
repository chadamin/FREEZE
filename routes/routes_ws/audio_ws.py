"""오디오 WebSocket 핸들러 (YAMNet + Whisper)"""
import asyncio
import json
import time
import io
import wave
import numpy as np
import re
from typing import Optional
from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from yamnet_module import classify_sound_with_confidence
from whisper_module import transcribe_audio_wav_bytes
from danger_check import is_significant_group
from runtime import (
    DOMAIN_HINT, NAMES_HINTS, WHISPER_ASYNC,
    state_lock, now_ms, broadcast_info, VIBRATE_MS,
    rms_and_dbfs, decode_from_canonical_payload, parse_binary_frame,
    vad_is_speech_int16, WhisperAccumulator, gate_is_speech,
    last_direction, last_group_label, last_group_conf,
    last_raw_idx, last_raw_label, last_raw_conf,
    last_energy_rms, last_dbfs, last_transcript, last_updated_ms
)
from .config import (
    AUDIO_WS_DEFAULT_MODE, RAW_SR, RAW_FRAME_MS, RAW_SAMPLES, RAW_FRAME_BYTES,
    FOCUS_HOTWORDS, FOCUS_WINDOW_MS, SAVE_AUDIO, SAVE_AUDIO_TS, AUDIO_OUT_DIR,
    SAVE_AUDIO_LEN_SEC
)
from .utils import maybe_set_base_url_from_ws, log_exc, wav_dur_sec

router = APIRouter()

# ====== 오디오 링 버퍼 ======
_audio_ring = bytearray()
_audio_ring_sr = RAW_SR
_last_ring_at_ms = 0

def _ring_cap_bytes(sr: int) -> int:
    """링 버퍼 용량(바이트)"""
    return int(SAVE_AUDIO_LEN_SEC * sr) * 2

def _ring_append_int16(int16_mono: np.ndarray, sr: int, frame_ms: int = 200):
    """수신 지연이 크면 무음(0)으로 패딩한 뒤 링버퍼에 붙인다"""
    global _audio_ring, _audio_ring_sr, _last_ring_at_ms
    
    if int16_mono is None or int16_mono.size == 0:
        return
    
    _audio_ring_sr = int(sr or _audio_ring_sr or RAW_SR)
    now_ms_val = int(time.time() * 1000)
    
    # 직전 append 이후 지연 체크
    if _last_ring_at_ms > 0:
        dt = now_ms_val - _last_ring_at_ms
        expect = frame_ms
        if dt > int(expect * 1.5):
            missed = max(1, int(round(dt / expect)) - 1)
            pad_bytes = missed * (_audio_ring_sr * frame_ms // 1000) * 2
            _audio_ring.extend(b"\x00" * pad_bytes)
    
    # 실제 프레임 추가
    _audio_ring.extend(int16_mono.astype(np.int16).tobytes())
    _last_ring_at_ms = now_ms_val
    
    # 링 용량 유지
    cap = _ring_cap_bytes(_audio_ring_sr)
    if len(_audio_ring) > cap:
        _audio_ring = _audio_ring[-cap:]

# ====== Hotword & Focus ======
_focus_until_ms = 0
_focus_lock = asyncio.Lock()

def _hit_hotword(text: str) -> bool:
    """핫워드 감지"""
    if not text or not FOCUS_HOTWORDS:
        return False
    for hw in FOCUS_HOTWORDS:
        if hw and re.search(re.escape(hw), text, re.IGNORECASE):
            return True
    return False

# ====== 오디오 처리 ======
def _dbfs_int16(x: np.ndarray) -> float:
    """Int16 음성의 dBFS 계산"""
    if x.size == 0:
        return float("-inf")
    peak = np.max(np.abs(x))
    return float("-inf") if peak == 0 else 20.0 * np.log10(peak / 32767.0)

# ====== WebSocket 엔드포인트 ======
@router.websocket("/ws/audio")
async def audio_ws(websocket: WebSocket):
    """
    오디오 WebSocket: Canonical 또는 RAW 바이너리 수신
    Query params:
        - mode: env | speech | both (기본값: AUDIO_WS_DEFAULT_MODE)
    
    Processes:
        - YAMNet 분류 (환경음)
        - Whisper 음성 인식
        - 위험 감지 및 브로드캐스트
    """
    global _focus_until_ms
    
    try:
        await websocket.accept()
        mode = websocket.query_params.get("mode", AUDIO_WS_DEFAULT_MODE).lower()
        
        # 🔥 디버그용: 모드와 상관없이 YAMNet + Whisper 둘 다 항상 켠다
        DO_WHISPER = True
        DO_YAMNET = True
        
        print(f"🎧 /ws/audio 연결됨 mode={mode} (whisper={DO_WHISPER}, yamnet={DO_YAMNET})")
    except Exception as e:
        log_exc("[AUDIO accept]", e)
        return
    
    maybe_set_base_url_from_ws(websocket)
    
    acc = WhisperAccumulator()
    bbuf = bytearray()
    
    # ====== Whisper 처리 ======
    async def run_whisper_once(wav_bytes: bytes):
        """Whisper 한 번 실행"""
        global _focus_until_ms, last_transcript, last_updated_ms
        
        if not DO_WHISPER:
            return
        
        try:
            phrase_boost = list(set(NAMES_HINTS + FOCUS_HOTWORDS))
            res = await transcribe_audio_wav_bytes(
                wav_bytes, lang="ko", initial_prompt=DOMAIN_HINT, phrase_hints=phrase_boost
            ) or {"text": "", "hits": []}
            text = res.get("text", "") if isinstance(res, dict) else str(res or "")
            hits = list(res.get("hits", []) or []) if isinstance(res, dict) else []
        except Exception as e:
            log_exc("[whisper error]", e)
            text, hits = "", []
        
        if hits:
            print(f"[WHISPER][HIT] {hits}")
        
        try:
            now = now_ms()
            opened = False
            
            # 핫워드 감지 시 포커스 윈도우 연장
            if text and _hit_hotword(text):
                async with _focus_lock:
                    _focus_until_ms = max(_focus_until_ms, now + FOCUS_WINDOW_MS)
                    opened = True
            
            allow = now <= _focus_until_ms
            
            # 포커스 내일 때만 전송
            if text and allow:
                async with state_lock:
                    last_transcript = text
                    last_updated_ms = now
                
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
            log_exc("[whisper broadcast]", e)
    
    # ====== 파형 처리 ======
    async def _process_waveform(waveform: np.ndarray, sr: int, dir_in: int = -1):
        """프레임 처리: YAMNet + Whisper"""
        global last_direction, last_group_label, last_group_conf
        global last_raw_idx, last_raw_label, last_raw_conf
        global last_energy_rms, last_dbfs, last_updated_ms
        
        if waveform is None or getattr(waveform, "size", 0) == 0:
            print("[AUDIO] _process_waveform: empty frame")
            return
        
        _ring_append_int16(waveform, int(sr or RAW_SR), frame_ms=RAW_FRAME_MS)
        
        dbfs = -120.0
        rms = 0.0
        group_label = "no-audio"
        group_conf = 0.0
        raw_idx = -1
        raw_label = ""
        raw_conf = 0.0
        
        # === YAMNet 분류 ===
        try:
            rms, dbfs = rms_and_dbfs(waveform)
            print(f"[YAMNET] 분류 시작 sr={sr} len={waveform.size} dbfs={dbfs:.1f}")
            
            if DO_YAMNET:
                result = await asyncio.to_thread(
                    classify_sound_with_confidence, waveform, sr
                )
                group_label = result.get("group_label", "unknown")
                group_conf = float(result.get("group_conf", 0.0))
                raw_idx = int(result.get("raw_idx", -1))
                raw_label = str(result.get("raw_label", ""))
                raw_conf = float(result.get("raw_conf", 0.0))
                
                print(
                    f"[YAMNET] label={group_label} g_conf={group_conf:.3f} "
                    f"raw={raw_label} r_conf={raw_conf:.3f} dbfs={dbfs:.1f}"
                )
            else:
                print("[YAMNET] 스킵 (DO_YAMNET=False)")
        except Exception as e:
            log_exc("[YAMNET error]", e)
        
        # 방향 정규화
        try:
            dir_norm = (int(dir_in) % 360) if int(dir_in) >= 0 else -1
        except Exception:
            dir_norm = -1
        
        # === 전역 상태 업데이트 ===
        try:
            async with state_lock:
                if 0 <= dir_norm < 360:
                    last_direction = dir_norm
                last_group_label = group_label
                last_group_conf = group_conf
                last_raw_idx = raw_idx
                last_raw_label = raw_label
                last_raw_conf = raw_conf
                last_energy_rms = rms
                last_dbfs = dbfs
                last_updated_ms = now_ms()
        except Exception as e:
            log_exc("[AUDIO state_lock]", e)
        
        # === Whisper (음성일 때만) ===
        try:
            vad_ok = False
            if waveform is not None and getattr(waveform, "size", 0) > 0:
                vad_ok = vad_is_speech_int16(waveform.astype(np.int16).tobytes(), sr)
            
            if DO_WHISPER and vad_ok and gate_is_speech(dbfs, raw_label, raw_conf, group_label, group_conf):
                acc.add(waveform, sr)
                if acc.ready():
                    wav_for_whisper = acc.flush_wav()
                    dur = wav_dur_sec(wav_for_whisper) if wav_for_whisper else 0.0
                    
                    # 오디오 저장
                    if SAVE_AUDIO:
                        try:
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
                            
                            last_path = f"{AUDIO_OUT_DIR}/last_in.wav"
                            with open(last_path, "wb") as f:
                                f.write(out_bytes)
                            
                            if SAVE_AUDIO_TS:
                                ts_name = f"in_{int(time.time()*1000)}.wav"
                                with open(f"{AUDIO_OUT_DIR}/{ts_name}", "wb") as f:
                                    f.write(out_bytes)
                            
                            print(f"[AUDIO][SAVE] {last_path} (≈{SAVE_AUDIO_LEN_SEC:.1f}s, sr={out_sr})")
                        except Exception as e:
                            log_exc("[AUDIO save wav]", e)
                    
                    # Whisper 실행
                    if wav_for_whisper and dur >= 0.8:
                        if WHISPER_ASYNC:
                            asyncio.create_task(run_whisper_once(wav_for_whisper))
                        else:
                            await run_whisper_once(wav_for_whisper)
        except Exception as e:
            log_exc("[AUDIO whisper path]", e)
        
        # === 위험/정보 브로드캐스트 ===
        try:
            significant = is_significant_group(group_label, group_conf, dbfs)
            await broadcast_info(
                direction=last_direction, group_label=group_label,
                group_conf=group_conf, dbfs=dbfs,
                ms=(VIBRATE_MS if significant else 0),
                raw={"idx": raw_idx, "label": raw_label, "conf": raw_conf},
                event=("danger" if significant else "info"),
                source="yamnet"
            )
        except Exception as e:
            log_exc("[AUDIO broadcast_info]", e)
    
    # ====== 메인 루프 ======
    try:
        while True:
            try:
                msg = await websocket.receive()
            except WebSocketDisconnect:
                break
            except Exception as e:
                log_exc("[AUDIO receive]", e)
                break
            
            # 수신 로그
            if msg.get("bytes"):
                print(f"[AUDIO] 수신 바이너리 len={len(msg['bytes'])}")
            elif msg.get("text"):
                print(f"[AUDIO] 수신 텍스트 len={len(msg['text'])}")
            
            waveform = None
            sr = None
            dir_in = -1
            
            # ─── 1) JSON(Canonical) ───
            if msg.get("text"):
                try:
                    data = json.loads(msg["text"])
                except Exception:
                    continue
                
                # 각도 업데이트 전용 메시지
                if isinstance(data, dict) and data.get("type") == "angle":
                    try:
                        raw_dir = data.get("dir_deg", data.get("direction", data.get("dir", -1)))
                        d = int(raw_dir) if raw_dir is not None else -1
                        if 0 <= d < 360:
                            async with state_lock:
                                last_direction = d
                                last_updated_ms = now_ms()
                            print(f"[AUDIO] 각도 업데이트 dir={d}")
                    except Exception as e:
                        log_exc("[AUDIO angle msg]", e)
                    continue
                
                # Canonical 오디오 JSON
                if "pcm_b64" in data or "audio_b64" in data:
                    try:
                        dir_val = data.get("direction", data.get("dir", -1))
                        try:
                            dir_in = int(dir_val)
                        except:
                            dir_in = -1
                        waveform, sr, ch, wav_bytes = decode_from_canonical_payload(data)
                    except Exception as e:
                        log_exc("[AUDIO canonical JSON]", e)
                        continue
            
            # ─── 2) Binary(Canonical with header) or RAW fallback ───
            elif msg.get("bytes"):
                b = msg["bytes"]
                parsed = None
                try:
                    parsed = parse_binary_frame(b)
                except Exception:
                    parsed = None
                
                if parsed:
                    try:
                        (wf, sr, ch, wav_bytes, seq, ts_ms, dir_in, flags) = parsed
                        print(f"[AUDIO][BIN] 파싱됨 seq={seq} sr={sr} ch={ch} len={wf.size}")
                        if wf.dtype != np.int16:
                            wf = np.clip(wf, -1.0, 1.0) if wf.dtype != np.int16 else wf
                            wf = (wf * 32768.0).astype(np.int16) if wf.dtype != np.int16 else wf
                        waveform = wf
                    except Exception as e:
                        log_exc("[AUDIO canonical BIN adapt]", e)
                        waveform = None
                else:
                    # RAW fallback
                    bbuf.extend(b)
                    while len(bbuf) >= RAW_FRAME_BYTES:
                        frame_bytes = bbuf[:RAW_FRAME_BYTES]
                        del bbuf[:RAW_FRAME_BYTES]
                        
                        if len(frame_bytes) != RAW_FRAME_BYTES or (len(frame_bytes) % 2) != 0:
                            print(f"[AUDIO][RAW] 드롭 (정렬 안됨) len={len(frame_bytes)}")
                            continue
                        
                        wf = np.frombuffer(frame_bytes, dtype=np.int16)
                        print(f"[AUDIO][RAW] 프레임 len={wf.size}")
                        
                        _ring_append_int16(wf, RAW_SR, frame_ms=RAW_FRAME_MS)
                        await _process_waveform(wf, RAW_SR, dir_in=-1)
                    
                    continue
            
            # 최종 처리
            if waveform is not None and getattr(waveform, "size", 0) > 0:
                await _process_waveform(waveform, int(sr or RAW_SR), dir_in)
    
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
        print("🎧 /ws/audio 연결 끊김")
