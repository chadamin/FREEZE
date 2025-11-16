"""카메라 WebSocket 핸들러 (YOLO + 녹화)"""
import asyncio
import json
import base64
import time
import datetime
import os
import threading
import uuid
from typing import Optional, Tuple, List, Deque
from collections import deque
from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect
import cv2
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from camera_module import CameraEventProcessor
from runtime import (
    now_ms, broadcast_info, set_latest_jpg, pulse, is_pulsing,
    last_direction, last_group_label, last_group_conf,
    last_raw_idx, last_raw_label, last_raw_conf, last_dbfs, last_transcript
)
from .config import (
    YOLO_MODEL, OUTPUT_DIR, SNAP_QUALITY, PRE_ROLL_SEC, POST_ROLL_SEC,
    FALLBACK_FPS, VIDEO_CODEC, MAX_WIDTH, INLINE_SNAPSHOT_B64,
    WS_VIDEO_STREAM, WS_VIDEO_CHUNK
)
from .utils import (
    maybe_set_base_url_from_ws, log_exc, file_url,
    app_broadcast_json, app_broadcast_binary, pack_media_header
)

router = APIRouter()

# ===== YOLO 프로세서 =====
processor = CameraEventProcessor(YOLO_MODEL)

# ===== Pre-roll 버퍼 =====
_prebuf: Deque[Tuple[int, bytes]] = deque()
_prelock = threading.Lock()


def _push_preroll(ts_ms: int, jpg: bytes):
    """Pre-roll 버퍼에 프레임 추가"""
    try:
        with _prelock:
            _prebuf.append((ts_ms, jpg))
            cutoff = ts_ms - int(PRE_ROLL_SEC * 1000)
            while _prebuf and _prebuf[0][0] < cutoff:
                _prebuf.popleft()
    except Exception as e:
        log_exc("[preroll push]", e)


def draw_border(bgr: np.ndarray, show: bool) -> np.ndarray:
    """위험 감지 시 빨간 테두리 표시"""
    if not show:
        return bgr
    out = bgr.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
    return out


# ===== 스냅샷 저장 & 브로드캐스트 =====
def _save_snapshot(jpg_annotated: bytes) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """스냅샷 저장"""
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    name = f"snap_{ts}.jpg"
    path = os.path.join(OUTPUT_DIR, name)
    try:
        with open(path, "wb") as f:
            f.write(jpg_annotated)
        print(f"[SNAP] {path}")
        return path, name, int(time.time())
    except Exception as e:
        log_exc("[SNAP save]", e)
        return None, None, None


async def _publish_snapshot(
    topic: str, jpg_bytes: bytes,
    file_path: Optional[str], file_name: Optional[str], ts_sec: Optional[int]
):
    """스냅샷 앱으로 브로드캐스트"""
    try:
        files = {}
        if file_name:
            files["snapshot_url"] = file_url(file_name)
        
        payload = {
            "type": "snapshot",
            "source": "yolo",
            "ts": now_ms(),
            "file": (file_url(file_name) if file_name else None),
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
        
        # Base64 인라인 전송 (옵션)
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
        log_exc("[publish_snapshot]", e)


# ===== 비디오 녹화 클래스 =====
class Recorder:
    """녹화 관리 (Pre-roll + Post-roll)"""
    
    def __init__(self, out_dir: str, topic: str):
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
        """타임스탬프로 FPS 추정"""
        if len(ts_list) < 2:
            return FALLBACK_FPS
        dt = (ts_list[-1] - ts_list[0]) / max(1, (len(ts_list) - 1))
        fps = 1000.0 / max(dt, 1.0)
        return float(max(5.0, min(30.0, fps)))
    
    def _ensure_writer(self, w: int, h: int):
        """VideoWriter 초기화"""
        if self._writer is not None:
            return
        fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
        self._writer = cv2.VideoWriter(self._path, fourcc, self._fps, (w, h))
        self._size = (w, h)
    
    def start(self, preroll: List[Tuple[int, bytes]]):
        """녹화 시작 (Pre-roll 포함)"""
        with self._lock:
            if self.recording:
                return
            
            ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            self._path = os.path.join(self.out_dir, f"clip_{ts}.mp4")
            
            # FPS 추정
            ts_ms = [t for (t, _) in preroll][-30:]
            self._fps = self._estimate_fps(ts_ms)
            
            self._writer = None
            self._size = None
            self._deadline = time.time() + POST_ROLL_SEC
            self.recording = True
            print(f"[REC] 시작 {self._path}")
            
            name = os.path.basename(self._path)
            video_url = file_url(name)
            epoch_sec = int(time.time())
            
            # 브로드캐스트
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
            
            # Pre-roll 프레임 쓰기
            for _, jpg in preroll:
                try:
                    frm = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                    if frm is None:
                        continue
                    h, w = frm.shape[:2]
                    if w > MAX_WIDTH:
                        scale = MAX_WIDTH / float(w)
                        frm = cv2.resize(frm, (int(w * scale), int(h * scale)))
                        h, w = frm.shape[:2]
                    self._ensure_writer(w, h)
                    if self._size != (w, h):
                        frm = cv2.resize(frm, self._size)
                    self._writer.write(frm)
                except Exception as e:
                    log_exc("[REC preroll write]", e)
    
    def write(self, bgr: np.ndarray):
        """프레임 쓰기"""
        with self._lock:
            if not self.recording:
                return
            try:
                h, w = bgr.shape[:2]
                if w > MAX_WIDTH:
                    scale = MAX_WIDTH / float(w)
                    bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)))
                    h, w = bgr.shape[:2]
                self._ensure_writer(w, h)
                if self._size != (w, h):
                    bgr = cv2.resize(bgr, self._size)
                self._writer.write(bgr)
            except Exception as e:
                log_exc("[REC write]", e)
    
    def maybe_stop(self):
        """Post-roll 시간 초과 시 녹화 종료"""
        path = None
        with self._lock:
            if self.recording and time.time() > self._deadline:
                path = self._path
                try:
                    if self._writer is not None:
                        self._writer.release()
                except Exception as e:
                    log_exc("[REC release]", e)
                finally:
                    print(f"[REC] 종료 {self._path}")
                    self._writer = None
                    self._size = None
                    self._path = None
                    self.recording = False
        
        if path:
            try:
                name = os.path.basename(path)
                video_url = file_url(name)
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
                
                # 비디오 스트리밍 (옵션)
                if WS_VIDEO_STREAM:
                    asyncio.create_task(_stream_video_to_app(self.topic, os.path.join(OUTPUT_DIR, name)))
            except Exception as e:
                log_exc("[REC stop notify]", e)


async def _stream_video_to_app(topic: str, path: str):
    """완성된 비디오를 청크 단위로 스트리밍"""
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
                if not chunk:
                    break
                hdr = pack_media_header(tid, seq, 2)
                await app_broadcast_binary(topic, hdr + chunk)
                seq += 1
        
        await app_broadcast_json(topic, {
            "type": "media_end", "transfer_id": str(tid), "received_chunks": seq
        })
    except Exception as e:
        log_exc("[stream_video]", e)
        await app_broadcast_json(topic, {
            "type": "media_error", "transfer_id": str(tid), "error": str(e)
        })


# ===== WebSocket 엔드포인트 =====
recorder = None  # 전역 Recorder 인스턴스


@router.websocket("/ws/camera")
async def ws_camera(ws: WebSocket):
    """카메라 WebSocket: JPEG 프레임 수신 → YOLO 처리 → 녹화"""
    global recorder
    
    try:
        await ws.accept()
    except Exception as e:
        log_exc("[CAMERA accept]", e)
        return
    
    maybe_set_base_url_from_ws(ws)
    print("📷 /ws/camera 연결됨")
    topic = ws.query_params.get("topic", "public")
    
    # Recorder 초기화
    if recorder is None:
        recorder = Recorder(OUTPUT_DIR, topic)
    
    async def handle_frame(jpg_bytes: bytes) -> bool:
        """프레임 처리: YOLO → 스냅샷/녹화"""
        ts_ms = now_ms()
        
        def work():
            try:
                bgr = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)
                if bgr is None:
                    return None, False, None, None
                
                annotated, triggered = processor.process_frame_and_get_annotated(bgr)
                show_border = is_pulsing() or triggered
                out = draw_border(annotated, show_border)
                
                ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                snap_ok, snap_buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), SNAP_QUALITY])
                
                return (out if ok else annotated), triggered, \
                       (buf.tobytes() if ok else None), \
                       (snap_buf.tobytes() if snap_ok else None)
            except Exception as e:
                log_exc("[CAM work]", e)
                return None, False, None, None
        
        out_bgr, triggered, latest_jpg, snap_jpg = await asyncio.to_thread(work)
        if out_bgr is None:
            return False
        
        try:
            if latest_jpg:
                set_latest_jpg(latest_jpg)
            _push_preroll(ts_ms, jpg_bytes)
        except Exception as e:
            log_exc("[CAM post-work]", e)
        
        # 위험 감지 시 처리
        if triggered:
            try:
                pulse(1200.0)
            except Exception as e:
                log_exc("[pulse]", e)
            
            # 스냅샷
            try:
                if snap_jpg:
                    snap_path, snap_name, ts_sec = _save_snapshot(snap_jpg)
                    await _publish_snapshot(topic, snap_jpg, snap_path, snap_name, ts_sec)
                    if snap_name and ts_sec:
                        snap_url = file_url(snap_name)
                        asyncio.create_task(broadcast_info(
                            direction=last_direction, group_label=last_group_label,
                            group_conf=last_group_conf, dbfs=last_dbfs,
                            raw={"idx": last_raw_idx, "label": last_raw_label, "conf": last_raw_conf},
                            event="snapshot", source="yolo",
                            files={"snapshot_url": snap_url}
                        ))
            except Exception as e:
                log_exc("[CAM snapshot]", e)
            
            # 녹화 시작
            try:
                if not recorder.recording:
                    with _prelock:
                        preroll = list(_prebuf)
                    recorder.start(preroll)
            except Exception as e:
                log_exc("[CAM recorder.start]", e)
        
        # 녹화 중이면 프레임 쓰기
        try:
            if recorder.recording:
                recorder.write(out_bgr)
                recorder.maybe_stop()
        except Exception as e:
            log_exc("[CAM recorder]", e)
        
        return True
    
    try:
        while True:
            try:
                msg = await ws.receive()
            except WebSocketDisconnect:
                break
            except Exception as e:
                log_exc("[CAM receive]", e)
                break
            
            try:
                # Binary JPEG
                if msg.get("bytes"):
                    await handle_frame(msg["bytes"])
                
                # JSON with base64
                elif msg.get("text"):
                    try:
                        data = json.loads(msg["text"])
                        if isinstance(data, dict) and isinstance(data.get("frame_b64"), str):
                            b64 = data["frame_b64"].split(",", 1)[-1]
                            jpg = base64.b64decode(b64 + ("=" * ((-len(b64)) % 4)))
                            await handle_frame(jpg)
                    except Exception as e:
                        log_exc("[CAM parse text]", e)
                        continue
            except Exception as e:
                log_exc("[CAM handle_frame]", e)
                continue
    
    finally:
        try:
            await ws.close()
        except Exception:
            pass
        print("📷 /ws/camera 연결 끊김")