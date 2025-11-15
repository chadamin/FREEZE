# camera_module.py (drop-in, BGR 표준/색반전 수정)
import os
import time
import cv2
import tempfile
import threading
import requests
import numpy as np
from typing import Dict, Optional, Tuple, Tuple as Tup, List
from ultralytics import YOLO

# =========================
# 환경변수(없으면 기본값)
# =========================
APP_SNAPSHOT_URL = os.getenv("APP_SNAPSHOT_URL", "http://3.37.15.3:8000/api/upload/snapshot")
APP_VIDEO_URL    = os.getenv("APP_VIDEO_URL",    "http://3.37.15.3:8000/api/upload/video")
DEVICE_ID        = os.getenv("DEVICE_ID",        "pi-cam-01")

IMG_SIZE     = int(os.getenv("IMG_SIZE", "640"))         # YOLO 입력 크기
CONF_THR     = float(os.getenv("CONF_THR", "0.35"))      # 신뢰도 임계
_RAW_CLASSES = os.getenv("TARGET_CLASSES", "0,1,2,3,5,6,7").strip()
if _RAW_CLASSES == "":
    CLASSES: Optional[List[int]] = None   # 빈 문자열이면 전체 허용
else:
    CLASSES = [int(x) for x in _RAW_CLASSES.split(",")]

MIN_DETS     = int(os.getenv("MIN_DETS", "1"))           # 트리거 최소 검출수
COOLDOWN_S   = float(os.getenv("COOLDOWN_SEC", "3"))     # 재트리거 쿨다운
VIDEO_SEC    = float(os.getenv("VIDEO_SEC", "15"))       # 녹화 길이
JPG_QLTY     = int(os.getenv("JPG_QUALITY", "90"))       # 스냅샷 JPEG 퀄리티
FPS_ASSUME   = float(os.getenv("FPS_ASSUME", "20"))      # 파일 저장 시 가정 FPS
MIN_BOX_AREA = int(os.getenv("MIN_BOX_AREA", "0"))       # 너무 작은 박스 무시(픽셀^2). 0=비활성
MATCH_MAX_DIST = float(os.getenv("MATCH_MAX_DIST", "120"))

# YOLO 이전 회전(반시계 90도). 기본값 true로 설정해 바로 적용되게 함.
ROTATE_BEFORE_YOLO = os.getenv("ROTATE_BEFORE_YOLO", "true").lower() == "true"

# 입력 프레임 색 공간: PiCamera2면 RGB가 기본 -> 1, 일반 OpenCV 캡쳐면 BGR -> 0
CAM_INPUT_IS_RGB = os.getenv("CAM_INPUT_IS_RGB", "1") == "1"

# =========================
# 내부 유틸
# =========================
def _boxes_after_area_filter(res, min_area: int):
    """면적 필터 적용 후 박스 리스트 반환"""
    if len(res.boxes) == 0:
        return []
    if min_area <= 0:
        return list(res.boxes)
    keep = []
    for b in res.boxes:
        x1, y1, x2, y2 = map(float, b.xyxy[0])
        if (x2 - x1) * (y2 - y1) >= min_area:
            keep.append(b)
    return keep

def _safe_post(url: str, files: Dict, data: Dict, timeout: int):
    try:
        requests.post(url, files=files, data=data, timeout=timeout)
    except Exception as e:
        print(f"[UPLOAD] error posting to {url}: {e}")

def _center_and_area(b):
    x1, y1, x2, y2 = map(float, b.xyxy[0])
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    area = (x2 - x1) * (y2 - y1)
    return cx, cy, area

def _dist(a, b):
    ax, ay = a
    bx, by = b
    dx, dy = ax - bx, ay - by
    return (dx*dx + dy*dy) ** 0.5

def _match_prev(curr: list[tuple[int, float, float]],
                prev: list[tuple[int, float, float, float]],
                now_ts: float):
    """속도 계산용 간단 매칭(트리거에는 사용 안 함)"""
    results = []
    used_prev = set()
    for cls, cx, cy in curr:
        best_i = -1
        best_d = 1e9
        best_prev = None
        for i, (pcls, pcx, pcy, pts) in enumerate(prev):
            if i in used_prev or pcls != cls:
                continue
            d = _dist((cx, cy), (pcx, pcy))
            if d < best_d:
                best_d = d
                best_i = i
                best_prev = (pcx, pcy, pts)
        speed = 0.0
        if best_prev and best_d <= MATCH_MAX_DIST:
            pcx, pcy, pts = best_prev
            dt = max(1e-3, now_ts - pts)
            speed = best_d / dt   # px/sec
            used_prev.add(best_i)
        results.append({'cls': cls, 'cx': cx, 'cy': cy, 'speed': speed})
    return results

# =========================
# 메인 클래스
# =========================
class CameraEventProcessor:
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model = YOLO(model_path)

        self._rec_lock = threading.Lock()
        self._recording = False
        self._rec_writer: Optional[cv2.VideoWriter] = None
        self._rec_deadline = 0.0
        self._video_size: Optional[Tuple[int, int]] = None
        self._rec_path: Optional[str] = None

        self._last_trigger_ts = 0.0
        self._prev_dets: list[tuple[int, float, float, float]] = []

        self._rotate_before = ROTATE_BEFORE_YOLO
        self._input_is_rgb = CAM_INPUT_IS_RGB
        print(f"[BOOT] camera_module loaded. ROTATE_BEFORE_YOLO={self._rotate_before} CAM_INPUT_IS_RGB={self._input_is_rgb}")

    def _normalize_to_bgr(self, frame: np.ndarray) -> np.ndarray:
        """입력을 내부 표준(BGR)로 통일."""
        if frame is None or frame.size == 0:
            return frame
        if self._input_is_rgb:
            # PiCamera2 등 RGB 입력을 BGR로 1회 변환
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # 일반 OpenCV 캡처는 원래 BGR
        return frame

    def process_frame_and_get_annotated(self, frame: np.ndarray) -> Tup[np.ndarray, bool]:
        """
        frame: 입력 프레임 (RGB 또는 BGR). CAM_INPUT_IS_RGB로 제어.
        return: (annotated_bgr, triggered_now)
        """
        # 0) 내부 표준 색공간(BGR)로 정규화
        bgr = self._normalize_to_bgr(frame)

        # 0.5) YOLO 이전 회전 적용 (BGR 유지)
        if self._rotate_before:
            # 반시계 90도: (H, W, C) -> (W, H, C)
            bgr_in = np.rot90(bgr, 1).copy()
        else:
            bgr_in = bgr

        h, w = bgr_in.shape[:2]

        # 1) YOLO 추론 (BGR, 회전된 프레임 기준)
        res = self.model.predict(
            bgr_in, imgsz=IMG_SIZE, conf=CONF_THR, classes=CLASSES, verbose=False
        )[0]

        filtered_boxes = _boxes_after_area_filter(res, MIN_BOX_AREA)

        now_ts = time.time()
        curr_centers: List[tuple[int, float, float]] = []
        curr_areas: List[float] = []
        for b in filtered_boxes:
            cls = int(b.cls)
            cx, cy, area = _center_and_area(b)
            curr_centers.append((cls, cx, cy))
            curr_areas.append(area)

        matched = _match_prev(curr_centers, self._prev_dets, now_ts)
        self._prev_dets = [(m['cls'], m['cx'], m['cy'], now_ts) for m in matched]

        # 2) 주석(박스/라벨)도 회전된 프레임 위에 그림 (BGR)
        annotated = bgr_in.copy()
        try:
            for i, b in enumerate(filtered_boxes):
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                cls = int(b.cls)
                conf = float(b.conf)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                name = res.names.get(cls, str(cls))
                label = f"{name} {conf:.2f}"
                cv2.putText(annotated, label, (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        except Exception:
            # Ultralytics가 제공하는 플로팅(※ BGR 반환)
            annotated = res.plot()

        # 3) 비디오 파일에 쓰기 (BGR, 회전된 프레임)
        self._maybe_write(annotated)

        # 4) 트리거 판정
        det_n = len(filtered_boxes)
        size_ok = any(area >= max(1, MIN_BOX_AREA) for area in curr_areas)
        now = now_ts

        debug_info = {
            "det_n": det_n,
            "size_ok": size_ok,
            "MIN_DETS": MIN_DETS,
            "COOLDOWN_S": COOLDOWN_S,
            "since_last": round(now - self._last_trigger_ts, 2),
            "recording": self._recording,
            "CONF_THR": CONF_THR,
            "MIN_BOX_AREA": MIN_BOX_AREA,
            "CLASSES": "ALL" if CLASSES is None else CLASSES,
        }
        print(f"[DEBUG] {debug_info}")

        should_trigger = (
            size_ok and
            det_n >= MIN_DETS and
            (now - self._last_trigger_ts) >= COOLDOWN_S and
            (not self._recording)
        )

        triggered_now = False
        if should_trigger:
            self._last_trigger_ts = now
            triggered_now = True

            # 스냅샷 인코드/저장/업로드 (BGR 그대로 인코드)
            ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), JPG_QLTY])
            if ok:
                snap_path = tempfile.NamedTemporaryFile(prefix="snap_", suffix=".jpg", delete=False).name
                try:
                    with open(snap_path, "wb") as f:
                        f.write(buf.tobytes())
                    print(f"[SNAPSHOT] saved local -> {snap_path}")
                except Exception as e:
                    print("[SNAPSHOT] local save error:", e)
            else:
                print("[SNAPSHOT] encode failed")

            if ok and APP_SNAPSHOT_URL and "<APP>" not in APP_SNAPSHOT_URL:
                try:
                    cls_names = [res.names[int(b.cls)] for b in filtered_boxes]
                except Exception:
                    cls_names = []
                meta = {
                    "device_id": DEVICE_ID,
                    "ts": int(now * 1000),
                    "det_count": det_n,
                    "classes": ",".join(cls_names),
                    "trigger_by": "size",
                }
                print(f"[SNAPSHOT] uploading -> {APP_SNAPSHOT_URL} meta={meta}")
                _safe_post(APP_SNAPSHOT_URL, {"file": ("snapshot.jpg", buf.tobytes(), "image/jpeg")}, meta, timeout=5)
            else:
                if not ok:
                    pass  # 위에서 이미 메시지 출력
                elif not APP_SNAPSHOT_URL or "<APP>" in APP_SNAPSHOT_URL:
                    print("[SNAPSHOT] skipped: APP_SNAPSHOT_URL not set or contains <APP>")

            # 녹화 시작: 회전된 프레임 크기 (width,height)=(w,h)
            self._start_recording((w, h))

        return annotated, triggered_now

    # -------------------------------
    def _start_recording(self, size: Tuple[int, int]):
        with self._rec_lock:
            if self._recording:
                return
            self._video_size = size
            self._rec_path = tempfile.NamedTemporaryFile(prefix="event_", suffix=".mp4", delete=False).name
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            # size는 (width, height)
            self._rec_writer = cv2.VideoWriter(self._rec_path, fourcc, FPS_ASSUME, size)
            if not self._rec_writer or not self._rec_writer.isOpened():
                print("[RECORD] open failed; skip recording")
                self._rec_writer = None
                return
            self._recording = True
            self._rec_deadline = time.time() + VIDEO_SEC
            print(f"[RECORD] start -> {self._rec_path} size={size} fps={FPS_ASSUME}")

        threading.Thread(target=self._finish_when_deadline, daemon=True).start()

    def _maybe_write(self, bgr: np.ndarray):
        if not self._recording:
            return
        with self._rec_lock:
            if self._rec_writer:
                try:
                    # BGR 프레임 그대로
                    self._rec_writer.write(bgr)
                except Exception as e:
                    print("[RECORD] write error:", e)

    def _finish_when_deadline(self):
        while time.time() < self._rec_deadline:
            time.sleep(0.05)

        with self._rec_lock:
            path = self._rec_path
            try:
                if self._rec_writer:
                    self._rec_writer.release()
            finally:
                self._rec_writer = None
                self._recording = False
                self._rec_path = None
                self._video_size = None
            print(f"[RECORD] finish -> {path}")

        if path and APP_VIDEO_URL and "<APP>" not in APP_VIDEO_URL:
            meta = {"device_id": DEVICE_ID, "ts": int(time.time() * 1000)}
            print(f"[VIDEO] uploading -> {APP_VIDEO_URL} file={os.path.basename(path)}")
            try:
                with open(path, "rb") as f:
                    files = {"file": (os.path.basename(path), f, "video/mp4")}
                    requests.post(APP_VIDEO_URL, files=files, data=meta, timeout=60)
            except Exception as e:
                print("[VIDEO] upload error:", e)
            finally:
                try:
                    os.remove(path)
                except Exception:
                    pass
        else:
            print("[VIDEO] skipped: APP_VIDEO_URL not set or contains <APP>")
