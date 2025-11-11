#!/usr/bin/env python3
# yamnet_module.py — drop-in (2 buckets: 사이렌 / 경보·알람) + throttled logging(기본 1s)
# - DANGER_T_SIREN / DANGER_T_ALARM / CAT_FLOOR / YAMNET_LOG_INTERVAL 등으로 튜닝
# - Public APIs:
#     classify_sound_with_confidence(waveform: np.ndarray, sr: int) -> dict
#     classify_bytes(audio_bytes: bytes, sr_hint: int = 16000) -> dict

import os, time, csv
from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# ---------------------------
# Logging / Env defaults
# ---------------------------
VERBOSE = int(os.getenv("YAMNET_VERBOSE", "1"))               # 0..3
LOG_TOPK = int(os.getenv("YAMNET_LOG_TOPK", "5"))
LOG_CAT_MIN = float(os.getenv("YAMNET_LOG_CAT_MIN", "0.05"))
LOG_INTERVAL = float(os.getenv("YAMNET_LOG_INTERVAL", "1.0")) # 최소 로그 간격(초)
_last_log_ts = 0.0

def _log(level: int, msg: str) -> None:
    if VERBOSE >= level:
        print(msg, flush=True)

def _log_throttled(level: int, msg: str) -> None:
    global _last_log_ts
    now = time.time()
    if VERBOSE >= level and (now - _last_log_ts) >= LOG_INTERVAL:
        print(msg, flush=True)
        _last_log_ts = now

# 기본 CPU
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TFHUB_CACHE_DIR", "/tmp/tfhub")

# ---------------------------
# Model / Runtime cache
# ---------------------------
_YAMNET = None
_LABEL_NAMES: Optional[List[str]] = None
_CATEGORY_INDEXES: Optional[Dict[str, List[int]]] = None

# Frame aggregation: "max" or "mean"
FRAME_AGG = os.getenv("FRAME_AGG", "max").lower()
TOPK = int(os.getenv("TOPK", "10"))

# ---------------------------
# Category config (2 buckets: 사이렌 / 경보·알람)
# ---------------------------
# 임계값(기본값은 환경변수로도 제어 가능)
DANGER_T_SIREN = float(os.getenv("DANGER_T_SIREN", "0.30"))
DANGER_T_ALARM = float(os.getenv("DANGER_T_ALARM", "0.45"))   # 알람/벨/초인종은 기본 더 높게
DEFAULT_FALLBACK_THRESH: float = float(os.getenv("DANGER_THRESH", "0.35"))

CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "사이렌": [
        "siren", "warning siren",
        "police car siren", "ambulance siren",
        "fire engine siren", "fire truck siren", "fire siren",
        "civil defense siren"
    ],
    "경보/알람": [
        # 각종 알람류
        "fire alarm", "smoke alarm", "carbon monoxide alarm", "co alarm",
        "evacuation alarm", "emergency alarm", "alarm bell", "alarm buzzer",
        "alarm beeping", "alarm tone",
        # 차량 경보/경적
        "car alarm", "vehicle alarm", "anti-theft alarm",
        "car horn", "vehicle horn", "air horn", "horn", "honk", "honking", "klaxon",
        # 벨/초인종/전화
        "doorbell", "door bell", "buzzer", "intercom buzzer",
        "telephone bell", "telephone bell ringing", "telephone ringing",
        "phone ringing", "ringtone", "ringer",
        "school bell", "alarm clock", "bell", "bell ringing", "church bell", "bicycle bell",
        # 기타 경보성 비프
        "beep", "beeping", "warning beep"
    ],
}

DEFAULT_CAT_THRESH: Dict[str, float] = {
    "사이렌": DANGER_T_SIREN,
    "경보/알람": DANGER_T_ALARM,
}
CATEGORY_IS_DANGER: Dict[str, bool] = {"사이렌": True, "경보/알람": True}

def _parse_cat_kv_env(env_name: str, defaults: Dict[str, float]) -> Dict[str, float]:
    """
    env 예:
      CAT_THRESH="사이렌:0.28;경보/알람:0.50"
      CAT_FLOOR="사이렌:0.10;경보/알람:0.20"
    """
    out = dict(defaults)
    spec = os.getenv(env_name, "").strip()
    if not spec:
        return out
    for chunk in spec.split(";"):
        if ":" not in chunk:
            continue
        k, v = chunk.split(":", 1)
        try:
            out[k.strip()] = float(v.strip())
        except:
            pass
    return out

# 1차 임계: 통과 기준
CAT_THRESH_MAP: Dict[str, float] = _parse_cat_kv_env("CAT_THRESH", DEFAULT_CAT_THRESH)
# 2차 플로어: 임계 미달이어도 이 값 넘으면 위험 처리(보수적으로)
DEFAULT_CAT_FLOOR = {"사이렌": 0.10, "경보/알람": 0.20}
CAT_FLOOR_MAP: Dict[str, float] = _parse_cat_kv_env("CAT_FLOOR", DEFAULT_CAT_FLOOR)

# ---------------------------
# Audio helpers
# ---------------------------
def _to_mono_float32(waveform: np.ndarray) -> np.ndarray:
    if waveform is None or waveform.size == 0:
        return np.zeros((0,), dtype=np.float32)
    x = waveform
    if x.ndim == 2:
        x = x.mean(axis=0 if x.shape[0] < x.shape[1] else 1)
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    else:
        x = x.astype(np.float32, copy=False)
    return np.clip(x, -1.0, 1.0)

def _resample_to_16k(mono: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000 or mono.size == 0:
        return mono
    target_len = int(round(mono.shape[0] * 16000.0 / float(sr)))
    if target_len <= 0:
        return np.zeros((0,), dtype=np.float32)
    return tf.signal.resample(tf.convert_to_tensor(mono, tf.float32), target_len).numpy()

# ---------------------------
# Labels / categories
# ---------------------------
def _load_labels_from_csv(filepath: str) -> List[str]:
    names_map: Dict[int, str] = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            idx_str = (row.get("index") or "").strip()
            if not idx_str:
                continue
            try:
                idx = int(idx_str)
            except:
                continue
            disp = (row.get("display_name") or row.get("display_str") or row.get("mid") or f"class_{idx}").strip()
            names_map[idx] = disp
    if not names_map:
        raise ValueError("No labels parsed from CSV")
    return [names_map.get(i, f"class_{i}") for i in range(max(names_map.keys()) + 1)]

def _get_yamnet():
    global _YAMNET
    if _YAMNET is not None:
        return _YAMNET
    url_or_dir = os.getenv("YAMNET_HUB_URL", "https://tfhub.dev/google/yamnet/1")
    os.makedirs(os.environ["TFHUB_CACHE_DIR"], exist_ok=True)
    _log(1, f"[YAMNET] loading from: {url_or_dir}")
    t0 = time.time()
    _YAMNET = hub.load(url_or_dir)
    try:
        _ = _YAMNET(tf.zeros([16000], dtype=tf.float32))
    except Exception as e:
        _log(1, f"[YAMNET] warmup warn: {e}")
    _log(1, f"[YAMNET] ready. load_time={time.time()-t0:.2f}s")
    return _YAMNET

def _get_label_names(model=None) -> List[str]:
    global _LABEL_NAMES
    if _LABEL_NAMES is not None:
        return _LABEL_NAMES
    env_csv = os.getenv("YAMNET_LABELS_CSV", "").strip()
    if env_csv and os.path.exists(env_csv):
        try:
            _LABEL_NAMES = _load_labels_from_csv(env_csv)
            return _LABEL_NAMES
        except Exception as e:
            _log(1, f"[YAMNET] labels from $YAMNET_LABELS_CSV failed: {e}")
    local_csv = "labels/class_labels_indices.csv"
    if os.path.exists(local_csv):
        try:
            _LABEL_NAMES = _load_labels_from_csv(local_csv)
            return _LABEL_NAMES
        except Exception as e:
            _log(1, f"[YAMNET] labels from {local_csv} failed: {e}")
    try:
        if model is None:
            model = _get_yamnet()
        class_map_asset = model.class_map_path().numpy().decode("utf-8")
        _LABEL_NAMES = _load_labels_from_csv(class_map_asset)
        return _LABEL_NAMES
    except Exception as e:
        _log(1, f"[YAMNET] labels from hub asset failed: {e}")
    _LABEL_NAMES = [f"class_{i}" for i in range(521)]
    return _LABEL_NAMES

def _build_category_indexes(label_names: List[str], categories: Dict[str, List[str]]) -> Dict[str, List[int]]:
    ln = [str(x).lower() for x in label_names]
    res: Dict[str, List[int]] = {}
    for cat, kws in categories.items():
        idxs = [i for i, name in enumerate(ln) if any(kw in name for kw in kws)]
        res[cat] = sorted(set(idxs))
    return res

def _get_category_indexes(model=None) -> Dict[str, List[int]]:
    global _CATEGORY_INDEXES
    if _CATEGORY_INDEXES is not None:
        return _CATEGORY_INDEXES
    names = _get_label_names(model)
    _CATEGORY_INDEXES = _build_category_indexes(names, CATEGORY_KEYWORDS)
    return _CATEGORY_INDEXES

# ---------------------------
# Public APIs
# ---------------------------
def classify_sound_with_confidence(waveform: np.ndarray, sr: int) -> dict:
    t0 = time.time()
    mono = _to_mono_float32(waveform)
    mono_16k = _resample_to_16k(mono, sr)

    if mono_16k.size == 0:
        thresholds = {k: CAT_THRESH_MAP.get(k, DEFAULT_FALLBACK_THRESH) for k in CATEGORY_KEYWORDS}
        return {
            "category_label": "safe", "category_conf": 0.0, "is_danger": False,
            "raw_idx": -1, "raw_label": "", "raw_conf": 0.0,
            "probs": np.zeros((len(_get_label_names()) or 521,), dtype=np.float32),
            "label_names": _get_label_names(), "topk_raw": [],
            "category_scores": {k: 0.0 for k in CATEGORY_KEYWORDS},
            "category_thresholds": thresholds, "group_label": "safe", "group_conf": 0.0
        }

    model = _get_yamnet()
    names = _get_label_names(model)
    cat_indexes = _get_category_indexes(model)

    scores, _, _ = model(mono_16k)  # [frames, num_classes]
    framewise = scores.numpy()
    probs = framewise.max(axis=0) if FRAME_AGG != "mean" else framewise.mean(axis=0)

    raw_idx = int(np.argmax(probs))
    raw_conf = float(probs[raw_idx])
    raw_label = names[raw_idx]

    cat_scores = {cat: (float(np.max(probs[idxs])) if idxs else 0.0)
                  for cat, idxs in cat_indexes.items()}
    thresholds = {k: CAT_THRESH_MAP.get(k, DEFAULT_FALLBACK_THRESH) for k in cat_indexes}

    passed = [(c, s) for c, s in cat_scores.items() if s >= thresholds.get(c, DEFAULT_FALLBACK_THRESH)]
    forced = [(c, s) for c, s in cat_scores.items() if s >= CAT_FLOOR_MAP.get(c, 1.0)]

    if passed:
        passed.sort(key=lambda x: x[1], reverse=True)
        category_label, category_conf = passed[0]
        is_danger = CATEGORY_IS_DANGER.get(category_label, True)
    elif forced:
        forced.sort(key=lambda x: x[1], reverse=True)
        category_label, category_conf = forced[0]
        is_danger = CATEGORY_IS_DANGER.get(category_label, True)
    else:
        category_label = "safe"
        category_conf = float(1.0 - max(cat_scores.values() or [0.0]))
        is_danger = False

    top_idx = np.argsort(probs)[::-1][:TOPK]
    topk_raw = [(int(i), names[i], float(probs[i])) for i in top_idx]

    # ---- throttled main log (기본 1초 간격) ----
    _log_throttled(
        1,
        f"[YAMNET] label={category_label} conf={category_conf:.2f} "
        f"raw={raw_label}({raw_conf:.2f}) danger={is_danger} "
        f"latency={time.time()-t0:.3f}s"
    )

    if VERBOSE >= 2:
        from itertools import islice
        _log_throttled(2, "[YAMNET] top-k:")
        for j, (i, nm, sc) in enumerate(islice(topk_raw, LOG_TOPK)):
            _log_throttled(2, f"  #{j+1:02d} {nm} ({sc:.2f})")
        lines = []
        for c, sc in sorted(cat_scores.items(), key=lambda x: x[1], reverse=True):
            if sc >= LOG_CAT_MIN:
                thr = thresholds.get(c, DEFAULT_FALLBACK_THRESH)
                mark = " *" if sc >= thr else ""
                lines.append(f"  {c:8s} {sc:.3f} thr={thr:.2f}{mark}")
        if lines:
            _log_throttled(2, "[YAMNET] categories(>={:.2f}):\n{}".format(LOG_CAT_MIN, "\n".join(lines)))

    return {
        "category_label": category_label, "category_conf": category_conf, "is_danger": is_danger,
        "raw_idx": raw_idx, "raw_label": raw_label, "raw_conf": raw_conf,
        "probs": probs, "label_names": names, "topk_raw": topk_raw,
        "category_scores": cat_scores, "category_thresholds": thresholds,
        "group_label": category_label, "group_conf": category_conf,
    }

def classify_bytes(audio_bytes: bytes, sr_hint: int = 16000) -> dict:
    try:
        audio_tf, sr_tf = tf.audio.decode_wav(audio_bytes)
        arr = audio_tf.numpy()
        if arr.ndim == 2 and arr.shape[1] > 1:
            arr = arr.mean(axis=1)
        sr = int(sr_tf.numpy())
        return classify_sound_with_confidence(arr, sr)
    except Exception:
        arr = np.frombuffer(audio_bytes, dtype=np.float32)
        return classify_sound_with_confidence(arr, sr_hint)
