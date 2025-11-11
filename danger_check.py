from typing import Dict, List, Optional, Tuple, Union
import os, csv, math
import numpy as np

# =========================
# 0) Config
# =========================

DBFS_FLOOR: float = -50.0  # ignore very quiet frames (dBFS)

GROUP_THRESHOLDS: Dict[str, float] = {
    "Siren":     0.40,
    "Car":       0.40,
    "Shout":     0.40,
    "Alarm":     0.40,
    "Explosion": 0.40,
    "Glass":     0.40,
    "Speech":    0.20,
}

GROUP_AGGREGATION: str = "sum"  # "sum" or "max"

GROUP_KEYWORDS: Dict[str, List[str]] = {
    "Siren":     ["siren", "police car (siren)", "ambulance (siren)", "fire engine", "emergency vehicle"],
    "Car":       ["car horn", "vehicle horn", "car alarm", "tire squeal", "skid", "car passing by", "engine (car)"],
    "Shout":     ["shout", "yell", "scream", "children shouting", "screaming"],
    "Alarm":     ["alarm", "buzzer", "beep", "smoke alarm", "fire alarm", "carbon monoxide alarm"],
    "Explosion": ["explosion", "blast", "boom"],
    "Glass":     ["glass", "shatter", "window breaking", "glass breaking"],
    "Speech":    ["speech", "conversation", "narration", "talking", "babbling", "male speech", "female speech"],
}

MANUAL_INDEX_GROUPS: Dict[str, List[int]] = {
    "Siren":     [317, 318, 319, 320],
    "Car":       [301, 302, 304, 308],
    "Shout":     [6, 9, 10, 11],
    "Alarm":     [382, 394],
    "Explosion": [420],
    "Glass":     [435],
}

# =========================
# 1) Label CSV loader
# =========================

def _load_index_to_label(csv_path: str = "labels/class_labels_indices.csv") -> Dict[int, str]:
    if not os.path.exists(csv_path):
        return {}
    out: Dict[int, str] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        import csv
        reader = csv.DictReader(f)
        for row in reader:
            try:
                idx = int(row["index"])
            except Exception:
                continue
            out[idx] = row.get("display_name", "").strip()
    return out

INDEX_TO_LABEL: Dict[int, str] = _load_index_to_label()

# =========================
# 2) Group mapping
# =========================

def _name_to_group(label_name: str) -> Optional[str]:
    name = (label_name or "").lower()
    for group, kws in GROUP_KEYWORDS.items():
        for kw in kws:
            if kw in name:
                return group
    return None

def _index_to_group(idx: int) -> Optional[str]:
    for g, indices in MANUAL_INDEX_GROUPS.items():
        if idx in indices:
            return g
    if INDEX_TO_LABEL:
        name = INDEX_TO_LABEL.get(idx, "")
        return _name_to_group(name)
    return None

def label_to_group(idx: Optional[int] = None, name: Optional[str] = None) -> Optional[str]:
    if idx is not None:
        g = _index_to_group(int(idx))
        if g:
            return g
    if name is not None:
        return _name_to_group(name)
    return None

# =========================
# 3) Group scoring
# =========================

def group_confidences(probs: np.ndarray, groups: dict, agg: str = "max") -> dict:
    out = {}
    for name, idxs in groups.items():
        vals = probs[idxs]
        if len(vals) == 0:
            out[name] = 0.0
            continue
        if   agg == "max":  score = float(vals.max())
        elif agg == "mean": score = float(vals.mean())
        elif agg == "sum":  score = float(vals.sum())
        else:               raise ValueError("agg must be one of: max|mean|sum")
        out[name] = score
    return out

# =========================
# 4) Top-K aggregation helper
# =========================

TOPK_SIZE = 10

def aggregate_groups_from_topk_vector(probs: np.ndarray, label_names: List[str]) -> Dict[str, float]:
    top_indices = np.argsort(probs)[::-1][:TOPK_SIZE]
    agg: Dict[str, float] = {}
    for idx in top_indices:
        score = float(probs[idx])
        name = label_names[idx] if idx < len(label_names) else None
        g = label_to_group(idx, name)
        if not g:
            continue
        if GROUP_AGGREGATION == "max":
            agg[g] = max(agg.get(g, 0.0), score)
        else:
            agg[g] = agg.get(g, 0.0) + score
    return agg

# =========================
# 5) Thresholding
# =========================

def is_significant_group(group_label: str, group_conf: float, dbfs: float) -> bool:
    if dbfs < DBFS_FLOOR:
        return False
    thr = GROUP_THRESHOLDS.get(group_label, 1.1)
    return float(group_conf) >= float(thr)

def pick_top_significant_group(group_scores: Dict[str, float], dbfs: float) -> Optional[Tuple[str, float]]:
    if not group_scores:
        return None
    candidates = [(g, sc) for g, sc in group_scores.items() if is_significant_group(g, sc, dbfs)]
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0]

# =========================
# 6) Direction (send raw angle)
# =========================
from typing import Optional, Union
Number = Union[int, float]

def get_vibration_angle(direction: Number) -> Optional[float]:
    """
    Return normalized angle in degrees [0, 360).
    If direction is invalid, return None.
    """
    try:
        d = float(direction) % 360.0
    except Exception:
        return None
    return d

# (구버전 호환용) 기존 호출부가 get_vibration_pin()을 부르면 angle만 넘기도록 유지
def get_vibration_pin(direction: Number) -> Optional[int]:
    """
    DEPRECATED: kept only for backward-compat.
    Returns None to signal 'use angle on device'.
    """
    return None

