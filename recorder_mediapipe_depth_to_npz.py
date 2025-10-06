import os
import re
import sys
import cv2
import time
import json
import math
import argparse
import numpy as np
import xml.etree.ElementTree as ET
from typing import Optional, Tuple, Dict

# -----------------------------
# Optional Intel RealSense depth
# -----------------------------
try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except Exception:
    HAS_REALSENSE = False

# -----------------------------
# MediaPipe Hands
# -----------------------------
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# -----------------------------
# Config
# -----------------------------
SAVE_DIR = "./demos"
os.makedirs(SAVE_DIR, exist_ok=True)

# Depth (meters) -> forward/backward mapping range for gripper_y
DEPTH_MIN_M = 0.35
DEPTH_MAX_M = 1.00

COUNTDOWN_SEC = 3
TARGET_FPS = 30
COLOR_SIZE = (640, 480)  # (w, h)

# -----------------------------
# Small utils
# -----------------------------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def angle_between(v1, v2, eps=1e-8):
    v1 = v1 / (np.linalg.norm(v1) + eps)
    v2 = v2 / (np.linalg.norm(v2) + eps)
    ang = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return math.acos(ang)

def poll_keys() -> set:
    """Poll keys from OpenCV windows (lowercase letters)."""
    k = cv2.waitKey(1) & 0xFF
    if k == 255:
        return set()
    try:
        return {chr(k).lower()}
    except ValueError:
        return set()

# -----------------------------
# Camera (RealSense or webcam)
# -----------------------------
class DepthCamera:
    """
    RealSense depth+color if available, else fallback to color-only webcam.
    Returns (color_bgr, depth_m) where depth_m is a float32 ndarray in meters or None.
    """
    def __init__(self, color_index=0, width=COLOR_SIZE[0], height=COLOR_SIZE[1], fps=TARGET_FPS):
        self.is_realsense = False
        self.depth_scale = 1.0
        self.align = None
        self.width = width
        self.height = height
        self.fps = fps

        if HAS_REALSENSE:
            try:
                self.pipeline = rs.pipeline()
                config = rs.config()
                config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
                config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
                self.profile = self.pipeline.start(config)
                self.is_realsense = True
                self.align = rs.align(rs.stream.color)
                depth_sensor = self.profile.get_device().first_depth_sensor()
                self.depth_scale = float(depth_sensor.get_depth_scale())
            except Exception:
                self.is_realsense = False

        if not self.is_realsense:
            self.cap = cv2.VideoCapture(color_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            if not self.cap.isOpened():
                raise RuntimeError("No camera found. Connect a webcam or a RealSense camera.")

    def read(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self.is_realsense:
            frames = self.pipeline.wait_for_frames()
            frames = self.align.process(frames)
            depth = frames.get_depth_frame()
            color = frames.get_color_frame()
            if not depth or not color:
                return None, None
            color_img = np.asanyarray(color.get_data())
            depth_img = np.asanyarray(depth.get_data()).astype(np.float32) * self.depth_scale  # meters
            return color_img, depth_img
        else:
            ok, frame = self.cap.read()
            return (frame if ok else None), None

    def release(self):
        if self.is_realsense:
            self.pipeline.stop()
        else:
            self.cap.release()

# -----------------------------
# MediaPipe wrapper
# -----------------------------
class HandTracker:
    def __init__(self, max_num_hands=1, detection_confidence=0.6, tracking_confidence=0.6):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )

    def process(self, bgr):
        if bgr is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self.hands.process(rgb)

    def close(self):
        self.hands.close()

# -----------------------------
# XML -> 24DOF actuator order
# -----------------------------
def build_actuator_order_from_xml(xml_path: str):
    """
    Parse MuJoCo XML and return a deterministic list of 24 actuator names.
    Strategy:
      1) Collect all actuator/@name.
      2) Group by semantic patterns to prefer gripper, wrist, fingers.
      3) Truncate/pad to exactly 24 in a stable order.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    acts = []
    for act in root.findall(".//actuator/*"):
        n = act.get("name")
        if n:
            acts.append(n)

    # Define ordered buckets we care about
    # (Edit patterns if your names differ)
    buckets = [
        # 3 for gripper xyz
        ("gripper", [r"^gripper_x.*", r"^gripper_y.*", r"^gripper_z.*"], 3),
        # 3 for wrist yaw/pitch/roll (or abduction/flex/roll)
        ("wrist", [r"^wrist_.*yaw.*", r"^wrist_.*pitch.*", r"^wrist_.*roll.*",
                   r"^wrist_.*abd.*", r"^wrist_.*flex.*"], 3),
        # Fingers (thumb 4, index 4, middle 3, ring 4, pinky 4) -> total 19
        ("thumb",  [r"^thumb_.*"], 4),
        ("index",  [r"^index_.*"], 4),
        ("middle", [r"^middle_.*"], 3),
        ("ring",   [r"^ring_.*"], 4),
        ("pinky",  [r"^pinky_.*"], 4),
    ]

    used = set()
    order = []

    def pick_by_patterns(patterns, max_count):
        out = []
        # prefer names ending with _act
        for pat in patterns:
            prog = re.compile(pat)
            exact = [a for a in acts if prog.search(a) and a not in used]
            # heuristic: prefer those ending with 'act'
            exact.sort(key=lambda x: (not x.endswith("act"), x))
            for a in exact:
                if a not in used:
                    out.append(a)
                    used.add(a)
                if len(out) >= max_count:
                    break
            if len(out) >= max_count:
                break
        return out

    for _, patterns, maxn in buckets:
        picked = pick_by_patterns(patterns, maxn)
        order.extend(picked)

    # Fill remaining slots (if fewer than 24) with any unused actuators
    if len(order) < 24:
        for a in acts:
            if a not in used:
                order.append(a)
                used.add(a)
            if len(order) >= 24:
                break

    # Truncate to 24 if we have more
    return order[:24]

# -----------------------------
# Feature extraction
# -----------------------------
def extract_features(results, depth_m, color_shape,
                     d_lo=DEPTH_MIN_M, d_hi=DEPTH_MAX_M) -> Dict:
    """
    Compute hand features from MediaPipe + depth.
    Returns dict containing:
      - centroid_px (cx,cy), depth_median
      - landmarks_norm (21,3)
      - normalized proxies: norm_gx, norm_gy, norm_gz in [-1,1]
      - wrist proxies in [0,1]: wrist_yaw, wrist_pitch, wrist_roll
      - finger curls in [0,1]: curls_{thumb,index,middle,ring,pinky}
    """
    H, W = color_shape[:2]
    out = dict(
        centroid_px=None,
        depth_median=None,
        landmarks_norm=None,
        norm_gx=0.0, norm_gy=0.0, norm_gz=0.0,
        wrist_yaw=0.5, wrist_pitch=0.5, wrist_roll=0.5,
        curls_thumb=0.0, curls_index=0.0, curls_middle=0.0, curls_ring=0.0, curls_pinky=0.0
    )

    if not results or not results.multi_hand_landmarks:
        return out

    hls = results.multi_hand_landmarks[0]
    lm = np.array([[p.x, p.y, p.z] for p in hls.landmark], dtype=np.float32)  # normalized
    out["landmarks_norm"] = lm.copy()

    # Centroid in pixels
    cx = int(np.clip(np.mean(lm[:, 0]) * W, 0, W - 1))
    cy = int(np.clip(np.mean(lm[:, 1]) * H, 0, H - 1))
    out["centroid_px"] = (cx, cy)

    # Depth median around centroid
    if depth_m is not None:
        half = 4
        x0 = clamp(cx - half, 0, W - 1)
        x1 = clamp(cx + half, 0, W - 1)
        y0 = clamp(cy - half, 0, H - 1)
        y1 = clamp(cy + half, 0, H - 1)
        patch = depth_m[y0:y1 + 1, x0:x1 + 1]
        patch = patch[np.isfinite(patch)]
        if patch.size > 0:
            out["depth_median"] = float(np.median(patch))

    # Map image X to gripper_x in [-1,1]
    out["norm_gx"] = (2.0 * (cx / max(1, W - 1))) - 1.0
    # Map image Y (top=0) to gripper_z (up/down) in [-1,1]
    out["norm_gz"] = 1.0 - (2.0 * (cy / max(1, H - 1)))
    # Map depth to gripper_y in [-1,1]
    if out["depth_median"] is not None:
        gy = (out["depth_median"] - d_lo) / max(1e-6, (d_hi - d_lo))
        out["norm_gy"] = (2.0 * clamp(gy, 0.0, 1.0)) - 1.0
    else:
        out["norm_gy"] = 0.0

    # Wrist orientation proxies from 2D landmarks
    idx_mcp = lm[5][:2]
    pnk_mcp = lm[17][:2]
    wrist = lm[0][:2]
    mid_tip = lm[12][:2]

    v_yaw = idx_mcp - pnk_mcp
    yaw = math.atan2(v_yaw[1], v_yaw[0])  # [-pi, pi] -> [0,1]
    out["wrist_yaw"] = (yaw + math.pi) / (2 * math.pi)

    v_pitch = mid_tip - wrist
    pitch = math.atan2(-v_pitch[1], v_pitch[0])
    out["wrist_pitch"] = (pitch + math.pi) / (2 * math.pi)

    span = float(np.linalg.norm(v_yaw))
    out["wrist_roll"] = clamp(span * 2.0, 0.0, 1.0)

    # Finger curls (0 open -> 1 closed)
    def curl_for(a, b, c):
        v1 = lm[b] - lm[a]
        v2 = lm[c] - lm[b]
        th = angle_between(v1, v2)
        return clamp(th / math.pi, 0.0, 1.0)

    out["curls_thumb"]  = curl_for(1, 2, 3)
    out["curls_index"]  = curl_for(5, 6, 7)
    out["curls_middle"] = curl_for(9,10,11)
    out["curls_ring"]   = curl_for(13,14,15)
    out["curls_pinky"]  = curl_for(17,18,19)

    return out

# -----------------------------
# Map features -> 24DOF vector
# -----------------------------
def features_to_action_vector(feats: Dict, actuator_order):
    """
    Map recorded features to a 24-D vector aligned with actuator_order.
    Values are in [-1, 1] (normalized).
    """
    gx = float(feats["norm_gx"])
    gy = float(feats["norm_gy"])
    gz = float(feats["norm_gz"])

    wy = 2.0 * float(feats["wrist_yaw"])   - 1.0
    wp = 2.0 * float(feats["wrist_pitch"]) - 1.0
    wr = 2.0 * float(feats["wrist_roll"])  - 1.0

    curls = {
        "thumb":  2.0 * float(feats["curls_thumb"])  - 1.0,
        "index":  2.0 * float(feats["curls_index"])  - 1.0,
        "middle": 2.0 * float(feats["curls_middle"]) - 1.0,
        "ring":   2.0 * float(feats["curls_ring"])   - 1.0,
        "pinky":  2.0 * float(feats["curls_pinky"])  - 1.0,
    }

    vec = np.zeros((len(actuator_order),), dtype=np.float32)

    for i, name in enumerate(actuator_order):
        n = name.lower()
        if "gripper_x" in n:
            vec[i] = gx
        elif "gripper_y" in n:
            vec[i] = gy
        elif "gripper_z" in n:
            vec[i] = gz
        elif "wrist" in n and ("yaw" in n or "abd" in n):
            vec[i] = wy
        elif "wrist" in n and ("pitch" in n or "flex" in n):
            vec[i] = wp
        elif "wrist" in n and "roll" in n:
            vec[i] = wr
        else:
            # fingers: same curl value for that finger's joints
            for finger, val in curls.items():
                if finger in n:
                    vec[i] = val
                    break
            # if unmatched, leave 0
    return vec

# -----------------------------
# Visualization (camera windows)
# -----------------------------
def show_camera_frames(color_bgr, depth_m,
                       dmin=DEPTH_MIN_M, dmax=DEPTH_MAX_M,
                       results=None, status="IDLE", countdown_left=0):
    if color_bgr is not None:
        overlay = color_bgr.copy()
        if results and results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                overlay, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS
            )
        title = f"Camera (color) — {status}"
        if countdown_left > 0:
            title += f" ({int(math.ceil(countdown_left))})"
        cv2.imshow("Camera (color)", overlay)
        cv2.setWindowTitle("Camera (color)", title)

    if depth_m is not None:
        depth_norm = (np.clip(depth_m, dmin, dmax) - dmin) / max(1e-6, (dmax - dmin))
        depth_u8 = (depth_norm * 255).astype(np.uint8)
        depth_cmap = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
        cv2.imshow("Camera (depth)", depth_cmap)

    cv2.waitKey(1)

# -----------------------------
# Save / Inspect
# -----------------------------
def save_npz(buf: Dict, take_idx: int, actuator_order, d_lo: float, d_hi: float) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SAVE_DIR, f"demo_take{take_idx:02d}_{ts}.npz")

    arr = {}
    # Convert lists to arrays
    for k, v in buf.items():
        if k == "landmarks_norm":
            arr[k] = np.stack(v, axis=0).astype(np.float32)  # (N,21,3)
        elif k == "centroid_xy":
            arr[k] = np.array(v, dtype=np.float32)           # (N,2)
        elif k == "action_24dof":
            arr[k] = np.stack(v, axis=0).astype(np.float32)  # (N,24)
        else:
            arr[k] = np.array(v, dtype=np.float32)

    meta = {
        "actuator_names": actuator_order,
        "depth_range_m": [float(d_lo), float(d_hi)],
        "fps_target": TARGET_FPS,
        "schema": {
            "t": "seconds since session start",
            "centroid_xy": "[cx, cy] in pixels",
            "depth_median": "meters (NaN if unavailable)",
            "landmarks_norm": "MediaPipe normalized xyz (N,21,3)",
            "action_24dof": "per-frame 24D vector aligned to actuator_names (normalized [-1,1])",
            "norm_gx/gy/gz": "image/depth-derived proxies in [-1,1]",
            "wrist_*": "0..1 normalized proxies",
            "curls_*": "0..1 open→closed"
        }
    }

    np.savez_compressed(path, **arr, meta=json.dumps(meta))
    return path

def _parse_frames_arg(frames, N):
    """
    frames can be:
      - "all"
      - "start:stop[:step]"  e.g. "0:200:5"
      - comma list "0,1,2,10"
      - int or list/tuple of ints
    Returns a sorted list of unique indices within [0, N).
    """
    if frames is None or frames == "":
        return list(range(min(N, 50)))  # default preview (first 50)
    if isinstance(frames, int):
        return [frames] if 0 <= frames < N else []
    if isinstance(frames, (list, tuple)):
        return [i for i in frames if 0 <= int(i) < N]

    s = str(frames).strip().lower()
    if s == "all":
        return list(range(N))
    if ":" in s:
        parts = [p for p in s.split(":") if p != ""]
        if len(parts) == 2:
            start, stop = int(parts[0]), int(parts[1])
            step = 1
        elif len(parts) == 3:
            start, stop, step = int(parts[0]), int(parts[1]), int(parts[2])
        else:
            raise ValueError(f"Bad frames spec: {frames}")
        start = max(0, start)
        stop  = min(N, stop)
        step  = max(1, step)
        return list(range(start, stop, step))
    # comma list
    idxs = []
    for tok in s.split(","):
        i = int(tok)
        if 0 <= i < N:
            idxs.append(i)
    return sorted(set(idxs))


def inspect_npz(npz_path: str,
                frames: str = "0:50:1",
                round_to: int = 3,
                show_features: bool = True,
                show_deltas: bool = True,
                export_csv: str = ""):
    """
    Print full per-step actuator values with names.
    Args:
      frames: "all", "start:stop[:step]", comma list "0,5,10", or single int.
      round_to: decimals for printing.
      show_features: also print centroid/depth/gx-gy-gz/wrist/curls.
      show_deltas: print Δ from previous step for each DOF.
      export_csv: path to write a CSV (optional).
    """
    import csv
    d = np.load(npz_path, allow_pickle=True)
    if "action_24dof" not in d.files:
        print("No 'action_24dof' in file. Record with the 24-DOF script first.")
        return

    # Load arrays (with NaN-safe handling)
    A = d["action_24dof"]                 # (N, 24)
    N = A.shape[0]
    t   = d["t"] if "t" in d.files else np.arange(N, dtype=np.float32)
    cxcy = d["centroid_xy"] if "centroid_xy" in d.files else np.full((N,2), np.nan, dtype=np.float32)
    depth = d["depth_median"] if "depth_median" in d.files else np.full((N,), np.nan, dtype=np.float32)

    # Optional features
    gx = d["norm_gx"] if "norm_gx" in d.files else np.zeros((N,), np.float32)
    gy = d["norm_gy"] if "norm_gy" in d.files else np.zeros((N,), np.float32)
    gz = d["norm_gz"] if "norm_gz" in d.files else np.zeros((N,), np.float32)
    wy = d["wrist_yaw"] if "wrist_yaw" in d.files else np.zeros((N,), np.float32)
    wp = d["wrist_pitch"] if "wrist_pitch" in d.files else np.zeros((N,), np.float32)
    wr = d["wrist_roll"] if "wrist_roll" in d.files else np.zeros((N,), np.float32)
    ci = d["curls_index"]  if "curls_index"  in d.files else np.zeros((N,), np.float32)
    cm = d["curls_middle"] if "curls_middle" in d.files else np.zeros((N,), np.float32)
    cr = d["curls_ring"]   if "curls_ring"   in d.files else np.zeros((N,), np.float32)
    ck = d["curls_pinky"]  if "curls_pinky"  in d.files else np.zeros((N,), np.float32)
    ct = d["curls_thumb"]  if "curls_thumb"  in d.files else np.zeros((N,), np.float32)

    # Actuator names (from meta)
    names = [f"act_{i:02d}" for i in range(A.shape[1])]
    if "meta" in d.files:
        try:
            meta = json.loads(d["meta"].item())
            if isinstance(meta.get("actuator_names"), (list, tuple)) and len(meta["actuator_names"]) == A.shape[1]:
                names = list(meta["actuator_names"])
        except Exception:
            pass

    idxs = _parse_frames_arg(frames, N)
    if not idxs:
        print(f"No frames selected (N={N}).")
        return

    # Header
    print("== NPZ:", os.path.basename(npz_path), "==")
    print(f"Total frames: {N} | Showing: {len(idxs)} (frames spec: {frames})")
    print("Actuators (ordered):")
    for j, n in enumerate(names):
        print(f"  [{j:02d}] {n}")
    print("-" * 72)

    # Optional CSV export
    writer = None
    fcsv = None
    if export_csv:
        cols = (["frame", "t", "centroid_x", "centroid_y", "depth_m",
                 "norm_gx", "norm_gy", "norm_gz",
                 "wrist_yaw", "wrist_pitch", "wrist_roll",
                 "curl_thumb", "curl_index", "curl_middle", "curl_ring", "curl_pinky"]
                + names)
        fcsv = open(export_csv, "w", newline="")
        writer = csv.writer(fcsv)
        writer.writerow(cols)

    # Print per-step values (and write CSV if requested)
    for i in idxs:
        cx, cy = cxcy[i] if cxcy.shape[0] > i else (np.nan, np.nan)
        header = f"step {i:04d} | t={t[i]:.3f}s | depth={float(depth[i]):.{round_to}f} m | centroid=({float(cx):.{round_to}f},{float(cy):.{round_to}f})"
        print(header)
        if show_features:
            print(f"  proxies: gx={gx[i]: .{round_to}f} gy={gy[i]: .{round_to}f} gz={gz[i]: .{round_to}f} "
                  f"| wrist(y/p/r)={wy[i]: .{round_to}f}/{wp[i]: .{round_to}f}/{wr[i]: .{round_to}f} "
                  f"| curls T/I/M/R/P={ct[i]: .{round_to}f}/{ci[i]: .{round_to}f}/{cm[i]: .{round_to}f}/{cr[i]: .{round_to}f}/{ck[i]: .{round_to}f}")

        print("  DOF values:")
        prev = A[i-1] if (show_deltas and i > 0) else None
        for j, n in enumerate(names):
            val = float(A[i, j])
            line = f"    [{j:02d}] {n:<24} : {val: .{round_to}f}"
            if prev is not None:
                dv = float(val - float(prev[j]))
                line += f"   Δ={dv:+.{round_to}f}"
            print(line)
        print("-" * 72)

        if writer:
            row = [i, float(t[i]), float(cx), float(cy), float(depth[i]),
                   float(gx[i]), float(gy[i]), float(gz[i]),
                   float(wy[i]), float(wp[i]), float(wr[i]),
                   float(ct[i]), float(ci[i]), float(cm[i]), float(cr[i]), float(ck[i])]
            row += [float(x) for x in A[i]]
            writer.writerow(row)

    # Range summary over the selected frames
    sel = A[idxs]
    print("Per-DOF range over shown frames:")
    for j, n in enumerate(names):
        mn = np.nanmin(sel[:, j])
        mx = np.nanmax(sel[:, j])
        print(f"  [{j:02d}] {n:<24} : min={mn: .{round_to}f}, max={mx: .{round_to}f}")

    if writer:
        fcsv.close()
        print(f"[CSV] wrote {len(idxs)} rows to {export_csv}")


# -----------------------------
# Recording session
# -----------------------------
def record_session(xml_path: str,
                   cam_index=0,
                   seconds=600,
                   d_lo=DEPTH_MIN_M,
                   d_hi=DEPTH_MAX_M):
    actuator_order = build_actuator_order_from_xml(xml_path)
    if len(actuator_order) != 24:
        print(f"[Warn] Actuator list has {len(actuator_order)} names, expected 24. Proceeding with {len(actuator_order)}.")

    cam = DepthCamera(color_index=cam_index, width=COLOR_SIZE[0], height=COLOR_SIZE[1], fps=TARGET_FPS)
    tracker = HandTracker()

    print("[Controls] r=start | s=stop+save | q=quit")
    print(f"[Info] Using XML: {xml_path}")
    print(f"[Info] Actuators (24): {actuator_order}")

    # Buffers for one take
    def new_buffers():
        return {
            "t": [],
            "centroid_xy": [],
            "depth_median": [],
            "landmarks_norm": [],
            "norm_gx": [], "norm_gy": [], "norm_gz": [],
            "wrist_yaw": [], "wrist_pitch": [], "wrist_roll": [],
            "curls_thumb": [], "curls_index": [], "curls_middle": [], "curls_ring": [], "curls_pinky": [],
            "action_24dof": []
        }

    buf = new_buffers()
    take_idx = 0
    is_recording = False
    countdown_left = 0.0
    t0 = time.time()

    prev_r = prev_s = prev_q = False

    try:
        while True:
            color_bgr, depth_m = cam.read()
            if color_bgr is None:
                print("[Warn] Camera frame none. Retrying...")
                continue

            results = tracker.process(color_bgr)

            # Visuals
            status = "RECORDING" if (is_recording and countdown_left <= 0) else ("COUNTDOWN" if countdown_left > 0 else "IDLE")
            show_camera_frames(color_bgr, depth_m, dmin=d_lo, dmax=d_hi, results=results, status=status, countdown_left=countdown_left)

            # Keys
            keys = poll_keys()
            r_down = ('r' in keys)
            s_down = ('s' in keys)
            q_down = ('q' in keys)

            if r_down and not prev_r:
                if not is_recording:
                    print("[Rec] Starting new take... (countdown)")
                    buf = new_buffers()
                    is_recording = True
                    countdown_left = float(COUNTDOWN_SEC)
                    t0 = time.time()
                else:
                    print("[Rec] Already recording.")
            if s_down and not prev_s:
                if is_recording:
                    is_recording = False
                    path = save_npz(buf, take_idx, actuator_order, d_lo, d_hi)
                    print(f"[Saved] {path}")
                    take_idx += 1
                else:
                    print("[Info] Not recording; nothing to save.")
            if q_down and not prev_q:
                print("[Exit] Quitting.")
                break

            prev_r, prev_s, prev_q = r_down, s_down, q_down

            # Countdown & logging
            if is_recording:
                if countdown_left > 0.0:
                    countdown_left -= 1.0 / max(1, TARGET_FPS)
                else:
                    feats = extract_features(results, depth_m, color_bgr.shape, d_lo, d_hi)
                    now = time.time() - t0
                    # Append raw features
                    buf["t"].append(now)
                    if feats["centroid_px"] is not None:
                        buf["centroid_xy"].append(feats["centroid_px"])
                    else:
                        buf["centroid_xy"].append((np.nan, np.nan))
                    buf["depth_median"].append(feats["depth_median"] if feats["depth_median"] is not None else np.nan)
                    buf["landmarks_norm"].append(feats["landmarks_norm"] if feats["landmarks_norm"] is not None else np.full((21,3), np.nan, dtype=np.float32))

                    buf["norm_gx"].append(feats["norm_gx"])
                    buf["norm_gy"].append(feats["norm_gy"])
                    buf["norm_gz"].append(feats["norm_gz"])
                    buf["wrist_yaw"].append(feats["wrist_yaw"])
                    buf["wrist_pitch"].append(feats["wrist_pitch"])
                    buf["wrist_roll"].append(feats["wrist_roll"])
                    buf["curls_thumb"].append(feats["curls_thumb"])
                    buf["curls_index"].append(feats["curls_index"])
                    buf["curls_middle"].append(feats["curls_middle"])
                    buf["curls_ring"].append(feats["curls_ring"])
                    buf["curls_pinky"].append(feats["curls_pinky"])

                    # 24-DOF action vector
                    a24 = features_to_action_vector(feats, actuator_order)
                    buf["action_24dof"].append(a24)

            # Session timeout
            # (We keep it simple. If you want, add a --seconds hard stop.)
            # Here, we only quit on 'q'

    finally:
        tracker.close()
        cam.release()
        cv2.destroyAllWindows()

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record 24-DOF hand actions from MediaPipe + Depth using MuJoCo XML actuator order.")
    parser.add_argument("--xml", type=str, default="/mnt/data/custom_env_mediapipe_demo.xml", help="Path to MuJoCo XML (actuator order source)")
    parser.add_argument("--cam", type=int, default=0, help="Camera index for OpenCV / RealSense")
    parser.add_argument("--inspect", type=str, default="", help="Path to NPZ to inspect (skip recording)")
    parser.add_argument("--frames", type=str, default="0:50:1",
                    help='Which frames to show: "all", "start:stop[:step]", "0,5,10", or a single int')
    parser.add_argument("--csv", type=str, default="",
                    help="Optional path to export a CSV of the selected frames")
    args = parser.parse_args()

    if args.inspect:
        inspect_npz(args.inspect, frames=args.frames, export_csv=args.csv, round_to=3)
        sys.exit(0)

    # Record session
    record_session(xml_path=args.xml, cam_index=args.cam)
