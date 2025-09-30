#!/usr/bin/env python3
"""
Demo Recorder: RealSense Depth + MediaPipe Hands → per-episode .npz

What it does
------------
- Opens Intel RealSense (color + aligned depth) and a MediaPipe Hands tracker.
- Detects the red goal area (largest red blob) in the color image.
- (Optional) Detects a cube with an ArUco marker; otherwise estimates cube
  position during grasp from the hand contact point.
- Computes per-frame hand 2D position, depth (meters), in-plane velocity, and
  a binary grasp signal from thumb–index distance (pinch).
- Records observations + actions into memory while "recording" is ON.
- Saves ONE episode per .npz with arrays: obs, acts, rews, dones, goal, meta.

Controls
--------
- Press 'r' to start/stop recording an EPISODE.
- Press 's' to STOP & SAVE current episode to .npz (also stops recording if on).
- Press 'q' to quit the program.

Dependencies
------------
- OpenCV (cv2) + opencv-contrib-python (for ArUco)
- mediapipe
- pyrealsense2
- numpy

Notes
-----
- Calibrate HSV thresholds for red in your lighting.
- If you can, place a small ArUco marker on the cube (e.g., DICT_4X4_50 id=0) for robust tracking.
- The action logged is hand velocity in the image plane (dx, dy) normalized to [-1,1].
- The observation vector is documented below in OBS_LAYOUT.

"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import cv2

# MediaPipe
import mediapipe as mp

# RealSense
import pyrealsense2 as rs

# ------------------------------ Config ------------------------------------
OBS_LAYOUT = [
    "hand_x",        # normalized [0,1] image x
    "hand_y",        # normalized [0,1] image y
    "hand_z_m",      # depth meters at palm center
    "hand_vx",       # normalized velocity in x (per frame)
    "hand_vy",       # normalized velocity in y (per frame)
    "grasp",         # 0/1 pinch (thumb-index distance below thresh)
    "goal_x",        # normalized [0,1] center of largest red blob
    "goal_y",        # normalized [0,1]
    "cube_x",        # normalized [0,1] (NaN if unknown)
    "cube_y",        # normalized [0,1] (NaN if unknown)
]

ACTION_LAYOUT = [
    "dx_cmd",        # same as hand_vx (proxy action)
    "dy_cmd",        # same as hand_vy
]

# Default HSV thresholds for detecting RED (two bands around 0° and 180°)
RED_LOWER_1 = np.array([0, 120, 70])
RED_UPPER_1 = np.array([10, 255, 255])
RED_LOWER_2 = np.array([170, 120, 70])
RED_UPPER_2 = np.array([180, 255, 255])

# Thumb–Index pinch distance threshold (in pixels on 640×480). Scale by width.
PINCH_THRESH_PX_AT_640 = 30

# ArUco dictionary and id for the cube (optional)
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
ARUCO_DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
CUBE_TAG_ID = 0  # change if needed

# Grace + success hold settings
RELEASE_GRACE_FRAMES = 10   # keep last cube pos for ~0.3s at 30 FPS
SUCCESS_HOLD_FRAMES  = 5    # require K consecutive frames inside goal before success
GOAL_LOCK_FRAMES     = 8    # frames to stabilize/lock red goal at REC start    # require K consecutive frames inside goal before success

# ------------------------------ Utilities ---------------------------------

def norm_xy(x: float, y: float, w: int, h: int) -> Tuple[float,float]:
    return x / float(w), y / float(h)


def denorm_xy(nx: float, ny: float, w: int, h: int) -> Tuple[int,int]:
    return int(np.clip(nx, 0, 1) * (w - 1)), int(np.clip(ny, 0, 1) * (h - 1))


def detect_red_center(bgr: np.ndarray) -> Optional[Tuple[int,int,int]]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, RED_LOWER_1, RED_UPPER_1)
    mask2 = cv2.inRange(hsv, RED_LOWER_2, RED_UPPER_2)
    mask = cv2.medianBlur(mask1 | mask2, 5)
    # morphology to clean
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 200:  # too small
        return None
    (x, y), radius = cv2.minEnclosingCircle(c)
    return int(x), int(y), int(radius)


def detect_aruco_center(gray: np.ndarray, marker_id: int=CUBE_TAG_ID) -> Optional[Tuple[int,int]]:
    corners, ids, _ = ARUCO_DETECTOR.detectMarkers(gray)
    if ids is None:
        return None
    ids = ids.flatten()
    for i, pts in zip(ids, corners):
        if i == marker_id:
            pts = pts.reshape(-1, 2)
            cx, cy = pts.mean(axis=0)
            return int(cx), int(cy)
    return None


def get_depth_at_pixel(depth_frame: rs.depth_frame, x: int, y: int) -> float:
    if x < 0 or y < 0 or x >= depth_frame.get_width() or y >= depth_frame.get_height():
        return float("nan")
    depth = depth_frame.get_distance(x, y)
    return float(depth) if depth > 0 else float("nan")


# ------------------------------ Main Recorder -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Demo recorder: RealSense + MediaPipe Hands → NPZ per episode")
    parser.add_argument("--outdir", type=str, default="./demos_npz", help="Directory to save .npz episodes")
    parser.add_argument("--camera_width", type=int, default=640)
    parser.add_argument("--camera_height", type=int, default=480)
    parser.add_argument("--camera_fps", type=int, default=30)
    parser.add_argument("--min_detection_conf", type=float, default=0.6)
    parser.add_argument("--min_tracking_conf", type=float, default=0.5)
    parser.add_argument("--show_debug", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---------- RealSense setup ----------
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, args.camera_width, args.camera_height, rs.format.z16, args.camera_fps)
    cfg.enable_stream(rs.stream.color, args.camera_width, args.camera_height, rs.format.bgr8, args.camera_fps)

    profile = pipe.start(cfg)
    align = rs.align(rs.stream.color)  # align depth to color stream

    # ---------- MediaPipe Hands ----------
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=args.min_detection_conf,
        min_tracking_confidence=args.min_tracking_conf,
        model_complexity=1,
    )

    # State buffers for an episode
    recording = False
    frames = []  # raw preview frames for debugging video (not saved to npz)
    obs_buf = []  # list of arrays shaped (obs_dim,)
    act_buf = []  # list of arrays shaped (2,)
    rew_buf = []  # list of floats
    done_buf = []  # list of bools

    episode_idx = len(list(outdir.glob("episode_*.npz")))

    # For velocity computation
    prev_nx, prev_ny = None, None
    # Latching for cube position after release + success hold counter
    grace_left = 0
    last_cx, last_cy = np.nan, np.nan
    success_hold = 0
    # Goal lock state
    goal_locked = False
    lock_count = 0
    locked_gx_px = np.nan
    locked_gy_px = np.nan
    locked_goal_r = 0
    # Latching for cube position after release + success hold counter
    grace_left = 0
    last_cx, last_cy = np.nan, np.nan
    success_hold = 0

    print("Press 'r' to start/stop recording, 's' to save episode, 'q' to quit.")

    try:
        while True:
            frameset = pipe.wait_for_frames()
            frameset = align.process(frameset)
            depth_frame = frameset.get_depth_frame()
            color_frame = frameset.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())  # uint16 depth in mm units (via scale)
            h, w, _ = color.shape

            # --- Detect goal (red area) ---
            red = detect_red_center(color)
            if recording and not goal_locked:
                if red is not None:
                    gx_px, gy_px, goal_r = red
                    lock_count += 1
                    if np.isnan(locked_gx_px):
                        locked_gx_px, locked_gy_px, locked_goal_r = gx_px, gy_px, goal_r
                    else:
                        alpha = 0.4
                        locked_gx_px = (1-alpha)*locked_gx_px + alpha*gx_px
                        locked_gy_px = (1-alpha)*locked_gy_px + alpha*gy_px
                        locked_goal_r = (1-alpha)*locked_goal_r + alpha*goal_r
                    if lock_count >= GOAL_LOCK_FRAMES:
                        goal_locked = True
            # choose goal to use
            if goal_locked:
                gx_px, gy_px, goal_r = int(locked_gx_px), int(locked_gy_px), int(max(locked_goal_r, 1))
            elif red is not None:
                gx_px, gy_px, goal_r = red
            else:
                gx_px = gy_px = goal_r = None

            if gx_px is not None:
                gx, gy = norm_xy(gx_px, gy_px, w, h)
            else:
                gx = gy = np.nan

            # --- MediaPipe Hands ---
            rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            hand_nx, hand_ny, hand_depth_m = np.nan, np.nan, np.nan
            grasp = 0
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0]
                # landmarks are normalized [0,1]
                # Use wrist (0) and MCPs to estimate palm center; or simply use wrist for stability
                wrist = lm.landmark[mp_hands.HandLandmark.WRIST]
                hand_nx, hand_ny = wrist.x, wrist.y
                px, py = denorm_xy(hand_nx, hand_ny, w, h)
                hand_depth_m = get_depth_at_pixel(depth_frame, px, py)

                # Pinch detection (thumb tip and index tip)
                th = lm.landmark[mp_hands.HandLandmark.THUMB_TIP]
                ix = lm.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thx, thy = denorm_xy(th.x, th.y, w, h)
                ixx, iyy = denorm_xy(ix.x, ix.y, w, h)
                pinch_px = np.hypot(thx - ixx, thy - iyy)
                pinch_thresh = PINCH_THRESH_PX_AT_640 * (w / 640.0)
                grasp = int(pinch_px < pinch_thresh)

                # Draw landmarks
                mp.solutions.drawing_utils.draw_landmarks(
                    color, lm, mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=mp.solutions.drawing_styles.get_default_hand_connections_style(),
                )
                # draw pinch midpoint if grasping
                try:
                    th = lm.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    ix = lm.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    pinch_cx = float((th.x + ix.x) * 0.5)
                    pinch_cy = float((th.y + ix.y) * 0.5)
                    ppx, ppy = denorm_xy(pinch_cx, pinch_cy, w, h)
                    cv2.circle(color, (ppx, ppy), 5, (0, 255, 255), -1)
                    cv2.putText(color, "pinch", (ppx+6, ppy-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                except Exception:
                    pass
                    connection_drawing_spec=mp.solutions.drawing_styles.get_default_hand_connections_style()

            # --- Cube detection (ArUco optional) ---
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            cube = detect_aruco_center(gray, CUBE_TAG_ID)
            if cube is not None:
                cx_px, cy_px = cube
                cx, cy = norm_xy(cx_px, cy_px, w, h)
                last_cx, last_cy = cx, cy
                grace_left = RELEASE_GRACE_FRAMES
            else:
                # prefer pinch point when grasping
                if grasp and 'pinch_cx' in locals() and not np.isnan(pinch_cx) and not np.isnan(pinch_cy):
                    cx, cy = pinch_cx, pinch_cy
                    last_cx, last_cy = cx, cy
                    grace_left = RELEASE_GRACE_FRAMES
                elif grasp and not np.isnan(hand_nx):
                    cx, cy = hand_nx, hand_ny
                    last_cx, last_cy = cx, cy
                    grace_left = RELEASE_GRACE_FRAMES
                else:
                    if grace_left > 0 and not np.isnan(last_cx) and not np.isnan(last_cy):
                        cx, cy = last_cx, last_cy
                        grace_left -= 1
                    else:
                        cx, cy = np.nan, np.nan

            # --- Compute velocity (normalized image-plane) --- (normalized image-plane) --- (normalized image-plane) ---
            if prev_nx is None or np.isnan(hand_nx) or np.isnan(hand_ny):
                vx, vy = 0.0, 0.0
            else:
                vx = float(hand_nx - prev_nx)
                vy = float(hand_ny - prev_ny)
                # clip for safety
                vx = float(np.clip(vx * 5.0, -1.0, 1.0))  # scale factor to fit [-1,1]
                vy = float(np.clip(vy * 5.0, -1.0, 1.0))
            prev_nx, prev_ny = hand_nx, hand_ny

            # --- Reward proxy + success hold ---
            if not np.isnan(cx) and not np.isnan(gx) and not np.isnan(gy):
                dist = float(np.hypot(cx - gx, cy - gy))
                reward = -dist
                goal_r_norm = max(goal_r / float(w), 0.02)
                in_goal = dist < goal_r_norm
                if in_goal:
                    success_hold = min(success_hold + 1, SUCCESS_HOLD_FRAMES)
                else:
                    success_hold = 0
                success = (success_hold >= SUCCESS_HOLD_FRAMES)
            else:
                reward = 0.0
                success = False
                success_hold = 0

            # --- Draw overlays ---
            if success:
                cv2.putText(color, "SUCCESS", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            if red is not None:
                cv2.circle(color, (gx_px, gy_px), goal_r, (0, 0, 255), 2)
                cv2.circle(color, (gx_px, gy_px), 4, (0, 0, 255), -1)
            if cube is not None:
                cv2.circle(color, (cx_px, cy_px), 6, (0, 255, 0), -1)
                cv2.putText(color, "cube(tag)", (cx_px+8, cy_px-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            elif grasp and not np.isnan(hand_nx):
                px, py = denorm_xy(hand_nx, hand_ny, w, h)
                cv2.circle(color, (px, py), 6, (0, 255, 255), -1)
                cv2.putText(color, "cube≈hand", (px+8, py-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

            # HUD text
            cv2.putText(color, f"REC: {'ON' if recording else 'OFF'}  frames:{len(obs_buf)}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0) if recording else (0,0,255), 2)
            cv2.putText(color, f"grasp:{grasp}  reward:{reward:.3f}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

            cv2.imshow("Demo Recorder", color)

            # --- Buffering ---
            if recording:
                obs = np.array([
                    hand_nx, hand_ny, hand_depth_m,
                    vx, vy,
                    float(grasp),
                    gx, gy,
                    cx, cy,
                ], dtype=np.float32)
                act = np.array([vx, vy], dtype=np.float32)
                obs_buf.append(obs)
                act_buf.append(act)
                rew_buf.append(float(reward))
                # Mark done only on success (we also allow manual stop)
                done_buf.append(bool(success))

            # --- Keyboard control ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                recording = not recording
                if recording:
                    print("[REC] started new episode buffer")
                    obs_buf.clear(); act_buf.clear(); rew_buf.clear(); done_buf.clear()
                    # reset goal lock
                    goal_locked = False
                    lock_count = 0
                    locked_gx_px = np.nan
                    locked_gy_px = np.nan
                    locked_goal_r = 0
                else:
                    print("[REC] paused (buffer kept, press 's' to save)")
            elif key == ord('s'):
                if len(obs_buf) == 0:
                    print("[SAVE] nothing to save; record first with 'r'")
                else:
                    # Build arrays
                    obs_arr = np.stack(obs_buf, axis=0)
                    act_arr = np.stack(act_buf, axis=0)
                    rew_arr = np.array(rew_buf, dtype=np.float32)
                    done_arr = np.array(done_buf, dtype=np.bool_)
                    goal_arr = np.array([gx, gy], dtype=np.float32)

                    # Episode metadata
                    meta: Dict[str, object] = {
                        "obs_layout": OBS_LAYOUT,
                        "action_layout": ACTION_LAYOUT,
                        "camera": {
                            "width": w, "height": h, "fps": args.camera_fps
                        },
                        "red_hsv": {
                            "lower1": RED_LOWER_1.tolist(),
                            "upper1": RED_UPPER_1.tolist(),
                            "lower2": RED_LOWER_2.tolist(),
                            "upper2": RED_UPPER_2.tolist(),
                        },
                        "pinch_thresh_px_at_640": PINCH_THRESH_PX_AT_640,
                        "timestamp": time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime()),
                        "aruco": {
                            "dict": "DICT_4X4_50", "cube_tag_id": CUBE_TAG_ID
                        },
                    }

                    # Success flag for episode
                    ep_success = bool(np.any(done_arr))

                    # Filename
                    fname = outdir / f"episode_{episode_idx:05d}_{'success' if ep_success else 'fail'}.npz"
                    np.savez_compressed(
                        str(fname),
                        obs=obs_arr,
                        acts=act_arr,
                        rews=rew_arr,
                        dones=done_arr,
                        goal=goal_arr,
                        meta=meta,
                    )
                    print(f"[SAVE] wrote {fname} | steps={len(obs_arr)} | success={ep_success}")
                    episode_idx += 1
                    # Reset buffers
                    recording = False
                    obs_buf.clear(); act_buf.clear(); rew_buf.clear(); done_buf.clear()

    finally:
        pipe.stop()
        hands.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
