# replay_pkl.py — side-by-side replay of landmarks (left) and recorded video (right)

import cv2
import mediapipe as mp
import pickle
import numpy as np
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList

# ---- Config ----
PKL_FILE = "recordings/hand_getball.pkl"   # landmarks recorded by hand_tracking_depth.py
VIDEO_FILE = "recordings/hand_getball.mp4" # mirrored preview saved during recording
CANVAS_H = 480                 # height of the hand panel (will match video height if possible)
CANVAS_W = 640                 # width of the hand panel
SHOW_INDEX = False             # show landmark indices
SHOW_Z_M = True                # overlay z (meters) if available
WINDOW_TITLE = "Hand Replay  |  Recorded Video"

# ---- Load landmark frames ----
with open(PKL_FILE, "rb") as f:
    frames = pickle.load(f)
num_frames = len(frames)
print(f"Loaded {num_frames} frames from {PKL_FILE}")

# Determine tuple width (x,y,z or x,y,z1,z2)
def detect_tuple_width(frames_list):
    for item in frames_list:
        if item:
            for lm in item:
                if lm is not None:
                    return len(lm)
    return 3  # default
TUPLE_W = detect_tuple_width(frames)

# ---- Video setup ----
cap = cv2.VideoCapture(VIDEO_FILE)
if not cap.isOpened():
    print(f"⚠️ Could not open video '{VIDEO_FILE}'. Proceeding with landmarks only.")
video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or CANVAS_W)
video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or CANVAS_H)

# If we have a video size, match the hand canvas height to it for clean side-by-side
if video_h > 0:
    CANVAS_H = video_h
# Keep canvas width proportional-ish (you can tweak if you prefer)
CANVAS_W = max(320, min(800, int(CANVAS_H * (640/480))))

# ---- MediaPipe drawing setup ----
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def blank_canvas():
    return 255 * np.ones((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

def draw_landmarks_panel(landmarks_tuple_list, frame_idx, total_frames):
    panel = blank_canvas()

    if landmarks_tuple_list:
        # Build a NormalizedLandmarkList for drawing
        nl = NormalizedLandmarkList()
        for lm in landmarks_tuple_list:
            if lm is None:
                nl.landmark.add(x=0.0, y=0.0, z=0.0)
                continue
            if TUPLE_W == 4:
                x, y, z1, z2 = lm
                z_draw = float(z1 if z1 is not None else 0.0)
                z_meters = z2
            else:  # TUPLE_W == 3
                x, y, z = lm
                z_draw = float(z if z is not None else 0.0)
                z_meters = z

            nl.landmark.add(x=float(x), y=float(y), z=z_draw)

        # Draw skeleton
        mp_drawing.draw_landmarks(
            panel,
            nl,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(thickness=2)
        )

        # Optional indices / depth text
        if SHOW_INDEX or SHOW_Z_M:
            for i, lm in enumerate(landmarks_tuple_list):
                if lm is None:
                    continue
                if TUPLE_W == 4:
                    x, y, z1, z2 = lm
                    z_m = z2
                else:
                    x, y, z = lm
                    z_m = z
                cx, cy = int(x * CANVAS_W), int(y * CANVAS_H)
                if SHOW_INDEX:
                    cv2.putText(panel, str(i), (cx, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
                if SHOW_Z_M and (z_m is not None) and not (isinstance(z_m, float) and np.isnan(z_m)):
                    cv2.putText(panel, f"{float(z_m):.2f} m", (cx, cy - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (40, 40, 40), 1, cv2.LINE_AA)

    # HUD
    cv2.putText(panel, f"Frame {frame_idx+1}/{total_frames}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)
    return panel

def pad_to_height(img, h):
    if img.shape[0] == h:
        return img
    scale = h / img.shape[0]
    new_w = int(img.shape[1] * scale)
    return cv2.resize(img, (new_w, h))

def pad_right_to(img, width):
    if img.shape[1] >= width:
        return img
    pad = np.full((img.shape[0], width - img.shape[1], 3), 255, dtype=np.uint8)
    return np.hstack([img, pad])

# ---- Replay loop ----
frame_idx = 0
while True:
    # 1) Build the hand panel from PKL
    lm_list = frames[frame_idx] if frame_idx < num_frames else None
    hand_panel = draw_landmarks_panel(lm_list, frame_idx, num_frames)

    # 2) Read next video frame (or placeholder)
    if cap.isOpened():
        ret, video_frame = cap.read()
        if not ret:
            # video ended — show a placeholder
            video_frame = 255 * np.ones((CANVAS_H, video_w if video_w > 0 else CANVAS_W, 3), dtype=np.uint8)
            cv2.putText(video_frame, "<video ended>", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        video_frame = 255 * np.ones((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        cv2.putText(video_frame, "<no video>", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 3) Normalize heights and widths for clean hstack
    hand_panel = pad_to_height(hand_panel, CANVAS_H)
    video_frame = pad_to_height(video_frame, CANVAS_H)

    max_w = max(hand_panel.shape[1], video_frame.shape[1])
    hand_panel_p = pad_right_to(hand_panel, max_w)
    video_panel_p = pad_right_to(video_frame, max_w)

    combined = np.hstack([hand_panel_p, video_panel_p])

    # 4) Show
    cv2.imshow(WINDOW_TITLE, combined)
    key = cv2.waitKey(int(1000 / video_fps)) & 0xFF
    if key == 27:  # ESC
        break

    # 5) Advance; stop when both sources are done
    frame_idx += 1
    video_done = not cap.isOpened() or \
                 (cap.get(cv2.CAP_PROP_POS_FRAMES) >= cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0)
    if frame_idx >= num_frames and video_done:
        break

# ---- Cleanup ----
cap.release()
cv2.destroyAllWindows()
print("Replay finished.")
