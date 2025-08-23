import cv2
import mediapipe as mp
import pickle
import numpy as np
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList  # ✅ Correct import
import pandas as pd

# Parameters
PKL_FILE = "hand_record.pkl"
FPS = 30
SHOW_INDEX = True

# Load the recorded landmark list
with open(PKL_FILE, "rb") as f:
    frames = pickle.load(f)

print(f"Loaded {len(frames)} frames from {PKL_FILE}")

# Build DataFrame for Excel export
rows = []
for frame_idx, landmarks in enumerate(frames):
    row = [frame_idx]  # Start with frame number
    if landmarks:
        # Flatten (x, y, z) for all 21 landmarks
        for lm in landmarks:
            row.extend([lm[0], lm[1], lm[2]])
    else:
        # Pad with NaNs if no data
        row.extend([np.nan] * 63)
    rows.append(row)

# Column names: Frame, L0_x, L0_y, L0_z, ..., L20_x, L20_y, L20_z
columns = ['Frame']
for i in range(21):
    columns += [f"L{i}_x", f"L{i}_y", f"L{i}_z"]

# Create and save DataFrame
df = pd.DataFrame(rows, columns=columns)
df.to_csv("landmark_data.csv", index=False)

print("📁 Saved Excel file: landmark_data.csv")

# MediaPipe drawing
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Create a blank white canvas
canvas_height, canvas_width = 480, 640
canvas = lambda: 255 * np.ones((canvas_height, canvas_width, 3), dtype=np.uint8)

# Replay loop
for idx, landmarks in enumerate(frames):
    frame = canvas()

    if landmarks is not None:
        # Build landmark list from saved coordinates
        hand_landmarks = NormalizedLandmarkList()
        for x, y, z in landmarks:
            hand_landmarks.landmark.add(x=x, y=y, z=z)

        # Draw skeleton
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )

        # Draw landmark indices
        if SHOW_INDEX:
            for i, (x, y, _) in enumerate(landmarks):
                cx, cy = int(x * canvas_width), int(y * canvas_height)
                cv2.putText(frame, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(frame, f"Frame {idx+1}/{len(frames)}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Hand Replay", frame)
    if cv2.waitKey(int(1000 / FPS)) & 0xFF == 27:
        break

cv2.destroyAllWindows()
