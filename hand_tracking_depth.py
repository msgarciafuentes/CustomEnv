# hand_tracking_depth.py
import cv2
import mediapipe as mp
import time
import pickle
import numpy as np

# -------- Settings --------
FPS_TARGET = 30
DURATION_SECONDS = 10
RES_W, RES_H = 640, 480   # RealSense color stream resolution
COUNTDOWN_FROM = 3
OUTPUT_PKL = "hand_record.pkl"
NEIGHBOR_K = 1  # radius for neighborhood depth median (1 => 3x3)

# --- NEW: video output filename ---
OUTPUT_VIDEO = "hand_record.mp4"
# --------------------------

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
    model_complexity=1,
)

# RealSense setup
try:
    import pyrealsense2 as rs
except ImportError:
    raise SystemExit("pyrealsense2 is not installed. Run: pip install pyrealsense2")

pipeline = rs.pipeline()
config = rs.config()
# Enable BOTH color and depth, same resolution/FPS, and align later
config.enable_stream(rs.stream.color, RES_W, RES_H, rs.format.bgr8, FPS_TARGET)
config.enable_stream(rs.stream.depth, RES_W, RES_H, rs.format.z16, FPS_TARGET)

# Start streaming and prepare alignment to color
profile = pipeline.start(config)
align = rs.align(rs.stream.color)

# --- NEW: Prepare video writer (we record the MIRRORED preview to match landmarks) ---
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, float(FPS_TARGET), (RES_W, RES_H))
if not video_writer.isOpened():
    pipeline.stop(); hands.close()
    raise SystemExit("Failed to open VideoWriter. Check codecs/permissions.")

def get_depth_meters(depth_frame, u, v, k=NEIGHBOR_K):
    """
    Return median depth (meters) around (u,v) over a (2k+1)x(2k+1) window,
    ignoring zeros/invalid. If none valid, return None.
    """
    vals = []
    w, h = RES_W, RES_H
    for dv in range(-k, k+1):
        vv = v + dv
        if vv < 0 or vv >= h:
            continue
        for du in range(-k, k+1):
            uu = u + du
            if uu < 0 or uu >= w:
                continue
            d = depth_frame.get_distance(uu, vv)  # meters
            if d and d > 0:
                vals.append(d)
    if not vals:
        return None
    return float(np.median(vals))

# Countdown
for countdown in reversed(range(1, COUNTDOWN_FROM + 1)):
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    color_frame = aligned.get_color_frame()
    if not color_frame:
        continue
    frame = np.asanyarray(color_frame.get_data())

    # Mirror preview (selfie)
    frame_flipped = cv2.flip(frame, 1)
    cv2.putText(
        frame_flipped, f"Starting in {countdown}",
        (60, 200), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 5, cv2.LINE_AA
    )

    # --- NEW: write countdown frames too (optional but nice) ---
    video_writer.write(frame_flipped)

    cv2.imshow("Hand Recorder", frame_flipped)
    if cv2.waitKey(1000) & 0xFF == 27:
        pipeline.stop(); hands.close(); cv2.destroyAllWindows(); video_writer.release()
        raise SystemExit("Cancelled.")

# Storage: each frame is either None, or a list of 21 tuples (x, y, z)
#   x,y in [0,1] (MediaPipe normalized on the *flipped* image)
#   z = RealSense depth in meters at the corresponding pixel (or None if invalid)
landmark_data = []
max_frames = int(FPS_TARGET * DURATION_SECONDS)
print(f"📹 Recording hand motion for {DURATION_SECONDS} seconds...")

start_time = time.time()

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())

        # Mirror preview for display *and* for mediapipe input
        frame_flipped = cv2.flip(frame, 1)

        # MediaPipe expects RGB
        rgb = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands.process(rgb)
        rgb.flags.writeable = True

        frame_landmarks = None
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            # Build (x, y, z) per landmark, where z is RealSense depth (meters)
            frame_landmarks = []
            for lm in hand_landmarks.landmark:
                nx, ny = float(lm.x), float(lm.y)

                # Convert normalized coords (on the *flipped* frame) to pixel u,v
                u_flipped = int(round(nx * (RES_W - 1)))
                v_flipped = int(round(ny * (RES_H - 1)))

                # Depth frame is aligned to color but *not flipped*, so mirror back:
                u_src = (RES_W - 1) - u_flipped  # horizontal mirror back
                v_src = v_flipped

                # Get depth (meters) with neighborhood median to reduce holes
                z_m = None
                if 0 <= u_src < RES_W and 0 <= v_src < RES_H:
                    z_m = get_depth_meters(depth_frame, u_src, v_src, k=NEIGHBOR_K)

                frame_landmarks.append((nx, ny, z_m))

            # Draw landmarks on the *display/recorded* frame (flipped)
            mp_drawing.draw_landmarks(
                frame_flipped,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(thickness=2),
            )

        landmark_data.append(frame_landmarks)

        # HUD
        elapsed = time.time() - start_time
        cv2.putText(
            frame_flipped,
            f"Recording... {len(landmark_data)}/{max_frames}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA,
        )

        # --- NEW: Write the mirrored preview to video ---
        video_writer.write(frame_flipped)

        cv2.imshow("Hand Recorder (RealSense depth z)", frame_flipped)

        # Stop by ESC or when we reach target frames/time
        if (cv2.waitKey(1) & 0xFF) == 27:
            break
        if len(landmark_data) >= max_frames or elapsed >= DURATION_SECONDS:
            break

finally:
    pipeline.stop()
    hands.close()
    cv2.destroyAllWindows()
    # --- NEW: release video writer ---
    video_writer.release()

# Save landmarks
with open(OUTPUT_PKL, "wb") as f:
    pickle.dump(landmark_data, f)

print(f"✅ Done. Saved landmarks to {OUTPUT_PKL}")
print(f"✅ Saved video to {OUTPUT_VIDEO} ({RES_W}x{RES_H} @ {FPS_TARGET} FPS)")
print("   Each landmark is (x, y, z) with z in meters from RealSense.")