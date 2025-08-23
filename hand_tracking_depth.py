import cv2
import mediapipe as mp
import pyrealsense2 as rs
import numpy as np
import pickle
import time

# === Setup MediaPipe ===
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
)

# === Setup RealSense pipeline ===
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

print("Starting RealSense pipeline...")
profile = pipeline.start(config)
depth_intrinsics = profile.get_stream(rs.stream.depth)\
    .as_video_stream_profile().get_intrinsics()

fps = 30
duration_seconds = 5
max_frames = int(fps * duration_seconds)

# Countdown
for countdown in reversed(range(1, 4)):
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue
    color_image = np.asanyarray(color_frame.get_data())
    cv2.putText(color_image, f"Starting in {countdown}", (100, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
    cv2.imshow("Hand Recorder", color_image)
    cv2.waitKey(1000)

landmark_data = []

print(f"📹 Recording hand motion for {duration_seconds} seconds...")

while len(landmark_data) < max_frames:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    if not color_frame or not depth_frame:
        continue

    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    frame_landmarks = None
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        frame_landmarks = []

        # Convert 2D MediaPipe landmarks to real-world 3D using depth
        for lm in hand_landmarks.landmark:
            u = int(lm.x * color_image.shape[1])
            v = int(lm.y * color_image.shape[0])

            # Get depth in meters
            depth_val = depth_image[v, u] * 0.001  # depth is in mm

            # Deproject pixel to 3D point (in camera coordinates, meters)
            X, Y, Z = rs.rs2_deproject_pixel_to_point(
                depth_intrinsics, [u, v], depth_image[v, u]
            )
            frame_landmarks.append((X, Y, Z))

        # Draw landmarks
        mp_drawing.draw_landmarks(
            color_image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )

    landmark_data.append(frame_landmarks)

    cv2.putText(color_image, f"Recording... {len(landmark_data)}/{max_frames}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Hand Recorder 3D", color_image)

    if cv2.waitKey(1) & 0xFF == 27:
        break

hands.close()
pipeline.stop()
cv2.destroyAllWindows()

# Save 3D landmarks
with open("hand_record_3d.pkl", "wb") as f:
    pickle.dump(landmark_data, f)

print("✅ Done. Saved to hand_record_3d.pkl")
