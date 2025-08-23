import cv2
import mediapipe as mp
import time
import pickle

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
)

# Camera setup
cap = cv2.VideoCapture(2)
fps = 30
duration_seconds = 5
max_frames = int(fps * duration_seconds)

# Countdown before recording
for countdown in reversed(range(1, 4)):  # 3, 2, 1
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)
    cv2.putText(frame, f"Starting in {countdown}", (100, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
    cv2.imshow("Hand Recorder", frame)
    cv2.waitKey(1000)  # wait for 1 second

# Storage
landmark_data = []

print(f"📹 Recording hand motion for {duration_seconds} seconds...")

while len(landmark_data) < max_frames:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    frame_landmarks = None
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        frame_landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

        # Draw the hand landmarks and connections
        mp_drawing.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )

    landmark_data.append(frame_landmarks)

    # Visual feedback
    cv2.putText(frame, f"Recording... {len(landmark_data)}/{max_frames}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Hand Recorder with Landmarks", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break  # ESC key

hands.close()
cap.release()
cv2.destroyAllWindows()

# Save landmarks
with open("hand_record.pkl", "wb") as f:
    pickle.dump(landmark_data, f)

print("✅ Done. Saved to hand_record.pkl")
