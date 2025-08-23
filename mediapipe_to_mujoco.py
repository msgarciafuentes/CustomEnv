import mujoco
import mujoco.viewer
import numpy as np
import pickle
import time
import collections
import csv

# --------------------------
# CONFIGURATION
# --------------------------
XML_PATH = 'assets/custom_env2.xml'
PICKLE_FILE = 'hand_record.pkl'
FPS = 30
MAX_ANGLE = 2.0944  # 120 degrees in radians
DIP_RATIO = 0.66
TRANSLATION_SCALE = 0.3
MIN_ANGLE = 0.1  # ~6 degrees
FLEX_SCALE = 2.0
ABD_SCALE = 2.0
log_data = []

# --------------------------
# HELPERS
# --------------------------
def angle_between(p1, p2, p3):
    a = np.array(p1) - np.array(p2)
    b = np.array(p3) - np.array(p2)
    cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))

def compute_wrist_flex(frame):
    v = np.array(frame[9]) - np.array(frame[0])  # middle_mcp - wrist
    return np.arcsin(v[1] / (np.linalg.norm(v) + 1e-6))

def compute_wrist_abduction(frame):
    v = np.array(frame[5]) - np.array(frame[17])  # index_mcp - pinky_mcp
    v[1] = 0  # project to x-z plane
    return np.arcsin(v[0] / (np.linalg.norm(v) + 1e-6))

def compute_abduction_angle(base, mcp, ref):
    v1 = np.array(mcp) - np.array(base)
    v2 = np.array(ref) - np.array(base)
    # Use x-z plane (ignore vertical y component)
    v1[1] = 0
    v2[1] = 0
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.arccos(np.clip(cos, -1.0, 1.0))

def compute_thumb_roll(frame):
    wrist = np.array(frame[0])
    index_mcp = np.array(frame[5])
    pinky_mcp = np.array(frame[17])

    # Palm normal
    palm_normal = np.cross(index_mcp - wrist, pinky_mcp - wrist)
    palm_normal /= (np.linalg.norm(palm_normal) + 1e-6)

    # Thumb direction
    thumb_dir = np.array(frame[3]) - np.array(frame[2])
    thumb_dir /= (np.linalg.norm(thumb_dir) + 1e-6)

    # Project thumb_dir into palm plane
    proj_thumb = thumb_dir - np.dot(thumb_dir, palm_normal) * palm_normal
    proj_thumb /= (np.linalg.norm(proj_thumb) + 1e-6)

    v_index = index_mcp - wrist
    v_index /= (np.linalg.norm(v_index) + 1e-6)

    # Signed roll angle
    roll_angle = np.arctan2(
        np.dot(np.cross(v_index, proj_thumb), palm_normal),
        np.dot(v_index, proj_thumb)
    )
    return abs(roll_angle)

def map_to_ctrl(value, center, scale):
    """Map MediaPipe [0-1] to MuJoCo [-1, 1] with scaling and clamping."""
    ctrl_val = (value - center) * 2 * scale
    print(f"value is {value} and ctrl_val is {ctrl_val}")
    return float(np.clip(ctrl_val, -1.0, 1.0))

# --------------------------
# LOAD LANDMARK DATA
# --------------------------
with open(PICKLE_FILE, "rb") as f:
    hand_frames = pickle.load(f)

print(f"✅ Loaded {len(hand_frames)} frames from {PICKLE_FILE}")

# --------------------------
# MUJOCO INIT
# --------------------------
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# Actuator names
actuator_names = [
    'gripper_x_act', 'gripper_y_act', 'gripper_z_act',
    'wrist_flex_act', 'wrist_abduction_act',
    'thumb_base_act', 'thumb_roll_act', 'thumb_mcp_act', 'thumb_ip_act',
    'index_mcp_act', 'index_pip_act', 'index_dip_act', 'index_abd_act',
    'middle_mcp_act', 'middle_pip_act', 'middle_dip_act',
    'ring_mcp_act', 'ring_pip_act', 'ring_dip_act', 'ring_abd_act',
    'pinky_mcp_act', 'pinky_pip_act', 'pinky_dip_act', 'pinky_abd_act'
]
actuator_ids = {name: model.actuator(name).id for name in actuator_names if model.actuator(name) is not None}

# Landmark triplets for angle computation
landmark_triplets = {
    'thumb_base_act': (0, 1, 2),
    'thumb_mcp_act': (1, 2, 3),
    'thumb_ip_act': (2, 3, 4),
    'index_mcp_act': (0, 5, 6),
    'index_pip_act': (5, 6, 7),
    'index_dip_act': (6, 7, 8),
    'middle_mcp_act': (0, 9, 10),
    'middle_pip_act': (9, 10, 11),
    'middle_dip_act': (10, 11, 12),
    'ring_mcp_act': (0, 13, 14),
    'ring_pip_act': (13, 14, 15),
    'ring_dip_act': (14, 15, 16),
    'pinky_mcp_act': (0, 17, 18),
    'pinky_pip_act': (17, 18, 19),
    'pinky_dip_act': (18, 19, 20),
}

# Abduction triplets (base, finger MCP, neighbor MCP)
abduction_map = {
    'index_abd_act': (0, 5, 9),   # wrist, index_mcp, middle_mcp
    'ring_abd_act':  (0, 13, 9),  # wrist, ring_mcp, middle_mcp
    'pinky_abd_act': (0, 17, 13), # wrist, pinky_mcp, ring_mcp
}

# --------------------------
# MAIN REPLAY
# --------------------------
base_wrist = None
baseline_angles = {}
angle_history = collections.defaultdict(lambda: collections.deque(maxlen=3))

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.azimuth = 120
    viewer.cam.elevation = -20
    viewer.cam.distance = 2
    viewer.cam.lookat[:] = [0.1, 0.0, 0.8]

    print("🎬 Replaying hand motion...")

    for frame_index, frame in enumerate(hand_frames):
        if not viewer.is_running():
            break  # wrap-around loop

        if frame is None:
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(1 / FPS)
            continue

        # Track wrist for translation
        if base_wrist is None:
            base_wrist = frame[0]

        print(f"Base wrist is: {base_wrist}\n")
        wrist_pos = frame[0]
        print(f"Frame {frame_index}: Wrist position (x={wrist_pos[0]:.4f}, y={wrist_pos[1]:.4f}, z={wrist_pos[2]:.4f})")

        wrist = frame[0]

        if 'gripper_x_act' in actuator_ids:
            # Map wrist movement into [-1, 1]
            x_ctrl = map_to_ctrl(wrist[0], base_wrist[0], TRANSLATION_SCALE)
            z_ctrl = map_to_ctrl(wrist[1], base_wrist[1], TRANSLATION_SCALE)

            data.ctrl[actuator_ids['gripper_x_act']] = x_ctrl
            data.ctrl[actuator_ids['gripper_z_act']] = -z_ctrl

            # Log for CSV
            frame_data = (frame_index, x_ctrl, -z_ctrl)
            log_data.append(frame_data)

            # Debug print
            print(f"Frame {frame_index}: gripper_x_act={x_ctrl:.2f}, gripper_z_act={-z_ctrl:.2f}")
        
        # Thumb abduction
        if 'thumb_base_act' in actuator_ids:
            thumb_abd = compute_abduction_angle(frame[0], frame[1], frame[5])
            thumb_abd = np.clip(thumb_abd, 0, MAX_ANGLE)
            if thumb_abd < MIN_ANGLE:
                thumb_abd = 0
            data.ctrl[actuator_ids['thumb_base_act']] = thumb_abd

        # Thumb roll
        if 'thumb_roll_act' in actuator_ids:
            roll_angle = compute_thumb_roll(frame)
            roll_angle = np.clip(roll_angle, 0, MAX_ANGLE)
            data.ctrl[actuator_ids['thumb_roll_act']] = roll_angle
        
        # Add wrist flexion/abduction
        """
        if 'wrist_flex_act' in actuator_ids:
            flex = np.clip(compute_wrist_flex(frame) * FLEX_SCALE, -1, 1)
            data.ctrl[actuator_ids['wrist_flex_act']] = flex
        
        if 'wrist_abduction_act' in actuator_ids:
            abd = np.clip(compute_wrist_abduction(frame) * ABD_SCALE, -1, 1)
            data.ctrl[actuator_ids['wrist_abduction_act']] = abd
        
        if 'wrist_roll_act' in actuator_ids:
            palm_normal = np.cross(index_mcp - wrist, pinky_mcp - wrist)
            roll = np.arctan2(palm_normal[0], palm_normal[2])  # angle in x-z plane
            data.ctrl[actuator_ids['wrist_roll_act']] = np.clip(roll, -1, 1)

        """
        # Compute and apply finger angles
        for actuator_name, (i1, i2, i3) in landmark_triplets.items():
            if actuator_name not in actuator_ids:
                continue

            raw_angle = angle_between(frame[i1], frame[i2], frame[i3])
            flexion_angle = np.pi - raw_angle

            if frame_index == 0:
                baseline_angles[actuator_name] = flexion_angle

            angle = flexion_angle - baseline_angles.get(actuator_name, 0)
            angle = np.clip(angle, 0, MAX_ANGLE)

            if angle < MIN_ANGLE:
                angle = 0

            angle_history[actuator_name].append(angle)
            smoothed = np.median(angle_history[actuator_name])
            data.ctrl[actuator_ids[actuator_name]] = smoothed
        """
        # Abduction control
        for act_name, (base_idx, mcp_idx, ref_idx) in abduction_map.items():
            if act_name not in actuator_ids:
                continue
            abd_angle = compute_abduction_angle(frame[base_idx], frame[mcp_idx], frame[ref_idx])
            abd_angle = np.clip(abd_angle, 0, MAX_ANGLE)
            if abd_angle < MIN_ANGLE:
                abd_angle = 0
            data.ctrl[actuator_ids[act_name]] = abd_angle
        """
        for name in ['index_pip_act', 'middle_mcp_act']:
            val = data.ctrl[actuator_ids[name]]
            print(f"{name}: {val:.2f} rad ({np.degrees(val):.1f}°)")
        
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(1 / FPS)

print("✅ Replay complete.")

with open('hand_position_log.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Frame', 'gripper_x_act', 'gripper_z_act'])
    writer.writerows(log_data)

print("📁 Saved motion log to hand_position_log.csv")
