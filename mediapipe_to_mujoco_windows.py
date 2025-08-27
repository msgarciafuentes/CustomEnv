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

# Angles & mapping
MAX_ANGLE = 2.0944   # 120 degrees in radians
MIN_ANGLE = 0.1      # ~6 degrees, deadzone
DIP_RATIO = 0.66

# Wrist translation from normalized x,y -> ctrl
TRANSLATION_SCALE = 0.3

# Use RealSense depth (meters) for forward/back (Y) translation
USE_DEPTH_FOR_Y = True
DEPTH_TO_CTRL_GAIN = 2.0  # meters delta * gain -> [-1,1] clipped

# Build a geometry z from metric depth: z_rel = (z_wrist - z_i) * scale
# (Closer => more negative, similar to MediaPipe)
USE_DEPTH_FOR_GEOMETRY = False
Z_REL_SCALE = 5.0

FLEX_SCALE = 2.0
ABD_SCALE = 2.0

log_data = []

# --------------------------
# HELPERS
# --------------------------
def safe_norm(v):
    n = np.linalg.norm(v)
    return n if n > 1e-9 else 1e-9

def angle_between(p1, p2, p3):
    a = np.array(p1) - np.array(p2)
    b = np.array(p3) - np.array(p2)
    na = safe_norm(a)
    nb = safe_norm(b)
    cos_theta = np.dot(a, b) / (na * nb)
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))

def compute_wrist_flex(frame3d):
    v = np.array(frame3d[9]) - np.array(frame3d[0])  # middle_mcp - wrist
    return np.arcsin(v[1] / safe_norm(v))

def compute_wrist_abduction(frame3d):
    v = np.array(frame3d[5]) - np.array(frame3d[17])  # index_mcp - pinky_mcp
    v[1] = 0  # project to x-z plane
    return np.arcsin(v[0] / safe_norm(v))

def compute_abduction_angle(base, mcp, ref):
    v1 = np.array(mcp) - np.array(base)
    v2 = np.array(ref) - np.array(base)
    v1[1] = 0; v2[1] = 0
    n1 = safe_norm(v1); n2 = safe_norm(v2)
    cos = np.dot(v1, v2) / (n1 * n2)
    return np.arccos(np.clip(cos, -1.0, 1.0))

def compute_thumb_roll(frame3d):
    wrist = np.array(frame3d[0])
    index_mcp = np.array(frame3d[5])
    pinky_mcp = np.array(frame3d[17])

    palm_normal = np.cross(index_mcp - wrist, pinky_mcp - wrist)
    palm_normal /= safe_norm(palm_normal)

    thumb_dir = np.array(frame3d[3]) - np.array(frame3d[2])
    thumb_dir /= safe_norm(thumb_dir)

    proj_thumb = thumb_dir - np.dot(thumb_dir, palm_normal) * palm_normal
    proj_thumb /= safe_norm(proj_thumb)

    v_index = index_mcp - wrist
    v_index /= safe_norm(v_index)

    roll_angle = np.arctan2(
        np.dot(np.cross(v_index, proj_thumb), palm_normal),
        np.dot(v_index, proj_thumb)
    )
    return abs(roll_angle)

def map_to_ctrl(value, center, scale):
    """Map MediaPipe normalized [0..1] around 'center' to [-1, 1] with scaling."""
    ctrl_val = (value - center) * 2.0 * scale
    print(f"value is {value} and ctrl_val is {ctrl_val}")
    return float(np.clip(ctrl_val, -1.0, 1.0))

def build_geometry_frame(raw_frame, base_wrist_z_m=None, use_depth_for_geom=True, z_rel_scale=5.0):
    """
    raw_frame: list of 21 tuples (x, y, z_meters or None)
    Returns:
      frame3d: list of 21 [x_norm, y_norm, z_geom] for angle computation
      z_m_list: list of 21 z in meters (or None)
    z_geom is either:
      - relative depth from meters (wrist-referenced, scaled), or
      - 0.0 if depth unavailable or not used (keeps old behavior stable).
    """
    frame3d = []
    z_m_list = []
    z_wrist = raw_frame[0][2] if (raw_frame and raw_frame[0] is not None) else None

    # baseline depth to reference relative z
    z_ref = base_wrist_z_m if base_wrist_z_m is not None else z_wrist

    for lm in raw_frame:
        if lm is None:
            frame3d.append([0.0, 0.0, 0.0])
            z_m_list.append(None)
            continue
        x, y, z_m = lm
        if use_depth_for_geom and (z_m is not None) and (z_ref is not None):
            z_geom = (z_ref - z_m) * z_rel_scale   # closer => negative
        else:
            z_geom = 0.0
        frame3d.append([float(x), float(y), float(z_geom)])
        z_m_list.append(z_m if z_m is None else float(z_m))
    return frame3d, z_m_list

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
base_wrist_norm = None          # [x_norm, y_norm, z_geom] baseline for x/z translation
base_wrist_z_m = None           # meters baseline for depth translation (Y)
baseline_angles = {}
angle_history = collections.defaultdict(lambda: collections.deque(maxlen=3))

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.azimuth = 120
    viewer.cam.elevation = -20
    viewer.cam.distance = 2
    viewer.cam.lookat[:] = [0.1, 0.0, 0.8]

    print("🎬 Replaying hand motion...")

    for frame_index, raw_frame in enumerate(hand_frames):
        if not viewer.is_running():
            break

        if raw_frame is None:
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(1 / FPS)
            continue

        # Build geometry frame (z_geom from depth if available) and keep metric z_m list
        frame3d, z_m_list = build_geometry_frame(
            raw_frame,
            base_wrist_z_m=base_wrist_z_m,
            use_depth_for_geom=USE_DEPTH_FOR_GEOMETRY,
            z_rel_scale=Z_REL_SCALE
        )

        # Establish baselines
        if base_wrist_norm is None and frame3d[0] is not None:
            base_wrist_norm = frame3d[0]  # normalized x,y and relative z
        if base_wrist_z_m is None and (z_m_list[0] is not None) and (z_m_list[0] > 0):
            base_wrist_z_m = z_m_list[0]

        wrist_norm = frame3d[0]            # [x_norm, y_norm, z_geom]
        wrist_z_m  = z_m_list[0]           # meters or None

        wz_str = f"{wrist_z_m:.4f}" if (wrist_z_m is not None) else "nan"
        print(f"Frame {frame_index}: Wrist x={wrist_norm[0]:.4f}, y={wrist_norm[1]:.4f}, z_geom={wrist_norm[2]:.4f}, z_m={wz_str}")

        # --------------------------
        # TRANSLATION CONTROLS
        # --------------------------
        # X/Z from normalized x,y
        if 'gripper_x_act' in actuator_ids:
            x_ctrl = map_to_ctrl(wrist_norm[0], base_wrist_norm[0], TRANSLATION_SCALE)
            data.ctrl[actuator_ids['gripper_x_act']] = x_ctrl
        else:
            x_ctrl = np.nan

        if 'gripper_z_act' in actuator_ids:
            z_ctrl = map_to_ctrl(wrist_norm[1], base_wrist_norm[1], TRANSLATION_SCALE)
            data.ctrl[actuator_ids['gripper_z_act']] = -z_ctrl  # invert to taste
        else:
            z_ctrl = np.nan

        # Y from metric depth delta (toward camera = closer)
        if USE_DEPTH_FOR_Y and ('gripper_y_act' in actuator_ids) and (wrist_z_m is not None) and (base_wrist_z_m is not None):
            depth_delta = base_wrist_z_m - wrist_z_m  # closer -> positive
            y_ctrl = float(np.clip(depth_delta * DEPTH_TO_CTRL_GAIN, -1.0, 1.0))
            data.ctrl[actuator_ids['gripper_y_act']] = y_ctrl
        else:
            y_ctrl = np.nan

        # Log XYZ ctrls
        log_data.append((frame_index, x_ctrl, y_ctrl, (-z_ctrl if not np.isnan(z_ctrl) else np.nan)))

        # --------------------------
        # THUMB abduction / roll (examples)
        # --------------------------
        if 'thumb_base_act' in actuator_ids:
            thumb_abd = compute_abduction_angle(frame3d[0], frame3d[1], frame3d[5])
            thumb_abd = np.clip(thumb_abd, 0, MAX_ANGLE)
            if thumb_abd < MIN_ANGLE: thumb_abd = 0
            data.ctrl[actuator_ids['thumb_base_act']] = float(thumb_abd)

        if 'thumb_roll_act' in actuator_ids:
            roll_angle = compute_thumb_roll(frame3d)
            roll_angle = np.clip(roll_angle, 0, MAX_ANGLE)
            data.ctrl[actuator_ids['thumb_roll_act']] = float(roll_angle)

        # --------------------------
        # FINGER FLEXION ANGLES
        # --------------------------
        for actuator_name, (i1, i2, i3) in landmark_triplets.items():
            if actuator_name not in actuator_ids:
                continue

            raw_angle = angle_between(frame3d[i1], frame3d[i2], frame3d[i3])
            flexion_angle = np.pi - raw_angle

            if frame_index == 0:
                baseline_angles[actuator_name] = flexion_angle

            angle = flexion_angle - baseline_angles.get(actuator_name, 0.0)
            angle = np.clip(angle, 0.0, MAX_ANGLE)
            if angle < MIN_ANGLE:
                angle = 0.0

            angle_history[actuator_name].append(angle)
            smoothed = float(np.median(angle_history[actuator_name]))
            data.ctrl[actuator_ids[actuator_name]] = smoothed

        # Example debug prints
        for name in ['index_pip_act', 'middle_mcp_act']:
            if name in actuator_ids:
                val = data.ctrl[actuator_ids[name]]
                print(f"{name}: {val:.2f} rad ({np.degrees(val):.1f}°)")

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(1 / FPS)

print("✅ Replay complete.")

with open('hand_position_log.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Frame', 'gripper_x_act', 'gripper_y_act', 'gripper_z_act'])
    writer.writerows(log_data)

print("📁 Saved motion log to hand_position_log.csv")
