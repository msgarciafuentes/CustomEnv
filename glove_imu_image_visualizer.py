import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd
import time
import cv2
import glob

# Load CyberGlove data
dir_name = 'data/'
#dir_name = 'data/20250507/' 
filename = 'fist'
cyberglove_data = pd.read_csv(f'{dir_name}{filename}.csv')

# Expect columns: timestamp, reliability, roll, pitch, yaw
imu_dir_name = 'imu_data/'
imu_filename = 'imu_measurements_pitch'   # adjust if needed
imu_df = pd.read_csv(f'{imu_dir_name}{imu_filename}.csv')

# Choose relevant sensors
sensor_columns = [f"sensor_{i}" for i in range(18)]  # Adjust if needed

# Subtract first row (calibration baseline)
baseline = cyberglove_data.iloc[0][sensor_columns]
calibrated_data = cyberglove_data[sensor_columns] - baseline

#print(calibrated_data)

# Clamp negative values to zero
calibrated_data = calibrated_data.clip(lower=0)

# Save calibrated data
calibrated_data.to_csv(f'output/calibrated_data_{filename}.csv', index=False)

# Normalize to range [0, 2.0944] radians (120°)
normalized_data = (calibrated_data / 255.0) * 2.0944 * 1.4 # or rescale differently if needed

normalized_data.to_csv(f'output/normalized_data_{filename}.csv', index=False)

# Use first row as baseline (so initial pose ≈ 0), convert deg -> rad
imu_angles = imu_df[['pitch', 'yaw', 'roll']].astype(float)
imu_baseline = imu_angles.iloc[0]
imu_centered_deg = imu_angles - imu_baseline
imu_centered_rad = np.deg2rad(imu_centered_deg)

# MuJoCo wrist joints have range [-1, 1] rad in XML, so clamp there
imu_centered_rad = imu_centered_rad.clip(lower=-1.0, upper=1.0)

# Resample IMU to match CyberGlove row count (simple linear interpolation)
L = len(normalized_data)
src_idx = np.arange(len(imu_centered_rad), dtype=float)
dst_idx = np.linspace(0, len(imu_centered_rad) - 1, L)

imu_resampled = pd.DataFrame({
    'wrist_flex_act':    np.interp(dst_idx, src_idx, imu_centered_rad['pitch'].to_numpy()),  # pitch -> flex
    'wrist_abduction_act': np.interp(dst_idx, src_idx, imu_centered_rad['yaw'].to_numpy()),  # yaw -> abduction
    'wrist_roll_act':    np.interp(dst_idx, src_idx, imu_centered_rad['roll'].to_numpy()),   # roll -> roll
})

imu_resampled.to_csv(f'output/imu_resampled_{imu_filename}.csv', index=False)

# Find your PNG sequences (adjust patterns/paths if needed)
color_files = sorted(glob.glob('images/color_image_*.png'))
depth_files = sorted(glob.glob('images/depth_image_*.png'))

if not color_files:
    print("[WARN] No color PNGs matched pattern: color_image_*.png")
if not depth_files:
    print("[WARN] No depth PNGs matched pattern: depth_image_*.png")

# --- NEW: make panels movable/resizable + set initial positions ---
cv2.namedWindow("Color Image", cv2.WINDOW_NORMAL)
cv2.namedWindow("Depth Image", cv2.WINDOW_NORMAL)

# Optional: set initial sizes (pixels)
cv2.resizeWindow("Color Image", 640, 480)
cv2.resizeWindow("Depth Image", 640, 480)

# Ensure they aren’t forced to be topmost (sometimes window managers pin them)
try:
    cv2.setWindowProperty("Color Image", cv2.WND_PROP_TOPMOST, 0)
    cv2.setWindowProperty("Depth Image", cv2.WND_PROP_TOPMOST, 0)
except Exception:
    pass  # not all backends support this

# Map CyberGlove sensor columns to MuJoCo actuators
sensor_to_actuator = {
    'sensor_0': 'thumb_roll_act', 
    'sensor_1': 'thumb_mcp_act', 
    'sensor_2': 'thumb_ip_act', 
    'sensor_3': 'index_abd_act',
    'sensor_4': 'index_mcp_act', 
    'sensor_5': 'index_pip_act', 
    'sensor_6': 'middle_mcp_act', 
    'sensor_7': 'middle_pip_act', 
    'sensor_8': 'ring_abd_act',
    'sensor_9': 'ring_mcp_act', 
    'sensor_10': 'ring_pip_act', 
    'sensor_11': 'pinky_abd_act',
    'sensor_12': 'pinky_mcp_act', 
    'sensor_13': 'pinky_pip_act', 
    'sensor_15': 'thumb_base_act',
    #'sensor_16': 'wrist_flex_act',
    #'sensor_17': 'wrist_abduction_act',
}

# Load your custom MuJoCo model
model = mujoco.MjModel.from_xml_path('assets/custom_env2.xml')
data = mujoco.MjData(model)

# Find actuator IDs
actuator_ids = {name: model.actuator(name).id for name in sensor_to_actuator.values()}

actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]

for dip_act in ['index_dip_act', 'middle_dip_act', 'ring_dip_act', 'pinky_dip_act']:
    if dip_act in actuator_names:
        actuator_ids[dip_act] = model.actuator(dip_act).id

for wrist_act in ['wrist_flex_act', 'wrist_abduction_act', 'wrist_roll_act']:
    if wrist_act in actuator_names:
        actuator_ids[wrist_act] = model.actuator(wrist_act).id

for actuator, id in actuator_ids.items():
    print(f"Actuator {id}: {actuator}")


# Set speed control
frame_duration = 0.001  # About 50 FPS
steps = 1  # Interpolation steps

# Start viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.azimuth = 120
    viewer.cam.elevation = -20
    viewer.cam.distance = 1
    viewer.cam.lookat[:] = [0.1, 0.0, 0.8]

    time.sleep(frame_duration)

    while viewer.is_running():
        for frame_index in range(len(normalized_data) - 1):

            if color_files:
                color_path = color_files[frame_index % len(color_files)]
                color_img = cv2.imread(color_path, cv2.IMREAD_COLOR)
                if color_img is not None:
                    cv2.imshow("Color Image", color_img)

            if depth_files:
                depth_path = depth_files[frame_index % len(depth_files)]
                # Try full-depth read; normalize to 8-bit for display if needed
                depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                if depth_raw is not None:
                    if depth_raw.dtype != np.uint8:
                        depth_vis = cv2.normalize(depth_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    else:
                        depth_vis = depth_raw
                    cv2.imshow("Depth Image", depth_vis)

            # Let OpenCV process window events without blocking
            cv2.waitKey(1)

            current_row = normalized_data.iloc[frame_index]
            next_row = normalized_data.iloc[frame_index + 1]

            imu_row = imu_resampled.iloc[frame_index]

            for step in range(steps):
                blend = step / steps

                for sensor_name, actuator_name in sensor_to_actuator.items():
                    sensor_value_current = current_row[sensor_name]
                    sensor_value_next = next_row[sensor_name]

                    interpolated_value = (1 - blend) * sensor_value_current + blend * sensor_value_next

                    actuator_id = actuator_ids[actuator_name]
                    #print(f"actuator id: {actuator_id} and interpolated data: {interpolated_value}")
                    data.ctrl[actuator_id] = interpolated_value
                
                data.ctrl[actuator_ids['wrist_flex_act']]       = imu_row['wrist_flex_act']
                data.ctrl[actuator_ids['wrist_abduction_act']]  = imu_row['wrist_abduction_act']
                data.ctrl[actuator_ids['wrist_roll_act']]       = imu_row['wrist_roll_act']

                dip_scale = 0.66  # or 2/3

                # Example: middle DIP is 2/3 of middle PIP
                data.ctrl[actuator_ids['index_dip_act']] = data.ctrl[actuator_ids['index_pip_act']] * dip_scale
                data.ctrl[actuator_ids['middle_dip_act']] = data.ctrl[actuator_ids['middle_pip_act']] * dip_scale
                data.ctrl[actuator_ids['ring_dip_act']] = data.ctrl[actuator_ids['ring_pip_act']] * dip_scale
                data.ctrl[actuator_ids['pinky_dip_act']] = data.ctrl[actuator_ids['pinky_pip_act']] * dip_scale


                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(frame_duration)
    cv2.destroyAllWindows()
