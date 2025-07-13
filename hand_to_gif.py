import imageio.v2 as imageio
from tqdm import tqdm  # Optional: for progress display
import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd
import time
from mujoco import Renderer

# Load CyberGlove data
dir_name = 'data/'
#dir_name = 'data/20250507/' 
filename = 'yooooo'
cyberglove_data = pd.read_csv(f'{dir_name}{filename}.csv')

# Choose relevant sensors
sensor_columns = [f"sensor_{i}" for i in range(18)]  # Adjust if needed

# Subtract first row (calibration baseline)
baseline = cyberglove_data.iloc[0][sensor_columns]
calibrated_data = cyberglove_data[sensor_columns] - baseline

print(calibrated_data)

# Clamp negative values to zero
calibrated_data = calibrated_data.clip(lower=0)

# Save calibrated data
calibrated_data.to_csv(f'output/calibrated_data_{filename}.csv', index=False)

# Normalize to range [0, 2.0944] radians (120°)
normalized_data = (calibrated_data / 255.0) * 2.0944 * 1.4 # or rescale differently if needed

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
    'sensor_16': 'wrist_flex_act',
    'sensor_17': 'wrist_abduction_act',
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

for actuator, id in actuator_ids.items():
    print(f"Actuator {id}: {actuator}")


# Set speed control
frame_duration = 0.001  # About 50 FPS
steps = 1  # Interpolation steps
# Initialize offscreen renderer
renderer = mujoco.Renderer(model)

# Set camera parameters
opt = mujoco.MjvOption()
scn = mujoco.MjvScene(model, maxgeom=10000)
cam = mujoco.MjvCamera()
cam.azimuth = 120
cam.elevation = -30
cam.distance = 1.5
cam.lookat[:] = [0.1, 0.0, 0.8]
ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

opt = mujoco.MjvOption()
scn = mujoco.MjvScene(model, maxgeom=10000)
con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

frames = []

for frame_index in tqdm(range(len(normalized_data) - 1)):
    current_row = normalized_data.iloc[frame_index]
    next_row = normalized_data.iloc[frame_index + 1]

    for step in range(steps):
        blend = step / steps

        for sensor_name, actuator_name in sensor_to_actuator.items():
            sensor_value_current = current_row[sensor_name]
            sensor_value_next = next_row[sensor_name]
            interpolated_value = (1 - blend) * sensor_value_current + blend * sensor_value_next
            actuator_id = actuator_ids[actuator_name]
            data.ctrl[actuator_id] = interpolated_value

        dip_scale = 0.66
        data.ctrl[actuator_ids['index_dip_act']] = data.ctrl[actuator_ids['index_pip_act']] * dip_scale
        data.ctrl[actuator_ids['middle_dip_act']] = data.ctrl[actuator_ids['middle_pip_act']] * dip_scale
        data.ctrl[actuator_ids['ring_dip_act']] = data.ctrl[actuator_ids['ring_pip_act']] * dip_scale
        data.ctrl[actuator_ids['pinky_dip_act']] = data.ctrl[actuator_ids['pinky_pip_act']] * dip_scale

        mujoco.mj_step(model, data)

        # Update scene before rendering
        mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
        mujoco.mjr_render(renderer.viewport, scn, ctx)

        # Render and store frame
        frame = renderer.render()
        frames.append(frame[::-1])  # Flip vertically if needed

        time.sleep(frame_duration)


# Save as GIF
imageio.mimsave('output/move_hand.gif', frames, fps=int(1 / frame_duration))
print("GIF saved to output/move_hand.gif")
