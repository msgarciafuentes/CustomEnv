import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd
import time

# Load CyberGlove data
filename = 'fist' 
cyberglove_data = pd.read_csv(f'data/{filename}.csv')

# Choose relevant sensors
sensor_columns = [f"sensor_{i}" for i in range(18)]  # Adjust if needed

# Subtract first row (calibration baseline)
baseline = cyberglove_data.iloc[0][sensor_columns]
calibrated_data = cyberglove_data[sensor_columns] - baseline

print(calibrated_data)

# Clamp negative values to zero
calibrated_data = calibrated_data.clip(lower=0)

# 🔥 Save calibrated data
calibrated_data.to_csv(f'output/calibrated_data_{filename}.csv', index=False)

# Normalize to range [0, 2.0944] radians (120°)
normalized_data = (calibrated_data / 255.0) * 2.0944  # or rescale differently if needed

# Map CyberGlove sensor columns to MuJoCo actuators
sensor_to_actuator = {
    'sensor_0': 'thumb_mcp_act',
    'sensor_1': 'thumb_ip_act',
    'sensor_2': 'thumb_base_act',
    'sensor_3': 'index_mcp_act',
    'sensor_4': 'index_pip_act',
    'sensor_5': 'middle_mcp_act',
    'sensor_6': 'middle_pip_act',
    'sensor_7': 'ring_mcp_act',
    'sensor_8': 'ring_pip_act',
    'sensor_9': 'pinky_mcp_act',
    'sensor_10': 'pinky_pip_act',
    'sensor_16': 'wrist_flex_act',
    'sensor_17': 'wrist_abduction_act',
}

# Load your custom MuJoCo model
model = mujoco.MjModel.from_xml_path('assets/custom_env2.xml')
data = mujoco.MjData(model)

# Find actuator IDs
actuator_ids = {name: model.actuator(name).id for name in sensor_to_actuator.values()}

print("Actuator IDs:", actuator_ids)

# Set speed control
frame_duration = 0.001  # About 50 FPS
steps = 5  # Interpolation steps



# Start viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.azimuth = 120
    viewer.cam.elevation = -30
    viewer.cam.distance = 1.5
    viewer.cam.lookat[:] = [0.1, 0.0, 0.8]

    time.sleep(0.001)

    while viewer.is_running():
        for frame_index in range(len(normalized_data) - 1):
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

                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(frame_duration)
