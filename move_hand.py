import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd

# Load CyberGlove data
cyberglove_data = pd.read_csv('data/base.csv')

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

# Create the viewer
viewer = mujoco.viewer.launch_passive(model, data)

# Find actuator IDs
actuator_ids = {name: model.actuator(name).id for name in sensor_to_actuator.values()}

print("Press ESC or close the window to exit.")

try:
    while viewer.is_running():
        for index, row in cyberglove_data.iterrows():
            if not viewer.is_running():
                break

            # Set actuator controls
            for sensor_name, actuator_name in sensor_to_actuator.items():
                sensor_value = row[sensor_name]

                # Normalize: CyberGlove output (0-255) --> MuJoCo actuator ctrlrange (0 to ~2 rad)
                normalized_value = (sensor_value / 255.0) * 2.0944  # Max 120 degrees in radians

                # Apply control
                actuator_id = actuator_ids[actuator_name]
                data.ctrl[actuator_id] = normalized_value

            # Step and update viewer
            mujoco.mj_step(model, data)
            viewer.sync()

finally:
    viewer.close()