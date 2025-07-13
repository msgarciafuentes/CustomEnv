import mujoco
import mujoco.viewer
import time

# Load model
model = mujoco.MjModel.from_xml_path('assets/custom_env2.xml')
data = mujoco.MjData(model)

# Get actuator ID
actuator_id = model.actuator('index_mcp_act').id

# Set up viewer
viewer = mujoco.viewer.launch_passive(model, data)

# Initialize control value
control_value = 0.0
max_value = 2.0
increment = 0.05

print("Starting simulation...")

# Main simulation loop
while viewer.is_running():
    # Update control
    if control_value < max_value:
        control_value += increment
        control_value = min(control_value, max_value)

    data.ctrl[actuator_id] = control_value

    # Step simulation
    mujoco.mj_step(model, data)
    viewer.sync()

    print(f"Control value: {control_value:.4f}")  # You will now see prints
    time.sleep(0.02)

viewer.close()



# Correct way to get actuator name:
actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)

print("Actuator ID:", actuator_id)
print("Actuator name:", actuator_name)
print("Actuator ctrlrange:", model.actuator_ctrlrange[actuator_id])
print("Number of actuators:", model.nu)
print("CTRL initial value:", data.ctrl[actuator_id])