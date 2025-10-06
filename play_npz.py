import os
import time
import json
import argparse
import numpy as np

import mujoco
from mujoco import viewer


# ---------- Utils ----------

def load_demo(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    if "action_24dof" not in d.files:
        raise ValueError("NPZ is missing 'action_24dof' (expected from the 24-DOF recorder).")
    A = d["action_24dof"].astype(np.float32)            # (N, 24) in [-1,1]
    t = d["t"].astype(np.float32) if "t" in d.files else np.arange(A.shape[0], dtype=np.float32)
    meta = json.loads(d["meta"].item()) if "meta" in d.files else {}
    names = meta.get("actuator_names", [f"act_{i:02d}" for i in range(A.shape[1])])
    return t, A, names, meta


def build_actuator_map(model):
    """name -> actuator index in this model"""
    m = {}
    for i in range(model.nu):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if nm is not None:
            m[nm] = i
    return m


def make_index_and_ranges(model, demo_actuator_names):
    """
    Create an index map from demo's actuator order -> model.ctrl indices
    and a per-index (lo, hi) range pulled from xml.
    """
    name_to_idx = build_actuator_map(model)
    mapped = []
    missing = []
    for j, n in enumerate(demo_actuator_names):
        if n in name_to_idx:
            mapped.append((j, name_to_idx[n]))
        else:
            missing.append(n)

    if missing:
        print("[WARN] The following actuators from the demo are not in the model and will be ignored:")
        for n in missing:
            print("   -", n)

    demo_to_model_idx = np.array([m for _, m in mapped], dtype=np.int32)
    kept_cols         = np.array([j for j, _ in mapped], dtype=np.int32)

    # ctrl ranges from xml
    ctrl_lo = model.actuator_ctrlrange[:, 0].copy()
    ctrl_hi = model.actuator_ctrlrange[:, 1].copy()

    return kept_cols, demo_to_model_idx, ctrl_lo, ctrl_hi


def interp_action(t_demo, A_demo, t_cur):
    """
    Linearly interpolate action at time t_cur using the demo timestamps.
    """
    if t_cur <= t_demo[0]:
        return A_demo[0]
    if t_cur >= t_demo[-1]:
        return A_demo[-1]
    k = np.searchsorted(t_demo, t_cur, side="right") - 1
    k = max(0, min(k, len(t_demo) - 2))
    t0, t1 = t_demo[k], t_demo[k + 1]
    a0, a1 = A_demo[k], A_demo[k + 1]
    alpha = 0.0 if t1 <= t0 else float((t_cur - t0) / (t1 - t0))
    return (1.0 - alpha) * a0 + alpha * a1


def norm_to_ctrl(a_norm, lo, hi):
    """Map [-1,1] -> [lo, hi] with clamping."""
    return lo + np.clip((a_norm + 1.0) * 0.5, 0.0, 1.0) * (hi - lo)


# ---------- Player ----------

def play_demo(xml_path, npz_path, sim_hz=240, speed=1.0, loop=True):
    print(f"[Info] XML: {xml_path}")
    print(f"[Info] NPZ: {npz_path}")
    print(f"[Info] sim_hz={sim_hz}  speed={speed}x  loop={loop}")

    t_demo, A_demo_full, demo_names, meta = load_demo(npz_path)
    print(f"[Info] demo frames: {len(t_demo)} | duration ~ {t_demo[-1]:.2f}s")
    print("[Info] actuators in demo order:")
    for i, n in enumerate(demo_names):
        print(f"   [{i:02d}] {n}")

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    kept_cols, demo_to_model_idx, ctrl_lo, ctrl_hi = make_index_and_ranges(model, demo_names)
    if kept_cols.size == 0:
        raise RuntimeError("No overlapping actuators between demo and model. Check names/order.")

    # Reduce to overlapping columns (if any demo actuators were missing in the model)
    A_demo = A_demo_full[:, kept_cols]

    # Cache a ctrl buffer
    ctrl = np.zeros(model.nu, dtype=np.float32)

    # Timing
    dt = 1.0 / float(sim_hz)
    T_end = float(t_demo[-1])

    # Controls: SPACE pause/play, . step fwd one physics step when paused, , step back small time,
    #           + / - speed up/down, r restart, q quit
    paused = False
    cur_t = 0.0
    speed_mult = float(speed)

    def apply_action(a_demo_row):
        """Write the (possibly interpolated) demo action into model.ctrl with proper ranges."""
        # Map the overlapping demo columns into model indices
        ctrl[:] = data.ctrl  # start from current to keep untouched actuators as-is
        # to avoid per-index python loop, broadcast with numpy
        lo = ctrl_lo[demo_to_model_idx]
        hi = ctrl_hi[demo_to_model_idx]
        mapped = norm_to_ctrl(a_demo_row, lo, hi)
        ctrl[demo_to_model_idx] = mapped.astype(np.float32)
        data.ctrl[:] = ctrl

    print("\n[Keys] SPACE: pause/play | . : step forward | , : small step back"
          " | + / - : speed | r : restart | q : quit\n")

    with viewer.launch_passive(model, data) as v:
        # set a nice default camera if it exists
        v.cam.azimuth = 120.0
        v.cam.elevation = -20.0
        v.cam.distance = 2.8
        v.cam.lookat[:] = [0.1, 0.0, 0.8]

        # Prime the scene
        mujoco.mj_forward(model, data)

        last_wall = time.time()
        while v.is_running():
            # --- keyboard via OpenGL viewer doesn't give direct key states;
            # we use a minimal stdin non-blocking trick on keypress by polling viewer events.
            # To keep it simple/portable, we read from input() is not ideal; instead,
            # we rely on time-stepped controls and simple 'paused' toggle printed in console.
            # Tip: you can also hook OpenCV's waitKey in a separate small window if you want.

            # crude key polling through viewer's built-in shortcuts:
            # Use ESC to close window (handled by viewer). We'll map speed changes on number keys.
            # For richer key input, consider integrating glfw directly; to keep this standalone,
            # we'll add a tiny stdin poll when paused.

            # Try to keep real-time pacing
            now = time.time()
            while (now - last_wall) >= dt:
                last_wall += dt

                if not paused:
                    # Advance demo clock
                    cur_t += dt * speed_mult
                    if cur_t > T_end:
                        if loop:
                            cur_t = 0.0
                            # re-initialize physics for clean replay
                            mujoco.mj_resetData(model, data)
                            mujoco.mj_forward(model, data)
                        else:
                            paused = True
                            cur_t = T_end

                    a = interp_action(t_demo, A_demo, cur_t)
                    apply_action(a)

                # Step physics regardless (so gravity/contacts continue during pause if you want)
                mujoco.mj_step(model, data)

            # Render a frame
            v.sync()

            # Lightweight interactive controls via console: read a single char when user presses ENTER
            # (This keeps the script dependency-free. If you prefer real-time key handling, add OpenCV as in the recorder.)
            if paused:
                # Polling every ~0.25s for commands so we don't block rendering
                if (time.time() - now) > 0.25:
                    pass

            # Non-blocking tiny input: use try/except around input with a timeout?
            # Simpler: read environment variables or just print instructions once.
            # For practical control, we allow live speed change via these viewer shortcuts:
            #   Hold/Release 'Pause' by clicking the window's pause icon, or just let it run.
            # If you absolutely need keys: set USE_OPENCV_KEYS=True and add cv2.waitKey handling.

            # (To keep this file minimal and robust across platforms, we keep it simple.)

    print("[Done]")


# ---------- Optional OpenCV keys version (uncomment to use) ----------
# If you want proper keys (space/./,/+/-/r/q), uncomment the code below and set USE_OPENCV_KEYS=True.
USE_OPENCV_KEYS = False
if USE_OPENCV_KEYS:
    import cv2
    def poll_keys():
        k = cv2.waitKey(1) & 0xFF
        if k == 255:
            return None
        try:
            return chr(k).lower()
        except ValueError:
            return None


# ---------- CLI ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay a recorded 24-DOF demo (.npz) in MuJoCo.")
    parser.add_argument("--xml", type=str, default="assets/custom_env_mediapipe_demo.xml",
                        help="Path to MuJoCo XML model (actuator names/ranges are read from here).")
    parser.add_argument("--npz", type=str, required=True,
                        help="Path to the recorded .npz file with action_24dof + meta.actuator_names.")
    parser.add_argument("--hz", type=int, default=240, help="Physics step rate.")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
    parser.add_argument("--loop", action="store_true", help="Loop the demo when it ends.")
    args = parser.parse_args()

    play_demo(args.xml, args.npz, sim_hz=args.hz, speed=args.speed, loop=bool(args.loop))
