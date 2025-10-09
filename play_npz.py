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
    and per-index (lo, hi) from xml.
    """
    name_to_idx = build_actuator_map(model)
    mapped, missing = [], []
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
    """Linearly interpolate action at time t_cur using the demo timestamps."""
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

def play_demo(xml_path, npz_path, sim_hz=240, speed=1.0, loop=True,
              zero_center=False, zero_groups="gripper"):
    """
    zero_center: enable subtracting the first frame on selected groups.
    zero_groups: comma list among {'gripper','wrist','fingers','all'}.
                 Default 'gripper' (x/y/z only) so fingers don't curl.
    """
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
    kept_names = [demo_names[j] for j in kept_cols]

    # --------- Identify gripper axes in the kept set ---------
    def idxs_for(prefix):
        return [i for i, n in enumerate(kept_names) if n.startswith(prefix)]

    gx_idx = idxs_for("gripper_x")
    gy_idx = idxs_for("gripper_y")
    gz_idx = idxs_for("gripper_z")

    # --------- Apply flips (BEFORE zero-centering) ----------
    #do_flip_x = flip_gripper or flip_x
    #do_flip_y = flip_gripper or flip_y
    #do_flip_z = flip_gripper or flip_z

    if gx_idx:
        A_demo[:, gx_idx] *= -1.0
        print("[Flip] gripper_x")
    if gy_idx:
        A_demo[:, gy_idx] *= -1.0
        print("[Flip] gripper_y")
    # if gz_idx:
    #     A_demo[:, gz_idx] *= -1.0
    #     print("[Flip] gripper_z")

    # Build masks for zero-centering groups
    groups = {g.strip().lower() for g in zero_groups.split(",")} if zero_center else set()
    if "all" in groups:
        mask = np.ones((A_demo.shape[1],), dtype=bool)
    else:
        def has_prefix(name, pref): return name.startswith(pref)
        is_gripper = np.array([has_prefix(n, "gripper_") for n in kept_names])
        is_wrist   = np.array([has_prefix(n, "wrist_")   for n in kept_names])
        # fingers: any of these prefixes
        finger_prefixes = ("thumb_", "index_", "middle_", "ring_", "pinky_")
        is_finger = np.array([n.startswith(finger_prefixes) for n in kept_names])

        mask = np.zeros((A_demo.shape[1],), dtype=bool)
        if "gripper" in groups: mask |= is_gripper
        if "wrist"   in groups: mask |= is_wrist
        if "fingers" in groups: mask |= is_finger

    # Optional zero-centering (by masked groups only)
    if zero_center and mask.any():
        a0 = A_demo[0].copy()
        A_demo[:, mask] = np.clip(A_demo[:, mask] - a0[mask], -1.0, 1.0)
        print(f"[Calib] Zero-centered groups: {', '.join(sorted(groups)) or '(none)'}")
    elif zero_center:
        print("[Calib] Zero-centering requested but no matching groups found for current actuators.")


    if gx_idx:
        A_demo[:, gx_idx] = np.clip(A_demo[:, gx_idx] * 0.6, -1.0, 1.0)
    if gy_idx:
        A_demo[:, gy_idx] = np.clip(A_demo[:, gy_idx] * 0.7, -1.0, 1.0)
    if gz_idx:
        A_demo[:, gz_idx] = np.clip(A_demo[:, gz_idx] * 0.7, -1.0, 1.0)

    # Timing
    dt = 1.0 / float(sim_hz)
    T_end = float(t_demo[-1])

    # Controls
    paused = False
    cur_t = 0.0
    speed_mult = float(speed)

    ctrl = np.zeros(model.nu, dtype=np.float32)

    def apply_action(a_demo_row):
        """Write the (possibly interpolated) demo action into model.ctrl with proper ranges."""
        ctrl[:] = data.ctrl  # keep other actuators untouched
        lo = ctrl_lo[demo_to_model_idx]
        hi = ctrl_hi[demo_to_model_idx]
        mapped = norm_to_ctrl(a_demo_row, lo, hi)
        ctrl[demo_to_model_idx] = mapped.astype(np.float32)
        data.ctrl[:] = ctrl

    print("\n[Keys] (window controls) ESC: close | use viewer toolbar to pause. "
          "This script replays continuously unless --loop is omitted.\n")

    with viewer.launch_passive(model, data) as v:
        # camera
        v.cam.azimuth = 120.0
        v.cam.elevation = -20.0
        v.cam.distance = 2.8
        v.cam.lookat[:] = [0.1, 0.0, 0.8]

        mujoco.mj_forward(model, data)

        last_wall = time.time()
        while v.is_running():
            now = time.time()
            while (now - last_wall) >= dt:
                last_wall += dt

                # Advance demo time
                cur_t += dt * speed_mult
                if cur_t > T_end:
                    if loop:
                        cur_t = 0.0
                        mujoco.mj_resetData(model, data)
                        mujoco.mj_forward(model, data)
                    else:
                        cur_t = T_end

                # Interpolate and apply
                a = interp_action(t_demo, A_demo, cur_t)
                apply_action(a)

                mujoco.mj_step(model, data)

            v.sync()

    print("[Done]")


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
    parser.add_argument("--zero-center", action="store_true", help="Enable zero-centering (default groups: gripper).")
    parser.add_argument("--zero-groups", type=str, default="gripper",
                        help="Comma list from {'gripper','wrist','fingers','all'}. Default 'gripper'.")

    args = parser.parse_args()

    play_demo(args.xml, args.npz,
              sim_hz=args.hz, speed=args.speed, loop=bool(args.loop),
              zero_center=args.zero_center, zero_groups=args.zero_groups)
