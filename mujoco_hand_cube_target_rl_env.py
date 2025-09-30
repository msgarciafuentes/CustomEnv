"""
Custom Gymnasium environment for a MuJoCo hand to grasp a cube and move it to a red target area.
This version is **bound to the uploaded XML** `custom_env_mediapipe_demo.xml`.

- Works with mujoco>=2.3, gymnasium>=0.29, stable-baselines3.
- Uses fingertip **body** names (because finger geoms are unnamed) to compute tip positions
  and contact counts vs the cube geom.
- Uses the **target body** position (red paper) as the goal center.

Run training:
    python mujoco_hand_cube_target_rl_env.py --xml custom_env_mediapipe_demo.xml --algo ppo --timesteps 1_000_000

Evaluate:
    python mujoco_hand_cube_target_rl_env.py --xml custom_env_mediapipe_demo.xml --eval runs/hand_cube/best/ppo_hand_best.zip
"""
from __future__ import annotations
import argparse
import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List


import gymnasium as gym
import numpy as np


import mujoco
from mujoco import MjModel, MjData


from gymnasium.spaces import Box
from gymnasium.utils.ezpickle import EzPickle


try:
    from stable_baselines3 import PPO, SAC, TD3
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except Exception:
    PPO = SAC = TD3 = None

import mujoco_viewer



# =========================
# ---- CONFIG SECTION ----
# =========================
HAND_JOINTS = [
"gripper_x", "gripper_y", "gripper_z",
"wrist_flex_pitch", "wrist_abduction_yaw", "wrist_roll",
"thumb_base", "thumb_mcp", "thumb_roll", "thumb_ip",
"index_mcp", "index_abd", "index_pip", "index_dip",
"middle_mcp", "middle_pip", "middle_dip",
"ring_mcp", "ring_abd", "ring_pip", "ring_dip",
"pinky_mcp", "pinky_abd", "pinky_pip", "pinky_dip",
]


HAND_ACTUATORS = [
"gripper_x_act", "gripper_y_act", "gripper_z_act",
"wrist_flex_pitch_act", "wrist_abduction_yaw_act", "wrist_roll_act",
"thumb_base_act", "thumb_mcp_act", "thumb_roll_act", "thumb_ip_act",
"index_mcp_act", "index_abd_act", "index_pip_act", "index_dip_act",
"middle_mcp_act", "middle_pip_act", "middle_dip_act",
"ring_mcp_act", "ring_abd_act", "ring_pip_act", "ring_dip_act",
"pinky_mcp_act", "pinky_abd_act", "pinky_pip_act", "pinky_dip_act",
]


FINGERTIP_BODIES = ["thumb_ip", "index_dip", "middle_dip", "ring_dip", "pinky_dip"]
CUBE_BODY = "cube"
CUBE_GEOM = "cube_geom"
TABLE_GEOM = "table_top"
TARGET_BODY = "target_paper"


ACTION_CLIP = 1.0
CTRL_LIMIT = 0.5
FRAME_SKIP = 10
MAX_EPISODE_STEPS = 300


W_DIST_CUBE_TARGET = 3
W_LIFT = 1.0
W_GRASP = 0.6
W_ACTION_PEN = 2e-4
W_TIP2CUBE = 0.1
SUCCESS_BONUS = 10.0
GRASP_CONTACT_THRESHOLD = 2
LIFT_HEIGHT = 0.02
SUCCESS_RADIUS = 0.07
W_EE2CUBE = 0.2
W_QVEL = 2e-5 # penalize joint speeds

W_PROGRESS = 2.0 
W_TOWARD_SPEED = 0.1

INIT_HAND_NOISE = 0.02
INIT_CUBE_NOISE = 0.04


@dataclass
class NamedIdx:
    qpos: Dict[str, int]
    qvel: Dict[str, int]
    act: Dict[str, int]
    geom: Dict[str, int]
    body: Dict[str, int]
    site: Dict[str, int]


class HandCubeTargetEnv(gym.Env, EzPickle):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, xml_path: str, render_mode: Optional[str] = None, seed: Optional[int] = None):
        EzPickle.__init__(self, xml_path=xml_path, render_mode=render_mode, seed=seed)
        self.render_mode = render_mode
        self.model: MjModel = mujoco.MjModel.from_xml_path(xml_path)
        self.data: MjData = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)

        self.named = self._build_named_idx()
        self._validate_names()

        n_act = len(HAND_ACTUATORS)
        self.action_space = Box(low=-ACTION_CLIP, high=ACTION_CLIP, shape=(n_act,), dtype=np.float32)

        obs_dim = self._obs().shape[0]
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self._viewer = None
        self._step_count = 0

        self._last_xy_dist = 0.0

    # ---------- Utils ----------
    def _build_named_idx(self) -> NamedIdx:
        qpos, qvel, act, geom, body, site = {}, {}, {}, {}, {}, {}
        for j in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if not name:
                continue
            qpos[name] = self.model.jnt_qposadr[j]
            qvel[name] = self.model.jnt_dofadr[j]
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                act[name] = i
        for i in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name:
                geom[name] = i
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name:
                body[name] = i
        for i in range(self.model.nsite):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, i)
            if name:
                site[name] = i
        return NamedIdx(qpos, qvel, act, geom, body, site)

    def _validate_names(self):
        missing = []
        for n in HAND_JOINTS:
            if n not in self.named.qpos:
                missing.append(f"joint:{n}")
        for n in HAND_ACTUATORS:
            if n not in self.named.act:
                missing.append(f"actuator:{n}")
        for n in FINGERTIP_BODIES:
            if n not in self.named.body:
                missing.append(f"body:{n}")
        for n in [CUBE_BODY]:
            if n not in self.named.body:
                missing.append(f"body:{n}")
        if CUBE_GEOM not in self.named.geom:
            missing.append(f"geom:{CUBE_GEOM}")
        if TABLE_GEOM not in self.named.geom:
            missing.append(f"geom:{TABLE_GEOM}")
        if TARGET_BODY not in self.named.body:
            missing.append(f"body:{TARGET_BODY}")
        if missing:
            raise ValueError("Missing names in XML: " + ", ".join(missing))

    # ---------- Core Gym API ----------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._step_count = 0

        mujoco.mj_resetData(self.model, self.data)

        # Joints init with small noise
        for jn in HAND_JOINTS:
            idx = self.named.qpos[jn]
            self.data.qpos[idx] = self.rng.normal(0.0, INIT_HAND_NOISE)
        for jn in HAND_JOINTS:
            vidx = self.named.qvel[jn]
            self.data.qvel[vidx] = 0.0

        # Randomize cube XY mildly
        cube_bid = self.named.body[CUBE_BODY]
        base = self.model.body_pos[cube_bid].copy()
        base[:2] += self.rng.uniform(-INIT_CUBE_NOISE, INIT_CUBE_NOISE, size=2)
        self.model.body_pos[cube_bid] = base

        mujoco.mj_forward(self.model, self.data)

        # initialize progress baseline: cube–target XY distance
        cube_p   = self._cube_pos()
        target_p = self._body_pos(TARGET_BODY)
        self._last_xy_dist = float(np.linalg.norm(cube_p[:2] - target_p[:2]))

        obs = self._obs()
        info = {"is_success": False}
        return obs, info

    def step(self, action: np.ndarray):
        action = np.clip(action, -CTRL_LIMIT, CTRL_LIMIT)
        for a_name, a in zip(HAND_ACTUATORS, action):
            self.data.ctrl[self.named.act[a_name]] = float(a)

        for _ in range(FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        obs = self._obs()
        reward, is_success, terminated = self._reward_done()
        truncated = self._step_count >= MAX_EPISODE_STEPS
        info = {"is_success": is_success}

        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    # ---------- Observations, Reward, Done ----------
    def _body_pos(self, name: str) -> np.ndarray:
        bid = self.named.body[name]
        return self.data.xpos[bid].copy()

    def _geom_pos(self, name: str) -> np.ndarray:
        gid = self.named.geom[name]
        bid = self.model.geom_bodyid[gid]
        return self.data.xpos[bid].copy()

    def _cube_pos(self) -> np.ndarray:
        bid = self.named.body[CUBE_BODY]
        return self.data.xpos[bid].copy()

    def _cube_vel(self) -> np.ndarray:
        bid = self.named.body[CUBE_BODY]
        return self.data.cvel[bid, :3].copy()

    def _hand_qpos_qvel(self) -> Tuple[np.ndarray, np.ndarray]:
        q = np.array([self.data.qpos[self.named.qpos[j]] for j in HAND_JOINTS], dtype=np.float32)
        v = np.array([self.data.qvel[self.named.qvel[j]] for j in HAND_JOINTS], dtype=np.float32)
        return q, v

    def _tips_pos(self) -> np.ndarray:
        ps = []
        for b in FINGERTIP_BODIES:
            bid = self.named.body[b]
            ps.append(self.data.xpos[bid])
        return np.asarray(ps, dtype=np.float32)

    def _obs(self) -> np.ndarray:
        q, v = self._hand_qpos_qvel()
        cube_p = self._cube_pos()
        cube_v = self._cube_vel()
        target_p = self._body_pos(TARGET_BODY)
        tips = self._tips_pos().reshape(-1)
        rel_cube_target = cube_p[:2] - target_p[:2]
        obs = np.concatenate([q, v, cube_p, cube_v, target_p, rel_cube_target, tips]).astype(np.float32)
        return obs

    def _count_grasp_contacts(self) -> int:
        """Count contacts between any fingertip body geom and the cube geom."""
        cube_gid = self.named.geom[CUBE_GEOM]
        tip_bids = {self.named.body[b] for b in FINGERTIP_BODIES}
        nc = 0
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1, g2 = con.geom1, con.geom2
            if g1 == -1 or g2 == -1:
                continue
            b1 = self.model.geom_bodyid[g1]
            b2 = self.model.geom_bodyid[g2]
            if (g1 == cube_gid and b2 in tip_bids) or (g2 == cube_gid and b1 in tip_bids):
                nc += 1
        return nc

    def _table_height(self) -> float:
        gid = self.named.geom[TABLE_GEOM]
        bid = self.model.geom_bodyid[gid]
        # for box, geom_size[z] is half-height
        return float(self.data.xpos[bid][2] + self.model.geom_size[gid][2])

    def _reward_done(self) -> Tuple[float, bool, bool]:

        cube_p = self._cube_pos()
        target_p = self._body_pos(TARGET_BODY)
        table_h = self._table_height()

        xy_dist = np.linalg.norm(cube_p[:2] - target_p[:2])
        lift = max(0.0, cube_p[2] - (table_h + LIFT_HEIGHT))

        tips = self._tips_pos()
        tip_d = float(np.mean(np.linalg.norm(tips - cube_p[None, :], axis=1)))

        grasp_contacts = self._count_grasp_contacts()
        grasp_flag = 1.0 if grasp_contacts >= GRASP_CONTACT_THRESHOLD else 0.0

        ee_p = np.array([
            self.data.qpos[self.named.qpos["gripper_x"]],
            self.data.qpos[self.named.qpos["gripper_y"]],
            self.data.qpos[self.named.qpos["gripper_z"]],
        ], dtype=np.float32)
        ee2cube = float(np.linalg.norm(ee_p - cube_p))

        r = 0.0
        r += W_DIST_CUBE_TARGET * (-xy_dist)
        print(f"r is now: {r}, adding xydist{-xy_dist} * {W_DIST_CUBE_TARGET}")
        r += W_LIFT * lift
        print(f"r is now: {r}, adding lift{lift} * {W_LIFT}")
        r += W_GRASP * grasp_flag
        print(f"r is now: {r}, adding grasp_flag{grasp_flag} * {W_GRASP}")
        r += W_TIP2CUBE * (-tip_d)
        print(f"r is now: {r}, adding tip_d{-tip_d} * {W_TIP2CUBE}")
        r += W_EE2CUBE * (-ee2cube)
        print(f"r is now: {r}, adding ee2cube{-ee2cube} * {W_EE2CUBE}")

        a = np.array([self.data.ctrl[self.named.act[n]] for n in HAND_ACTUATORS])
        r -= W_ACTION_PEN * float(np.sum(a * a))
        print(f"r is now: {r}, subtracting action penalty {W_ACTION_PEN * float(np.sum(a * a))}")
        qv = np.array([self.data.qvel[self.named.qvel[j]] for j in HAND_JOINTS])
        r -= W_QVEL * float(np.sum(qv * qv))
        print(f"r is now: {r}, subtracting qvel penalty {W_QVEL * float(np.sum(qv * qv))}")

        # --- Progress reward: distance improvement since last step ---
        progress = self._last_xy_dist - xy_dist     # >0 if we got closer
        r += W_PROGRESS * progress                          # weight; tune as W_PROGRESS
        self._last_xy_dist = xy_dist

        # --- Bonus for cube velocity toward the target ---
        v_xy    = self._cube_vel()[:2]
        dir_vec = (target_p[:2] - cube_p[:2])
        norm = np.linalg.norm(dir_vec)
        if norm > 1e-6:
            dir_unit = dir_vec / (norm + 1e-6)
            toward_speed = float(np.dot(v_xy, dir_unit))   # positive if moving toward goal
            r += W_TOWARD_SPEED * toward_speed

        in_zone = xy_dist < SUCCESS_RADIUS
        high_enough = cube_p[2] > table_h + 0.01
        is_success = bool(in_zone and high_enough)
        if is_success:
            r += SUCCESS_BONUS
        terminated = False
        print(f"total r: {r} (success={is_success})")
        return float(r), is_success, terminated

    # ---------- Rendering ----------
    def render(self):
        if self._viewer is None:
            # Create interactive viewer
            self._viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self._viewer.render()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


# =========================
# ---- TRAIN / EVAL  ------
# =========================

def make_env(xml_path: str, render: bool = False):
    def _thunk():
        env = HandCubeTargetEnv(xml_path=xml_path, render_mode="human" if render else None)
        env = Monitor(env)
        return env
    return _thunk


def train(xml_path: str, algo: str, logdir: str, total_iters: int = 100, timesteps: int = 10000):
    assert PPO is not None, "Install stable-baselines3 to train: pip install stable-baselines3[extra]"
    os.makedirs(logdir, exist_ok=True)

    venv = DummyVecEnv([make_env(xml_path, render=False)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    if algo.lower() == "ppo":
        model = PPO("MlpPolicy", venv, verbose=1, device="cuda", tensorboard_log=logdir)
    elif algo.lower() == "sac":
        model = SAC("MlpPolicy", venv, verbose=1, device="cuda", tensorboard_log=logdir)
    else:
        model = TD3("MlpPolicy", venv, verbose=1, device="cuda", tensorboard_log=logdir)

    iters = 0
    while iters < total_iters:
        iters += 1
        print(f"Iteration {iters} - training for {timesteps} steps")
        model.learn(total_timesteps=timesteps, reset_num_timesteps=False)
        save_path = os.path.join(logdir, f"{algo}_hand_iter{iters}")
        model.save(save_path)
        venv.save(os.path.join(logdir, "vecnorm.pkl"))
        print(f"Saved model checkpoint at {save_path}")


def evaluate(xml_path: str, model_path: str, episodes: int = 5, deterministic: bool = True):
     # 1) Build a renderable vec env
    venv = DummyVecEnv([make_env(xml_path, render=True)])

    # 2) Try to load VecNormalize stats (preferred path = run root)
    #    e.g. model_path = runs/hand_cube/ppo_hand_iter50.zip
    run_root = os.path.normpath(os.path.join(os.path.dirname(model_path), ".."))
    vecnorm_path = os.path.join(run_root, "hand_cube/vecnorm.pkl")

    if os.path.isfile(vecnorm_path):
        venv = VecNormalize.load(vecnorm_path, venv)
        venv.training = False         # eval mode
        venv.norm_reward = False
        print(f"[Eval] Loaded VecNormalize stats from: {vecnorm_path}")
    else:
        print(f"[Eval] WARNING: VecNormalize stats not found at {vecnorm_path}. "
              "Evaluating without normalization (results may differ).")

    # 3) Infer algo from filename and load model onto the vec env
    base = os.path.basename(model_path).lower()
    if base.startswith("ppo_"):
        model = PPO.load(model_path, env=venv)
    elif base.startswith("sac_"):
        model = SAC.load(model_path, env=venv)
    elif base.startswith("td3_"):
        model = TD3.load(model_path, env=venv)
    else:
        # default to PPO if ambiguous
        model = PPO.load(model_path, env=venv)

    # 4) Roll out episodes (VecEnv API: obs, rewards, dones, infos)
    for ep in range(episodes):
        obs = venv.reset()
        ep_return = 0.0
        done = [False]
        while not done[0]:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, rewards, dones, infos = venv.step(action)
            ep_return += float(rewards.mean())
            done = dones

        success = bool(infos[0].get("is_success", False))
        print(f"Episode {ep+1}: return={ep_return:.3f}  success={success}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--xml", type=str, required=True)
    p.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac", "td3"])
    p.add_argument("--logdir", type=str, default="runs/hand_cube")
    p.add_argument("--timesteps", type=int, default=10000)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--eval", type=str, default=None)
    args = p.parse_args()

    if args.eval:
        evaluate(args.xml, args.eval)
    else:
        train(args.xml, args.algo, args.logdir, total_iters=args.iters, timesteps=args.timesteps)

