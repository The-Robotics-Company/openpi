"""Convert RoboLab (IsaacLab) recordings into a LeRobot v2.1 dataset for openpi fine-tuning.

RoboLab output layout (one folder per task, see RoboLab docs/data.md):
    <output>/episode_results.jsonl            one line per episode: env_name, episode, env_id, instruction, success, ...
    <output>/<Task>/run_<r>.hdf5              data/demo_<env_id>/{actions, obs/*, states/*, ...}
    <output>/<Task>/<instr>_<r>_env<i>.mp4    policy cameras side by side (exterior | wrist), 15 fps, half resolution

Two recording modes are supported:
  1. Recorded with `--record-image-data` (preferred): `obs/arm_joint_pos`, `obs/gripper_pos` and the camera images
     are in the HDF5 and are the exact policy observations for `actions[t]`.
  2. Default recording (no `obs/` group, e.g. our pi05 eval runs): joint positions come from
     `states/articulation/robot/joint_position` (post-step state), images are decoded from the side-by-side mp4.
     Frame t of the video is the observation *after* step t, so sample t pairs (state[t], frame[t]) with actions[t+1].

Output features (fps 15, robot_type from --robot):
    exterior_image, wrist_image        uint8 (180, 320, 3)
    joint_position                     float32 (n_arm,)   rad
    gripper_position                   float32 (1,)       0 open .. 1 closed
    actions                            float32 (n_arm+1,) absolute joint targets (rad) + gripper 0/1
    task                               instruction string (-> prompt_from_task)

Usage (from the openpi-train repo, uses the pinned LeRobot v2.1):
    uv run examples/robolab/convert_robolab_to_lerobot.py --output-dir ~/git/RoboLab/output/<run> \
        --repo-id trc/robolab_piperx --robot piperx [--only-success] [--tasks BananaInBowlTask ...]
The dataset is written to $HF_LEROBOT_HOME/<repo_id> (default ~/.cache/huggingface/lerobot/<repo_id>).
"""

import dataclasses
import json
import pathlib
import re
import shutil

import cv2
import h5py
from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tqdm
import tyro

IMAGE_HW = (180, 320)


@dataclasses.dataclass(frozen=True)
class RobotSpec:
    robot_type: str
    n_arm: int
    # Fallback (no obs/ group): index of the gripper joint in states/.../joint_position and how to map it to 0..1.
    gripper_joint_index: int
    gripper_to_01: callable  # joint value -> 0 open .. 1 closed
    # Fallback: which obs camera names to look for when obs/ exists (exterior, wrist).
    exterior_obs_keys: tuple[str, ...]
    wrist_obs_keys: tuple[str, ...]


ROBOTS = {
    # Franka + Robotiq 2F-85 (RoboLab DROID config): finger_joint 0 (open) .. pi/4 (closed).
    "franka": RobotSpec("panda", 7, 7, lambda v: np.clip(v / (np.pi / 4), 0.0, 1.0),
                        ("over_shoulder_left_camera", "exo_camera"), ("wrist_cam", "wrist_camera")),
    # AgileX Piper X (robolab/robots/piper_x.py): gripper_joint1 0.025 m (open) .. 0 (closed).
    "piperx": RobotSpec("piper_x", 6, 6, lambda v: np.clip(1.0 - v / 0.025, 0.0, 1.0),
                        ("exo_camera", "over_shoulder_left_camera"), ("wrist_camera", "wrist_cam")),
}


def _resize(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, (IMAGE_HW[1], IMAGE_HW[0]), interpolation=cv2.INTER_AREA)


def _clean_instruction(instruction: str) -> str:
    # Mirrors robolab/eval/episode.py: re.sub(r'[^\w\s]', '', s).replace(' ', '_')
    return re.sub(r"[^\w\s]", "", instruction).replace(" ", "_")


def _read_video_frames(path: pathlib.Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def _pick_key(group: h5py.Group, candidates: tuple[str, ...], contains: str | None = None) -> str | None:
    for k in candidates:
        if k in group:
            return k
    if contains:
        for k in group:
            if contains in k:
                return k
    return None


def _episode_samples(demo: h5py.Group, task_dir: pathlib.Path, ep: dict, spec: RobotSpec):
    """Yield (exterior, wrist, joint_pos, gripper_pos, action) per step."""
    actions = demo["actions"][:].astype(np.float32)
    n = spec.n_arm
    if "obs" in demo and "arm_joint_pos" in demo["obs"]:
        obs = demo["obs"]
        ext_key = _pick_key(obs, spec.exterior_obs_keys)
        wr_key = _pick_key(obs, spec.wrist_obs_keys, contains="wrist")
        if ext_key is None or wr_key is None:
            raise KeyError(f"camera obs not found in {list(obs.keys())}; record with --record-image-data")
        joints = obs["arm_joint_pos"][:].astype(np.float32).reshape(len(actions), -1)[:, :n]
        grip = obs["gripper_pos"][:].astype(np.float32).reshape(len(actions), -1)[:, :1]
        for t in range(len(actions)):
            yield _resize(obs[ext_key][t]), _resize(obs[wr_key][t]), joints[t], grip[t], actions[t, : n + 1]
        return

    # Fallback: post-step states + side-by-side mp4.
    jp = demo["states/articulation/robot/joint_position"][:].astype(np.float32)
    joints = jp[:, :n]
    grip = spec.gripper_to_01(jp[:, spec.gripper_joint_index : spec.gripper_joint_index + 1]).astype(np.float32)
    run_idx, env_id = int(ep["episode"]) // ep["_num_envs"], int(ep["env_id"])
    video = task_dir / f"{_clean_instruction(ep['instruction'])}_{run_idx}_env{env_id}.mp4"
    if not video.exists():
        raise FileNotFoundError(video)
    frames = _read_video_frames(video)
    steps = min(len(frames), len(actions) - 1)
    for t in range(steps):
        frame = frames[t]
        half = frame.shape[1] // 2
        yield _resize(frame[:, :half]), _resize(frame[:, half:]), joints[t], grip[t], actions[t + 1, : n + 1]


def main(
    output_dir: str,
    repo_id: str,
    robot: str = "piperx",
    only_success: bool = True,
    tasks: list[str] | None = None,
    max_episodes_per_task: int | None = None,
    overwrite: bool = True,
):
    spec = ROBOTS[robot]
    out_root = pathlib.Path(output_dir)
    results = [json.loads(l) for l in (out_root / "episode_results.jsonl").read_text().splitlines() if l.strip()]
    # num_envs is not stored per line; infer from the largest env_id seen per task.
    num_envs = {}
    for r in results:
        num_envs[r["env_name"]] = max(num_envs.get(r["env_name"], 0), int(r["env_id"]) + 1)
    if only_success:
        results = [r for r in results if r.get("success")]
    if tasks:
        results = [r for r in results if r["env_name"] in set(tasks)]
    per_task = {}
    for r in results:
        per_task.setdefault(r["env_name"], []).append(r)
    if max_episodes_per_task:
        per_task = {k: v[:max_episodes_per_task] for k, v in per_task.items()}
    episodes = [r for v in per_task.values() for r in v]
    print(f"{len(episodes)} episodes across {len(per_task)} tasks -> {repo_id}")

    dest = HF_LEROBOT_HOME / repo_id
    if dest.exists():
        if not overwrite:
            raise FileExistsError(dest)
        shutil.rmtree(dest)
    n = spec.n_arm
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type=spec.robot_type,
        fps=15,
        features={
            "exterior_image": {"dtype": "image", "shape": (*IMAGE_HW, 3), "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "image", "shape": (*IMAGE_HW, 3), "names": ["height", "width", "channel"]},
            "joint_position": {"dtype": "float32", "shape": (n,), "names": ["joint_position"]},
            "gripper_position": {"dtype": "float32", "shape": (1,), "names": ["gripper_position"]},
            "actions": {"dtype": "float32", "shape": (n + 1,), "names": ["actions"]},
        },
        image_writer_threads=8,
        image_writer_processes=4,
    )

    skipped = 0
    for ep in tqdm.tqdm(episodes, desc="episodes"):
        task_dir = out_root / ep["env_name"]
        ep = dict(ep, _num_envs=num_envs[ep["env_name"]])
        run_idx = int(ep["episode"]) // ep["_num_envs"]
        h5_path = task_dir / f"run_{run_idx}.hdf5"
        try:
            with h5py.File(h5_path, "r") as f:
                demo = f["data"][f"demo_{int(ep['env_id'])}"]
                count = 0
                for ext, wr, jp, gp, act in _episode_samples(demo, task_dir, ep, spec):
                    dataset.add_frame(
                        {
                            "exterior_image": ext,
                            "wrist_image": wr,
                            "joint_position": jp,
                            "gripper_position": gp,
                            "actions": act,
                            "task": ep["instruction"],
                        }
                    )
                    count += 1
            if count == 0:
                raise ValueError("no samples")
            dataset.save_episode()
        except Exception as e:  # noqa: BLE001
            skipped += 1
            print(f"skip {ep['env_name']} ep{ep['episode']}: {type(e).__name__}: {e}")
            dataset.episode_buffer = dataset.create_episode_buffer()  # drop the partial episode
    print(f"done: {len(episodes) - skipped} episodes written, {skipped} skipped -> {dest}")


if __name__ == "__main__":
    tyro.cli(main)
