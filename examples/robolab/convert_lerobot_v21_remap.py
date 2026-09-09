"""Remap a raw LeRobot v2.1 Piper X export (trc-spaces generator keys) into the v2.1 layout/feature names that
`LeRobotRoboLabDataConfig` expects. Same transforms as convert_lerobot_v30_to_v21.py, for v2.1 inputs
(per-episode parquet under data/chunk-*/episode_XXXXXX.parquet, per-episode mp4 under videos/chunk-*/<key>/).

  exterior_image, wrist_image (180x320 uint8) <- observation.images.exo / .wrist
  joint_position (6)                          <- observation.state.joint_position
  gripper_position (1)                        <- clip(1 - observation.state.gripper_position / open_value)   0 open .. 1 closed
  actions (7)                                 <- action.joint_position ++ clip(1 - action.gripper_position / open_value)  (continuous)
  task                                        <- --task-override, else the dataset's task string
Frames with an all-zero joint action (the first ~15 per episode) are dropped.

Usage:
  HF_LEROBOT_HOME=/mnt/efs/datasets/lerobot_v21 uv run examples/robolab/convert_lerobot_v21_remap.py \
      --src /mnt/efs/datasets/lerobot_v21/_incoming/piperx_rubiks_cube_bowl_fixedpose_varpath_v21 \
      --repo-id trc/robolab_piperx_fixedpose --task-override "Put the cube in the bowl"
"""

import json
import pathlib
import shutil

import av
import cv2
from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import pyarrow.parquet as pq
import tqdm
import tyro

IMAGE_HW = (180, 320)


def _decode_all(path: pathlib.Path) -> list[np.ndarray]:
    with av.open(str(path)) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(c.streams.video[0])]


def _resize(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, (IMAGE_HW[1], IMAGE_HW[0]), interpolation=cv2.INTER_AREA)


def main(
    src: str,
    repo_id: str,
    task_override: str | None = None,
    exo_key: str = "observation.images.exo",
    wrist_key: str = "observation.images.wrist",
    gripper_open_value: float = 0.7,
    drop_zero_actions: bool = True,
    max_episodes: int | None = None,
    overwrite: bool = True,
):
    src_p = pathlib.Path(src)
    info = json.loads((src_p / "meta" / "info.json").read_text())
    assert info["codebase_version"].startswith("v2"), info["codebase_version"]
    fps = int(info["fps"])
    tasks = {json.loads(l)["task_index"]: json.loads(l)["task"] for l in (src_p / "meta" / "tasks.jsonl").read_text().splitlines() if l.strip()}
    episodes = [json.loads(l) for l in (src_p / "meta" / "episodes.jsonl").read_text().splitlines() if l.strip()]
    if max_episodes:
        episodes = episodes[:max_episodes]

    dest = HF_LEROBOT_HOME / repo_id
    if dest.exists():
        if not overwrite:
            raise FileExistsError(dest)
        shutil.rmtree(dest)
    ds = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type=info.get("robot_type", "piper_x"),
        fps=fps,
        features={
            "exterior_image": {"dtype": "image", "shape": (*IMAGE_HW, 3), "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "image", "shape": (*IMAGE_HW, 3), "names": ["height", "width", "channel"]},
            "joint_position": {"dtype": "float32", "shape": (6,), "names": ["joint_position"]},
            "gripper_position": {"dtype": "float32", "shape": (1,), "names": ["gripper_position"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        },
        image_writer_threads=8,
        image_writer_processes=4,
    )
    chunk_size = int(info.get("chunks_size", 1000))
    dropped = 0
    for ep in tqdm.tqdm(episodes, desc="episodes"):
        e = int(ep["episode_index"]); chunk = e // chunk_size
        rows = pq.read_table(src_p / info["data_path"].format(episode_chunk=chunk, episode_index=e)).to_pandas()
        exo = _decode_all(src_p / info["video_path"].format(episode_chunk=chunk, video_key=exo_key, episode_index=e))
        wr = _decode_all(src_p / info["video_path"].format(episode_chunk=chunk, video_key=wrist_key, episode_index=e))
        n = min(len(rows), len(exo), len(wr))
        if n != len(rows):
            print(f"episode {e}: {len(rows)} rows vs {len(exo)}/{len(wr)} frames; truncating to {n}")
        task = task_override or tasks[int(rows["task_index"].iloc[0])]
        for i in range(n):
            r = rows.iloc[i]
            aj = np.asarray(r["action.joint_position"], dtype=np.float32)
            if drop_zero_actions and not np.any(aj):
                dropped += 1
                continue
            g_state = float(np.asarray(r["observation.state.gripper_position"]).reshape(-1)[0])
            g_act = float(np.asarray(r["action.gripper_position"]).reshape(-1)[0])
            ds.add_frame({
                "exterior_image": _resize(exo[i]),
                "wrist_image": _resize(wr[i]),
                "joint_position": np.asarray(r["observation.state.joint_position"], dtype=np.float32),
                "gripper_position": np.array([np.clip(1.0 - g_state / gripper_open_value, 0.0, 1.0)], np.float32),
                "actions": np.concatenate([aj, [np.clip(1.0 - g_act / gripper_open_value, 0.0, 1.0)]]).astype(np.float32),
                "task": task,
            })
        ds.save_episode()
    print(f"done -> {dest} ({len(episodes)} episodes, {dropped} zero-action frames dropped, task={task!r})")


if __name__ == "__main__":
    tyro.cli(main)
