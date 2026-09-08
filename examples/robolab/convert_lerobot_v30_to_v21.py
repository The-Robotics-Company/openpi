"""Convert a LeRobot v3.0 Piper X dataset (lerobot 0.4.x, e.g. trc-spaces / piper-x-policy exports) into the
LeRobot v2.1 layout openpi's pinned LeRobot reads, with the feature names `LeRobotRoboLabDataConfig` expects.

Input (v3.0):  observation.images.exo / .wrist (video), observation.state.joint_position (6),
               observation.state.gripper_position (1, 1 = open, per-jaw travel / 0.05),
               action.joint_position (6, absolute rad), action.gripper_position (1, same scale), task_index.
Output (v2.1): exterior_image, wrist_image (180x320 uint8), joint_position (6), gripper_position (1),
               actions (7 = joints + gripper), task.

Gripper state and action stay CONTINUOUS (as in the DROID recipe: the model regresses the gripper position and the
execution client binarises it), only the polarity is flipped to RoboLab's Piper X convention (0 = open, 1 = closed):
    gripper_position' = clip(1 - g / --gripper-open-value, 0, 1)      state
    gripper action'   = clip(1 - g / --gripper-open-value, 0, 1)      action
Measured command levels in piperx_rubiks_cube_bowl after the flip: approach 0.18 (jaws 29 mm, partially open),
close 0.97 (15 frames), hold 0.27-0.33 (cube width), release 0.0. The Piper eval client must therefore threshold
at ~0.22 (between approach and hold), ideally with hysteresis (close > 0.22, open < 0.10) -- NOT at DROID's 0.5.
Frames whose joint action is exactly all-zero (a recording artifact on the first ~15 frames of each episode) are dropped
with --drop-zero-actions (default on).

Usage:
    HF_LEROBOT_HOME=/mnt/efs/datasets/lerobot_v21 uv run examples/robolab/convert_lerobot_v30_to_v21.py \
        --src /mnt/efs/datasets/lerobot_v21/_incoming/piperx_rubiks_cube_bowl --repo-id trc/robolab_piperx
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
    frames = []
    with av.open(str(path)) as c:
        for f in c.decode(c.streams.video[0]):
            frames.append(f.to_ndarray(format="rgb24"))
    return frames


def _resize(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, (IMAGE_HW[1], IMAGE_HW[0]), interpolation=cv2.INTER_AREA)


def main(
    src: str,
    repo_id: str,
    exo_key: str = "observation.images.exo",
    wrist_key: str = "observation.images.wrist",
    gripper_open_value: float = 0.7,
    drop_zero_actions: bool = True,
    overwrite: bool = True,
):
    src_p = pathlib.Path(src)
    info = json.loads((src_p / "meta" / "info.json").read_text())
    assert info["codebase_version"].startswith("v3"), info["codebase_version"]
    fps = int(info["fps"])
    robot_type = info.get("robot_type", "piper_x")

    tasks_tbl = pq.read_table(src_p / "meta" / "tasks.parquet").to_pandas()
    # v3 tasks.parquet: index = task string, column task_index.
    task_by_index = {int(r["task_index"]): str(idx) for idx, r in tasks_tbl.iterrows()}

    episodes = pq.read_table(*sorted((src_p / "meta" / "episodes").rglob("*.parquet"))[:1]).to_pandas()
    for extra in sorted((src_p / "meta" / "episodes").rglob("*.parquet"))[1:]:
        episodes = episodes._append(pq.read_table(extra).to_pandas())  # noqa: SLF001
    data = pq.read_table(*sorted((src_p / "data").rglob("*.parquet"))[:1]).to_pandas()
    for extra in sorted((src_p / "data").rglob("*.parquet"))[1:]:
        data = data._append(pq.read_table(extra).to_pandas())  # noqa: SLF001
    data = data.sort_values("index").reset_index(drop=True)

    dest = HF_LEROBOT_HOME / repo_id
    if dest.exists():
        if not overwrite:
            raise FileExistsError(dest)
        shutil.rmtree(dest)
    ds = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type=robot_type,
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

    video_cache: dict[tuple, list[np.ndarray]] = {}

    def frames_for(key: str, ep_row) -> list[np.ndarray]:
        ci, fi = int(ep_row[f"videos/{key}/chunk_index"]), int(ep_row[f"videos/{key}/file_index"])
        path = src_p / info["video_path"].format(video_key=key, chunk_index=ci, file_index=fi)
        if (key, ci, fi) not in video_cache:
            video_cache[(key, ci, fi)] = _decode_all(path)
        start = int(round(float(ep_row[f"videos/{key}/from_timestamp"]) * fps))
        return video_cache[(key, ci, fi)][start : start + int(ep_row["length"])]

    dropped = 0
    for _, ep in tqdm.tqdm(episodes.iterrows(), total=len(episodes), desc="episodes"):
        rows = data.iloc[int(ep["dataset_from_index"]) : int(ep["dataset_to_index"])]
        exo, wr = frames_for(exo_key, ep), frames_for(wrist_key, ep)
        assert len(exo) == len(rows) == len(wr), (len(exo), len(rows), len(wr))
        task = task_by_index[int(rows["task_index"].iloc[0])]
        for i, (_, r) in enumerate(rows.iterrows()):
            aj = np.asarray(r["action.joint_position"], dtype=np.float32)
            if drop_zero_actions and not np.any(aj):
                dropped += 1
                continue
            g_state = float(np.asarray(r["observation.state.gripper_position"]).reshape(-1)[0])
            g_act = float(np.asarray(r["action.gripper_position"]).reshape(-1)[0])
            ds.add_frame(
                {
                    "exterior_image": _resize(exo[i]),
                    "wrist_image": _resize(wr[i]),
                    "joint_position": np.asarray(r["observation.state.joint_position"], dtype=np.float32),
                    "gripper_position": np.array([np.clip(1.0 - g_state / gripper_open_value, 0.0, 1.0)], np.float32),
                    "actions": np.concatenate([aj, [np.clip(1.0 - g_act / gripper_open_value, 0.0, 1.0)]]).astype(np.float32),
                    "task": task,
                }
            )
        ds.save_episode()
    print(f"done -> {dest} ({len(episodes)} episodes, {dropped} zero-action frames dropped)")


if __name__ == "__main__":
    tyro.cli(main)
