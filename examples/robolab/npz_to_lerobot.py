# SPDX-License-Identifier: Apache-2.0
"""Turn gen_demos.py .npz episodes into a LeRobot v2.1 dataset for pi05 fine-tuning.

Same feature schema as convert_robolab_to_lerobot.py --robot franka, so
pi05_robolab_franka's LeRobotRoboLabDataConfig consumes it unchanged. We skip the
RoboLab HDF5 intermediate because the demos are generated directly in the eval
harness (same robot/cameras/scene), not by a RoboLab recording run.

    uv run examples/robolab/npz_to_lerobot.py --src ~/Desktop/trc/datasets/food_packing_demos \
        --repo-id trc/robolab_franka_foodpacking
"""
import argparse, pathlib, shutil

import numpy as np
import tqdm
from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

IMAGE_HW = (180, 320)
N_ARM = 7


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--repo-id", default="trc/robolab_franka_foodpacking")
    p.add_argument("--overwrite", action="store_true", default=True)
    a = p.parse_args()

    files = sorted(pathlib.Path(a.src).glob("ep*.npz"))
    if not files:
        raise SystemExit(f"no .npz in {a.src}")

    dest = HF_LEROBOT_HOME / a.repo_id
    if dest.exists():
        if not a.overwrite:
            raise FileExistsError(dest)
        shutil.rmtree(dest)

    ds = LeRobotDataset.create(
        repo_id=a.repo_id, robot_type="panda", fps=15,
        features={
            "exterior_image": {"dtype": "image", "shape": (*IMAGE_HW, 3), "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "image", "shape": (*IMAGE_HW, 3), "names": ["height", "width", "channel"]},
            "joint_position": {"dtype": "float32", "shape": (N_ARM,), "names": ["joint_position"]},
            "gripper_position": {"dtype": "float32", "shape": (1,), "names": ["gripper_position"]},
            "actions": {"dtype": "float32", "shape": (N_ARM + 1,), "names": ["actions"]},
        },
        image_writer_threads=8, image_writer_processes=4,
    )

    frames = 0
    for f in tqdm.tqdm(files, desc="episodes"):
        # Materialise every array ONCE. NpzFile.__getitem__ re-reads and re-inflates the whole
        # compressed array on each access, so indexing d["exterior_image"][i] inside the frame
        # loop cost 48 ms x 4 arrays x 618 frames = 2 min per episode (profiled).
        with np.load(f, allow_pickle=True) as z:
            d = {k: z[k] for k in z.files}
        prompt = str(d["prompt"]) if "prompt" in d else "put the mustard bottle in the left bin and the spam can in the right bin"
        n = len(d["actions"])
        for i in range(n):
            ds.add_frame({
                "exterior_image": d["exterior_image"][i],
                "wrist_image": d["wrist_image"][i],
                "joint_position": d["joint_position"][i].astype(np.float32),
                "gripper_position": d["gripper_position"][i].astype(np.float32),
                "actions": d["actions"][i].astype(np.float32),
                "task": prompt,
            })
            frames += 1
        ds.save_episode()

    print(f"\nwrote {len(files)} episodes / {frames} frames -> {dest}")


main()
