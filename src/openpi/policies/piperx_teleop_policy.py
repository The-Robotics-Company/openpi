"""Transforms for Piper X *real-hardware teleop* recordings (XR-controller teleop, not RoboLab sim).

These datasets are written as canonical LeRobot v2.1 with flat, standard feature names:

  observation.state            (7,) float  joint1..joint6 (rad) ++ gripper aperture (m)
  action                       (7,) float  commanded joints (rad) ++ commanded aperture (m)
  observation.images.external  video, static camera
  observation.images.wrist     video, moves with the arm

`RoboLabInputs` expects the arm joints and the gripper as *separate* keys, with the gripper
already in RoboLab's 0 = open .. 1 = closed convention. This module bridges the two: split the
flat vector and rescale the gripper, then reuse `RoboLabInputs`/`RoboLabOutputs` and the delta
action transform unchanged.

Gripper: the recordings carry raw aperture in metres, larger = more open, so the mapping is
`clip(1 - aperture / open_value)` — the same formula `examples/robolab/convert_lerobot_v21_remap.py`
applies, but at transform time instead of baked into a converted copy of the dataset.

Must run *before* `RoboLabInputs` (which builds `state`) and therefore before `DeltaActions`,
so the gripper is already normalised when the delta mask leaves it absolute.
"""

import dataclasses

import numpy as np

from openpi import transforms


@dataclasses.dataclass(frozen=True)
class PiperXTeleopStateSplit(transforms.DataTransformFn):
    """Split the flat teleop state/action into the keys `RoboLabInputs` expects."""

    # Piper X has 6 arm joints; the gripper is the trailing dimension.
    n_arm_joints: int = 6

    # Aperture in metres at which the jaw is fully open. Maps to 0.0 (open); 0 m maps to 1.0 (closed).
    # 0.07 for the piper_x_pick_cube_v1 recordings (commanded aperture maxes at exactly 0.070000).
    gripper_open_value: float = 0.07

    def _normalize_gripper(self, aperture: np.ndarray) -> np.ndarray:
        """Aperture in metres -> 0.0 open .. 1.0 closed."""
        return np.clip(1.0 - aperture / self.gripper_open_value, 0.0, 1.0)

    def __call__(self, data: dict) -> dict:
        data = dict(data)
        n = self.n_arm_joints

        state = np.asarray(data.pop("observation/state"), dtype=np.float32)
        data["observation/joint_position"] = state[..., :n]
        data["observation/gripper_position"] = self._normalize_gripper(state[..., n : n + 1])

        if "actions" in data:
            # Copy: DeltaActions mutates `actions` in place further down the chain.
            actions = np.asarray(data["actions"], dtype=np.float32).copy()
            actions[..., n] = self._normalize_gripper(actions[..., n])
            data["actions"] = actions

        return data


def make_piperx_teleop_example(n_arm_joints: int = 6) -> dict:
    """A random input example, matching the raw dataset keys (pre-repack shapes, metres for the gripper)."""
    return {
        "observation/image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/state": np.concatenate(
            [np.random.rand(n_arm_joints).astype(np.float32), np.array([0.035], dtype=np.float32)]
        ),
        "prompt": "pick up the cube and put it in the cup",
    }
