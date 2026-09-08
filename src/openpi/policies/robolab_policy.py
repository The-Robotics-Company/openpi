"""Input/output transforms for policies trained on RoboLab (IsaacLab) recordings.

Covers any single-arm RoboLab robot with a joint-position action space:
  - Piper X:       6 arm joints + 1 gripper  -> action_dim 7
  - DROID Franka:  7 arm joints + 1 gripper  -> action_dim 8

Expected (repacked) keys, produced by `examples/robolab/convert_robolab_to_lerobot.py`:
  observation/image             exterior camera, uint8 HWC (any resolution; model transforms resize to 224)
  observation/wrist_image       wrist camera, uint8 HWC
  observation/joint_position    (n_arm_joints,) rad
  observation/gripper_position  (1,) 0 = open .. 1 = closed  (RoboLab convention, same polarity as the action)
  actions                       (horizon, n_arm_joints + 1)  absolute joint targets (rad) + gripper 0 open .. 1 closed (continuous)
  prompt                        task instruction string

Delta-vs-absolute is handled outside this file by DeltaActions / AbsoluteActions in the data config,
so the model always sees delta joint actions with an absolute gripper (same as pi05_droid_jointpos).
State/action padding to the model's 32 dims is done by ModelTransformFactory (PadStatesAndActions).
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_robolab_example(n_arm_joints: int = 6) -> dict:
    """A random input example, used by compute_norm_stats/tests and to sanity-check the transform."""
    return {
        "observation/image": np.random.randint(256, size=(180, 320, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(180, 320, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(n_arm_joints).astype(np.float32),
        "observation/gripper_position": np.random.rand(1).astype(np.float32),
        "prompt": "pick up the cube and place it in the cup",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:  # CHW (LeRobot video decode) -> HWC
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RoboLabInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        gripper = np.atleast_1d(np.asarray(data["observation/gripper_position"], dtype=np.float32))
        state = np.concatenate([np.asarray(data["observation/joint_position"], dtype=np.float32), gripper])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # pi0/pi05 mask the missing third camera; FAST models do not.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class RoboLabOutputs(transforms.DataTransformFn):
    # n_arm_joints + 1 (gripper). 7 for Piper X, 8 for the DROID Franka.
    action_dim: int = 7

    def __call__(self, data: dict) -> dict:
        # Model predicts 32 dims; keep the real ones. Gripper stays continuous here (DROID convention);
        # the execution client binarises it: 0.5 for DROID data, ~0.22 with hysteresis for the planner-generated Piper X data.
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
