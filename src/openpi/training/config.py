"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.policies.piperx_teleop_policy as piperx_teleop_policy
import openpi.policies.robolab_policy as robolab_policy
import openpi.shared.download as _download
import openpi.shared.nnx_utils as nnx_utils
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.polaris_config as polaris_config
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.

    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = (
        droid_rlds_dataset.RLDSDataset(
            name="droid",
            version="1.0.1",
            weight=1.0,
            filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        ),
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            datasets=self.datasets,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotRoboLabDataConfig(DataConfigFactory):
    """RoboLab (IsaacLab) recordings converted with examples/robolab/convert_robolab_to_lerobot.py.

    LeRobot features written by the converter: exterior_image, wrist_image, joint_position (n_arm_joints),
    gripper_position (1, 0 open .. 1 closed), actions (n_arm_joints + 1: absolute joint targets in rad + continuous gripper 0 open .. 1 closed;
    binarised only at execution, like DROID),
    task (instruction). 15 fps = RoboLab's 15 Hz control rate.
    """

    # 6 for Piper X, 7 for the DROID Franka.
    n_arm_joints: int = 6
    # n_arm_joints + 1.
    action_dim: int = 7

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "exterior_image",
                        "observation/wrist_image": "wrist_image",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[robolab_policy.RoboLabInputs(model_type=model_config.model_type)],
            outputs=[robolab_policy.RoboLabOutputs(action_dim=self.action_dim)],
        )
        # Train on joint deltas (arm joints only, gripper stays absolute) and convert back at inference,
        # exactly like pi05_droid_jointpos. RoboLab's client expects absolute joint targets.
        delta_action_mask = _transforms.make_bool_mask(self.n_arm_joints, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )
        model_transforms = ModelTransformFactory()(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotPiperXTeleopDataConfig(DataConfigFactory):
    """Piper X real-hardware teleop recordings, read from canonical LeRobot v2.1 in place.

    Unlike the RoboLab sim exports, these datasets are already written with standard LeRobot
    feature names, so nothing needs converting: the repack below renames the keys and
    `PiperXTeleopStateSplit` does the flat-state split plus the gripper metre -> 0..1 rescale.
    Everything after that -- `RoboLabInputs`/`RoboLabOutputs`, the delta-action mask, the model
    transforms -- is shared with `LeRobotRoboLabDataConfig` unchanged.

    Reading the dataset in place (rather than through the 180x320 converter) keeps the native
    480x640 video, so `resize_with_pad` to 224 yields 168x224 of real content instead of 126x224
    upsampled from a squashed 16:9 intermediate.

    Expected raw features:
      observation.state           (7,)  joint1..joint6 (rad) ++ gripper aperture (m)
      action                      (7,)  commanded joints (rad) ++ commanded aperture (m)
      observation.images.external video, static camera  -> base_0_rgb
      observation.images.wrist    video, on the arm     -> left_wrist_0_rgb
    """

    # 6 for Piper X.
    n_arm_joints: int = 6
    # n_arm_joints + 1.
    action_dim: int = 7
    # Aperture in metres at which the jaw is fully open. 0.07 for piper_x_pick_cube_v1.
    gripper_open_value: float = 0.07
    # LeRobot video keys, in case a future recording renames the cameras.
    exterior_image_key: str = "observation.images.external"
    wrist_image_key: str = "observation.images.wrist"
    # The action-chunk column to slice with delta_timestamps. Canonical LeRobot names it "action";
    # only the RoboLab converter's output uses the plural "actions" that DataConfig defaults to.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": self.exterior_image_key,
                        "observation/wrist_image": self.wrist_image_key,
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[
                # Must precede RoboLabInputs: it builds `state` from the split keys.
                piperx_teleop_policy.PiperXTeleopStateSplit(
                    n_arm_joints=self.n_arm_joints,
                    gripper_open_value=self.gripper_open_value,
                ),
                robolab_policy.RoboLabInputs(model_type=model_config.model_type),
            ],
            outputs=[robolab_policy.RoboLabOutputs(action_dim=self.action_dim)],
        )
        # Same as LeRobotRoboLabDataConfig: joint deltas for the arm, gripper stays absolute.
        # Mean |action - state| is 0.0184 rad on piper_x_pick_cube_v1, so absolute targets are
        # predictable from proprio alone -- this mask is what stops the policy ignoring the images.
        delta_action_mask = _transforms.make_bool_mask(self.n_arm_joints, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )
        model_transforms = ModelTransformFactory()(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        # Joint-position variant served for RoboLab / Piper X sim eval (ported from xuningy/openpi).
        name="pi05_droid_jointpos",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[_transforms.AbsoluteActions(_transforms.make_bool_mask(7, -1)), droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        # Joint-position variant served for RoboLab / Piper X sim eval (ported from xuningy/openpi).
        name="pi0_droid_jointpos",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[_transforms.AbsoluteActions(_transforms.make_bool_mask(7, -1)), droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        # Joint-position variant served for RoboLab / Piper X sim eval (ported from xuningy/openpi).
        name="pi0_fast_droid_jointpos",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[_transforms.AbsoluteActions(_transforms.make_bool_mask(7, -1)), droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0_config.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # Here is an example of loading a pi0-FAST model for full finetuning.
        # Modify action_dim and action_horizon to match your dataset (action horizon is equal to
        # the desired action chunk length).
        # The max_token_len is the maximum number of (non-image) tokens the model can handle.
        # This includes the tokenized prompt, proprioceptive state, and (FAST-tokenized) action tokens.
        # Choosing this value too small may chop off tokens at the end of your sequence (the code will throw
        # a warning), while choosing it too large will waste memory (since we pad each batch element to the
        # max_token_len). A good rule of thumb is to use approx 180 for single-arm robots, and approx 250 for
        # two-arm robots. Generally, err on the lower side here first, and potentially increase the value if
        # you see many warnings being thrown during training.
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instructions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
    ),
    #
    # Fine-tuning DROID configs.
    #
    TrainConfig(
        # This config is for fine-tuning pi0-FAST-base on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps should be sufficient, takes ~2 days on 8x H100s
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi05_full_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="/mnt/pi-data/kevin",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets/",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05-DROID on a custom (smaller) DROID dataset.
        # Here, we use LeRobot data format (like for all other fine-tuning examples)
        # To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
        name="pi05_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # Replace with your custom DROID LeRobot dataset repo id.
            repo_id="your_hf_username/my_droid_dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Important: reuse the original DROID norm stats during fine-tuning!
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    #
    # RoboLab (IsaacLab) fine-tuning configs. Data: examples/robolab/convert_robolab_to_lerobot.py.
    # Norm stats: `uv run scripts/compute_norm_stats.py --config-name <name>` (writes assets/<name>/<asset_id>/).
    #
    TrainConfig(
        # Piper X (6 joints + gripper), full fine-tune from pi05_base. Needs an 80 GB GPU.
        name="pi05_piperx",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim padded actions
            action_horizon=15,  # 1 s of actions at RoboLab's 15 Hz
        ),
        data=LeRobotRoboLabDataConfig(
            repo_id="trc/robolab_piperx",  # <HF_LEROBOT_HOME>/trc/robolab_piperx, see examples/robolab/README.md
            n_arm_joints=6,
            action_dim=7,
            base_config=DataConfig(prompt_from_task=True),
            # No assets_dir/asset_id: norm stats are computed fresh for this dataset by compute_norm_stats and stored
            # under assets/<config>/<repo_id>/ (Piper joint ranges differ from every pretraining robot).
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=32,
        # Defaults kept: AdamW (0.9/0.95, clip 1.0), 1k warmup -> 2.5e-5 cosine -> 2.5e-6 @30k, EMA 0.99, nothing frozen.
    ),
    TrainConfig(
        # Same recipe, LoRA on both Gemma towers so it fits a 48 GB L40S. EMA off, as in the other LoRA configs.
        name="pi05_piperx_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotRoboLabDataConfig(
            repo_id="trc/robolab_piperx",
            n_arm_joints=6,
            action_dim=7,
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=16,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        # Piper X REAL-HARDWARE teleop (piper_x_pick_cube_v1: 49 eps / 18,431 frames / 30 Hz, XR teleop).
        # Expert-only fine-tune from pi05_base: VLM frozen (SigLIP + Gemma-2B), only the action
        # expert trains. Chosen over LoRA because 49 episodes is little data and the pretrained VLM
        # is better left untouched at this scale.
        #
        # Dataset is read in place from HF_LEROBOT_HOME/<repo_id>; set
        #   HF_LEROBOT_HOME=/home/ubuntu/training/data
        # so repo_id resolves to .../data/teleop-data-hugo/piper_x_pick_cube_v1.
        #
        # Measured on this box (200-step smoke, LoRA at batch 16 / horizon 30):
        #   2.1 s/it steady state (4.5 s/it during warmup), GPU util median 100% / mean 82.8%.
        #   Checkpoints are ~8.9 GB each and take ~19 s to write.
        # Expert-only removes backward work through the frozen towers, so steps get FASTER and the
        # dataloader has less slack -- re-check GPU utilisation before trusting the 2.1 s/it figure.
        #
        # Still untuned for this data: action_horizon=30 (1 s at 30 Hz; RoboLab used 15 for 1 s at
        # 15 Hz) and the LR schedule, which is inherited from the sim configs.
        name="pi05_piperx_teleop_expert",
        # Absolute paths, tied to the g6e.xlarge training box (see the HF_LEROBOT_HOME note above).
        # compute_norm_stats.py has no path-override flag, so assets_base_dir has to live here rather
        # than being passed at launch. Change both if this config ever moves to another machine or
        # gets merged into the shared `trc` branch.
        assets_base_dir="/home/ubuntu/training/assets",
        checkpoint_base_dir="/home/ubuntu/training/runs",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=30,
        ),
        data=LeRobotPiperXTeleopDataConfig(
            repo_id="teleop-data-hugo/piper_x_pick_cube_v1",
            n_arm_joints=6,
            action_dim=7,
            gripper_open_value=0.07,
            base_config=DataConfig(prompt_from_task=True),
            # No assets_dir/asset_id: Piper X is not in the pretraining mixture, so norm stats are
            # computed fresh by compute_norm_stats into assets/<config>/<repo_id>/.
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        # changan's expert configs rescaled to 10k steps: 400 warmup keeps his 4% warmup fraction
        # (200/5k), rather than inheriting his absolute 200 and ending up at 2%. Peak/floor unchanged.
        # Cosine anneal kept (both changan's configs and the openpi default anneal); this means the
        # 2k/4k/6k checkpoints sit at different points on the LR curve and are NOT directly
        # comparable to 10k -- read them as "still improving?", not as candidates.
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=400, peak_lr=2.5e-5, decay_steps=10_000, decay_lr=2.5e-6
        ),
        num_train_steps=10_000,
        batch_size=16,
        log_interval=25,
        # Checkpoint every 2k -> 5 checkpoints (2k/4k/6k/8k/10k = 1.7/3.5/5.2/6.9/8.7 epochs),
        # ~8.9 GB each, so eval can pick the best by hardware rollout rather than by loss.
        # keep_period must match save_interval or the manager prunes the ones you wanted.
        save_interval=2_000,
        keep_period=2_000,
        num_workers=4,
        # Expert-only: freeze the SigLIP tower (".*img.*") and Gemma-2B (".*llm.*" minus the action
        # expert ".*llm.*_1.*"), so only the ~430M action expert trains. Same filter as
        # pi05_piperx_rubiks_expert / pi05_piperx_fixedpose_expert.
        freeze_filter=nnx.Any(
            nnx.All(nnx_utils.PathRegex(".*llm.*"), nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*"))),
            nnx_utils.PathRegex(".*img.*"),
        ),
        # EMA OFF. At 0.99 (changan's expert-config default) train.py:113 keeps a full shadow copy
        # of all 3.353B params, which OOM-killed the host at step 194/200: anon-rss 23.2 GB +
        # shmem 4.3 GB = 27.5 GB of this box's 30 GB, no swap. The LoRA smoke with ema_decay=None
        # completed the same 200 steps and wrote a checkpoint. Costs a modest final-quality bump;
        # the openpi LoRA configs disable it for the same reason.
        ema_decay=None,
    ),

    TrainConfig(
        # SIM counterpart of pi05_piperx_teleop_expert. Identical recipe, identical hyperparameters;
        # the ONLY change is repo_id -> the sim dataset. Built to isolate sim-vs-real on the same task.
        #
        # sim-data-1 was produced by replaying the 49 real teleop trajectories through
        # GT-scenes/usd/scene1.usd (see meta/sim_report.json): 49/49 successes, same 49 episodes,
        # same per-episode lengths, same 18,431 frames, same 30 Hz, same 7-dim schema.
        #   actions are BIT-IDENTICAL to the real set (same commanded joint targets replayed)
        #   states differ 0.018-0.043 rad/joint (sim physics tracking those commands)
        #   gripper normalised corr 0.943 vs real; sim reports TRUE jaw gap so it closes ~3 mm
        #     further than the real encoder, which deflects when the jaw stalls on the cube
        #   images differ substantially: PSNR 10.4-13.9 dB vs real, sim notably brighter
        # gripper_open_value stays 0.07, identical to the real runs, so preprocessing is
        # byte-identical and any result difference comes from the data rather than our handling.
        #
        # Dataset is read in place from HF_LEROBOT_HOME/<repo_id>; set
        #   HF_LEROBOT_HOME=/home/ubuntu/training/data
        # so repo_id resolves to .../data/teleop-data-hugo/piper_x_pick_cube_v1.
        #
        # Measured on this box (200-step smoke, LoRA at batch 16 / horizon 30):
        #   2.1 s/it steady state (4.5 s/it during warmup), GPU util median 100% / mean 82.8%.
        #   Checkpoints are ~8.9 GB each and take ~19 s to write.
        # Expert-only removes backward work through the frozen towers, so steps get FASTER and the
        # dataloader has less slack -- re-check GPU utilisation before trusting the 2.1 s/it figure.
        #
        # Still untuned for this data: action_horizon=30 (1 s at 30 Hz; RoboLab used 15 for 1 s at
        # 15 Hz) and the LR schedule, which is inherited from the sim configs.
        name="pi05_piperx_sim_expert",
        # Absolute paths, tied to the g6e.xlarge training box (see the HF_LEROBOT_HOME note above).
        # compute_norm_stats.py has no path-override flag, so assets_base_dir has to live here rather
        # than being passed at launch. Change both if this config ever moves to another machine or
        # gets merged into the shared `trc` branch.
        assets_base_dir="/home/ubuntu/training/assets",
        checkpoint_base_dir="/home/ubuntu/training/runs",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=30,
        ),
        data=LeRobotPiperXTeleopDataConfig(
            repo_id="sim-data-1/piper_x_pick_cube_v1",
            n_arm_joints=6,
            action_dim=7,
            gripper_open_value=0.07,
            base_config=DataConfig(prompt_from_task=True),
            # No assets_dir/asset_id: Piper X is not in the pretraining mixture, so norm stats are
            # computed fresh by compute_norm_stats into assets/<config>/<repo_id>/.
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        # changan's expert configs rescaled to 10k steps: 400 warmup keeps his 4% warmup fraction
        # (200/5k), rather than inheriting his absolute 200 and ending up at 2%. Peak/floor unchanged.
        # Cosine anneal kept (both changan's configs and the openpi default anneal); this means the
        # 2k/4k/6k checkpoints sit at different points on the LR curve and are NOT directly
        # comparable to 10k -- read them as "still improving?", not as candidates.
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=400, peak_lr=2.5e-5, decay_steps=10_000, decay_lr=2.5e-6
        ),
        num_train_steps=10_000,
        batch_size=16,
        log_interval=25,
        # Checkpoint every 2k -> 5 checkpoints (2k/4k/6k/8k/10k = 1.7/3.5/5.2/6.9/8.7 epochs),
        # ~8.9 GB each, so eval can pick the best by hardware rollout rather than by loss.
        # keep_period must match save_interval or the manager prunes the ones you wanted.
        save_interval=2_000,
        keep_period=2_000,
        num_workers=4,
        # Expert-only: freeze the SigLIP tower (".*img.*") and Gemma-2B (".*llm.*" minus the action
        # expert ".*llm.*_1.*"), so only the ~430M action expert trains. Same filter as
        # pi05_piperx_rubiks_expert / pi05_piperx_fixedpose_expert.
        freeze_filter=nnx.Any(
            nnx.All(nnx_utils.PathRegex(".*llm.*"), nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*"))),
            nnx_utils.PathRegex(".*img.*"),
        ),
        # EMA OFF. At 0.99 (changan's expert-config default) train.py:113 keeps a full shadow copy
        # of all 3.353B params, which OOM-killed the host at step 194/200: anon-rss 23.2 GB +
        # shmem 4.3 GB = 27.5 GB of this box's 30 GB, no swap. The LoRA smoke with ema_decay=None
        # completed the same 200 steps and wrote a checkpoint. Costs a modest final-quality bump;
        # the openpi LoRA configs disable it for the same reason.
        ema_decay=None,
    ),

    TrainConfig(
        # ft02: pi05_piperx_teleop_expert PLUS LoRA adapters on Gemma-2B. Single-variable change
        # from that config -- SigLIP still frozen, action expert still trained in full -- so eval
        # differences are attributable to the Gemma adaptation alone.
        #   trainable 458.0M of 3.381B (13.54%) = 427.93M action expert + 27.87M LoRA + 2.17M proj
        #   vs ft01's 430.1M (12.83%).
        # Rationale: Gemma-2B is the backbone fusing vision tokens, text and state into what the
        # action expert reads, so adapting it can help even though the prompt is constant.
        # Caveat: pi0.5's headline feature is knowledge insulation (training actions WITHOUT
        # disturbing the VLM); this deliberately relaxes that, so watch for brittleness on cube
        # positions outside the training distribution rather than for worse loss.
        # NOTE: LoRA forces a backward pass through Gemma-2B that the expert-only config skips
        # entirely -- slower per step, and Gemma activations must now be retained (new OOM risk).
        #
        # Dataset is read in place from HF_LEROBOT_HOME/<repo_id>; set
        #   HF_LEROBOT_HOME=/home/ubuntu/training/data
        # so repo_id resolves to .../data/teleop-data-hugo/piper_x_pick_cube_v1.
        #
        # Measured on this box (200-step smoke, LoRA at batch 16 / horizon 30):
        #   2.1 s/it steady state (4.5 s/it during warmup), GPU util median 100% / mean 82.8%.
        #   Checkpoints are ~8.9 GB each and take ~19 s to write.
        # Expert-only removes backward work through the frozen towers, so steps get FASTER and the
        # dataloader has less slack -- re-check GPU utilisation before trusting the 2.1 s/it figure.
        #
        # Still untuned for this data: action_horizon=30 (1 s at 30 Hz; RoboLab used 15 for 1 s at
        # 15 Hz) and the LR schedule, which is inherited from the sim configs.
        name="pi05_piperx_teleop_expert_lora",
        # Absolute paths, tied to the g6e.xlarge training box (see the HF_LEROBOT_HOME note above).
        # compute_norm_stats.py has no path-override flag, so assets_base_dir has to live here rather
        # than being passed at launch. Change both if this config ever moves to another machine or
        # gets merged into the shared `trc` branch.
        assets_base_dir="/home/ubuntu/training/assets",
        checkpoint_base_dir="/home/ubuntu/training/runs",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=30,
            # LoRA on Gemma-2B only. The action expert variant is left plain so it trains fully,
            # and no lora is added to the SigLIP tower.
            paligemma_variant="gemma_2b_lora",
        ),
        data=LeRobotPiperXTeleopDataConfig(
            repo_id="teleop-data-hugo/piper_x_pick_cube_v1",
            n_arm_joints=6,
            action_dim=7,
            gripper_open_value=0.07,
            base_config=DataConfig(prompt_from_task=True),
            # No assets_dir/asset_id: Piper X is not in the pretraining mixture, so norm stats are
            # computed fresh by compute_norm_stats into assets/<config>/<repo_id>/.
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        # changan's expert configs rescaled to 10k steps: 400 warmup keeps his 4% warmup fraction
        # (200/5k), rather than inheriting his absolute 200 and ending up at 2%. Peak/floor unchanged.
        # Cosine anneal kept (both changan's configs and the openpi default anneal); this means the
        # 2k/4k/6k checkpoints sit at different points on the LR curve and are NOT directly
        # comparable to 10k -- read them as "still improving?", not as candidates.
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=400, peak_lr=2.5e-5, decay_steps=10_000, decay_lr=2.5e-6
        ),
        num_train_steps=10_000,
        batch_size=16,
        log_interval=25,
        # Checkpoint every 2k -> 5 checkpoints (2k/4k/6k/8k/10k = 1.7/3.5/5.2/6.9/8.7 epochs),
        # ~8.9 GB each, so eval can pick the best by hardware rollout rather than by loss.
        # keep_period must match save_interval or the manager prunes the ones you wanted.
        save_interval=2_000,
        keep_period=2_000,
        num_workers=4,
        # Expert-only: freeze the SigLIP tower (".*img.*") and Gemma-2B (".*llm.*" minus the action
        # expert ".*llm.*_1.*"), so only the ~430M action expert trains. Same filter as
        # pi05_piperx_rubiks_expert / pi05_piperx_fixedpose_expert.
        freeze_filter=nnx.Any(
            nnx.All(
                nnx_utils.PathRegex(".*llm.*"),
                nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*")),  # action expert trains fully
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),     # LoRA adapters train
            ),
            nnx_utils.PathRegex(".*img.*"),  # SigLIP frozen
        ),
        # EMA OFF. At 0.99 (changan's expert-config default) train.py:113 keeps a full shadow copy
        # of all 3.353B params, which OOM-killed the host at step 194/200: anon-rss 23.2 GB +
        # shmem 4.3 GB = 27.5 GB of this box's 30 GB, no swap. The LoRA smoke with ema_decay=None
        # completed the same 200 steps and wrote a checkpoint. Costs a modest final-quality bump;
        # the openpi LoRA configs disable it for the same reason.
        ema_decay=None,
    ),
    TrainConfig(
        # Recommended run for the current 10-episode / 1.4k-frame piperx_rubiks_cube_bowl set: full fine-tune from pi05_base
        # with a short schedule (3k steps ~ 70 epochs at batch 32), warmup and cosine decay compressed to the run length,
        # checkpoints every 500 steps so the best one can be picked by RoboLab rollouts rather than by loss.
        name="pi05_piperx_rubiks",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=15),
        data=LeRobotRoboLabDataConfig(
            repo_id="trc/robolab_piperx",
            n_arm_joints=6,
            action_dim=7,
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=200, peak_lr=2.5e-5, decay_steps=3_000, decay_lr=2.5e-6),
        num_train_steps=3_000,
        batch_size=32,
        log_interval=25,
        save_interval=500,
        keep_period=500,
        num_workers=4,
    ),
    TrainConfig(
        # Action-expert-only fine-tune: the whole PaliGemma VLM (SigLIP + Gemma 2B) stays frozen, only the 300M action
        # expert (Gemma params suffixed _1) and the action/state/time projections train. Same 3k-step small-data schedule.
        name="pi05_piperx_rubiks_expert",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=15),
        data=LeRobotRoboLabDataConfig(
            repo_id="trc/robolab_piperx",
            n_arm_joints=6,
            action_dim=7,
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=200, peak_lr=2.5e-5, decay_steps=3_000, decay_lr=2.5e-6),
        num_train_steps=3_000,
        batch_size=32,
        log_interval=25,
        save_interval=500,
        keep_period=500,
        num_workers=4,
        # Freeze: Gemma-2B params (".*llm.*" minus the action expert ".*llm.*_1.*") and the SigLIP tower (".*img.*").
        freeze_filter=nnx.Any(
            nnx.All(nnx_utils.PathRegex(".*llm.*"), nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*"))),
            nnx_utils.PathRegex(".*img.*"),
        ),
    ),
    TrainConfig(
        # Expert-only, 5k steps with a checkpoint every 1k for rollout-based model selection (L40S-sized: ~430M trainable).
        name="pi05_piperx_rubiks_expert_5k",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=15),
        data=LeRobotRoboLabDataConfig(
            repo_id="trc/robolab_piperx",
            n_arm_joints=6,
            action_dim=7,
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=200, peak_lr=2.5e-5, decay_steps=5_000, decay_lr=2.5e-6),
        num_train_steps=5_000,
        batch_size=32,
        log_interval=25,
        save_interval=1_000,
        keep_period=1_000,
        num_workers=4,
        freeze_filter=nnx.Any(
            nnx.All(nnx_utils.PathRegex(".*llm.*"), nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*"))),
            nnx_utils.PathRegex(".*img.*"),
        ),
    ),
    TrainConfig(
        # fixedpose_varpath set (200 eps, ~27.7k frames after dropping the zero-action warm-up): action-expert-only
        # fine-tune, 5k steps (~5.8 epochs at batch 32), checkpoint every 1k for interleaved RoboLab rollouts.
        # Prompt = RoboLab RubiksCubeTask default instruction ("Put the cube in the bowl"), written into the dataset.
        name="pi05_piperx_fixedpose_expert",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=15),
        data=LeRobotRoboLabDataConfig(
            repo_id="trc/robolab_piperx_fixedpose",
            n_arm_joints=6,
            action_dim=7,
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=200, peak_lr=2.5e-5, decay_steps=5_000, decay_lr=2.5e-6),
        num_train_steps=5_000,
        batch_size=32,
        log_interval=25,
        save_interval=1_000,
        keep_period=1_000,
        num_workers=4,
        freeze_filter=nnx.Any(
            nnx.All(nnx_utils.PathRegex(".*llm.*"), nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*"))),
            nnx_utils.PathRegex(".*img.*"),
        ),
    ),
    TrainConfig(
        # Same run for the 48 GB L40S: LoRA on both Gemma towers, batch 16, EMA off.
        name="pi05_piperx_rubiks_lora",
        model=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=15,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotRoboLabDataConfig(
            repo_id="trc/robolab_piperx",
            n_arm_joints=6,
            action_dim=7,
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=200, peak_lr=2.5e-5, decay_steps=3_000, decay_lr=2.5e-6),
        num_train_steps=3_000,
        batch_size=16,
        log_interval=25,
        save_interval=500,
        keep_period=500,
        num_workers=4,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=15,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        # DROID Franka in RoboLab (7 joints + gripper), e.g. self-distillation on successful pi05 rollouts.
        # Starts from the RoboLab sim checkpoint and reuses its DROID norm stats (same robot, same action space).
        name="pi05_robolab_franka",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
        ),
        data=LeRobotRoboLabDataConfig(
            repo_id="trc/robolab_franka",
            n_arm_joints=7,
            action_dim=8,
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets-simeval/pi05_droid_jointpos/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets-simeval/pi05_droid_jointpos/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    #
    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    # RoboArena & PolaRiS configs.
    *roboarena_config.get_roboarena_configs(),
    *polaris_config.get_polaris_configs(),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
