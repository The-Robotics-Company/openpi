import dataclasses
import logging
import os
import pathlib
from typing import Any

import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms


def _load_prompt_tokens(prompt_tokens: pathlib.Path | str) -> tuple[np.ndarray, tuple[str, ...]]:
    """Read a trained visual-prompt-token file.

    The file is self-describing -- it carries the camera names alongside the matrix -- so sweeping
    the token count K needs no new train config, and serving a set against the wrong cameras is a
    load-time error rather than a silent mis-assignment.
    """
    path = download.maybe_download(str(prompt_tokens))
    with np.load(path) as data:
        missing = {"tokens", "cameras"} - set(data)
        if missing:
            raise ValueError(f"{path} is missing {sorted(missing)} (found {sorted(data)})")
        tokens = np.asarray(data["tokens"], dtype=np.float32)
        cameras = tuple(str(c) for c in data["cameras"])
    if tokens.ndim != 3:
        raise ValueError(f"prompt tokens must be (sets, tokens, width), got shape {tokens.shape}")
    if tokens.shape[0] != len(cameras):
        raise ValueError(f"{path} has {tokens.shape[0]} token sets but names {len(cameras)} cameras: {cameras}")
    logging.info("Loaded prompt tokens %s for cameras %s from %s", tokens.shape, cameras, path)
    return tokens, cameras


def _with_prompt_tokens(params: Any, tokens: np.ndarray) -> Any:
    """Splice the token matrix into a restored params tree.

    The tokens are trained against the frozen encoder, separately from the policy, and amount to a
    few hundred KB -- so they ship as their own file rather than as another ~8.8 GB checkpoint per
    configuration. Splicing them here is also what makes the load possible at all: `load` narrows
    the tree to what the model declares and then demands an exact match, so it drops extra params
    but never fills in missing ones.

    Only the dicts along the modified path are copied; the arrays themselves are shared.
    """
    params = dict(params)
    params["PaliGemma"] = dict(params["PaliGemma"])
    img = dict(params["PaliGemma"]["img"])
    if "prompt_tokens" in img:
        raise ValueError("checkpoint already carries prompt_tokens; refusing to overwrite them")
    img["prompt_tokens"] = tokens
    params["PaliGemma"]["img"] = img
    return params


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
    prompt_tokens: pathlib.Path | str | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        prompt_tokens: Optional path to an npz of trained visual prompt tokens, holding "tokens"
            shaped (sets, tokens, width) and "cameras" naming the camera each set belongs to. The
            model config is reconfigured from the file, so the checkpoint and the train config need
            know nothing about the tokens.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    # The token file, not the train config, decides K and the camera order: the sweep produces one
    # file per configuration and the config that trained the policy knows nothing about any of them.
    tokens = None
    if prompt_tokens is not None:
        tokens, cameras = _load_prompt_tokens(prompt_tokens)
        train_config = dataclasses.replace(
            train_config,
            model=dataclasses.replace(
                train_config.model, num_prompt_tokens=tokens.shape[1], prompt_token_cameras=cameras
            ),
        )

    # Check if this is a PyTorch model by looking for model.safetensors
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)

    logging.info("Loading model...")
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        params = _model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)
        if tokens is not None:
            params = _with_prompt_tokens(params, tokens)
        model = train_config.model.load(params)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    # Determine the device to use for PyTorch models
    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )
