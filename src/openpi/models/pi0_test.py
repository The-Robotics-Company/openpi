import flax.nnx as nnx
import jax
import pytest

import openpi.models.pi0_config as _pi0_config

# SigLIP So400m/14 at 224px: the prompt tokens live at the patch-embedding width, not the
# 2048-wide projection the tower hands to Gemma.
SIGLIP_WIDTH = 1152


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)


def _vision_params(config: _pi0_config.Pi0Config) -> dict:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))
    return nnx.state(abstract_model, nnx.Param).to_pure_dict()["PaliGemma"]["img"]


def test_prompt_tokens_absent_by_default():
    """Every existing checkpoint has to keep loading, so the default graph must be unchanged."""
    assert "prompt_tokens" not in _vision_params(_pi0_config.Pi0Config())


def test_prompt_tokens_allocate_one_set_per_camera():
    config = _pi0_config.Pi0Config(
        pi05=True,
        num_prompt_tokens=16,
        prompt_token_cameras=("base_0_rgb", "left_wrist_0_rgb"),
    )
    assert _vision_params(config)["prompt_tokens"].shape == (2, 16, SIGLIP_WIDTH)


def test_prompt_tokens_and_cameras_must_agree():
    with pytest.raises(ValueError, match="must be set together"):
        _pi0_config.Pi0Config(num_prompt_tokens=16)
    with pytest.raises(ValueError, match="must be set together"):
        _pi0_config.Pi0Config(prompt_token_cameras=("base_0_rgb",))
    with pytest.raises(ValueError, match="must be unique"):
        _pi0_config.Pi0Config(num_prompt_tokens=16, prompt_token_cameras=("base_0_rgb", "base_0_rgb"))


def _prefix_shape(config: _pi0_config.Pi0Config):
    """Trace embed_prefix without allocating weights, and report what Gemma would receive."""

    def run():
        model = config.create(jax.random.key(0))
        tokens, _, _ = model.embed_prefix(config.fake_obs())
        return tokens

    return nnx.eval_shape(run).shape


def test_tokens_do_not_change_what_gemma_receives():
    """The prefix length is the load-bearing invariant: nothing downstream is being retrained.

    This also exercises the nnx bridge, which the pure-Flax tests in siglip_test.py bypass --
    `prompt_index` has to survive ToNNX's kwarg forwarding to reach the tower at all.
    """
    plain = _pi0_config.Pi0Config(pi05=True)
    prompted = _pi0_config.Pi0Config(
        pi05=True,
        num_prompt_tokens=64,
        prompt_token_cameras=("base_0_rgb", "left_wrist_0_rgb"),
    )
    assert _prefix_shape(prompted) == _prefix_shape(plain)
