"""Learnable visual prompt tokens in the SigLIP tower.

The contract these tests pin down is the one the whole experiment rests on: adding the token
capability must not perturb the tower unless a call actually asks for tokens, and using tokens must
not change the length or layout of what the tower hands downstream. Everything the policy sees
outside the patch features has to be untouched, because nothing else in the policy is being
retrained.
"""

import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.siglip as _siglip

WIDTH = 32
PATCH = 16
IMAGE_SHAPE = (2, 64, 64, 3)  # -> 4x4 = 16 patches
NUM_PATCHES = 16


def _tower(pool_type="none", **kwargs):
    """A tower small enough to run on CPU; the token logic is independent of the variant."""
    return _siglip._Module(  # noqa: SLF001
        num_classes=None,
        patch_size=(PATCH, PATCH),
        width=WIDTH,
        depth=2,
        num_heads=2,
        mlp_dim=64,
        pool_type=pool_type,
        **kwargs,
    )


def _image():
    return jnp.asarray(np.random.default_rng(0).uniform(-1, 1, IMAGE_SHAPE), jnp.float32)


def _shared_params(plain, prompted, image):
    """Init both towers and give the prompted one the plain one's weights verbatim.

    Re-initialising is not enough: the two must differ in the prompt tokens and in nothing else,
    or an equality check proves nothing about the insertion.
    """
    plain_params = plain.init(jax.random.key(0), image)["params"]
    prompted_params = prompted.init(jax.random.key(1), image)["params"]
    merged = {**prompted_params, **plain_params}
    assert "prompt_tokens" in merged
    assert set(merged) - set(plain_params) == {"prompt_tokens"}
    return plain_params, merged


def test_zero_tokens_creates_no_parameter():
    image = _image()
    params = _tower().init(jax.random.key(0), image)["params"]
    assert "prompt_tokens" not in params


def test_parameter_shape_is_sets_by_tokens_by_width():
    image = _image()
    params = _tower(num_prompt_tokens=4, num_prompt_sets=3).init(jax.random.key(0), image)["params"]
    assert params["prompt_tokens"].shape == (3, 4, WIDTH)


def test_unused_tokens_leave_the_tower_bit_identical():
    """A camera given no token set must encode exactly as it did before the feature existed."""
    image = _image()
    plain, prompted = _tower(), _tower(num_prompt_tokens=4, num_prompt_sets=2)
    plain_params, merged = _shared_params(plain, prompted, image)

    expected, _ = plain.apply({"params": plain_params}, image)
    actual, _ = prompted.apply({"params": merged}, image, prompt_index=None)

    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_tokens_preserve_sequence_length_but_change_features():
    """The tokens must modulate the patches and then vanish from the output."""
    image = _image()
    plain, prompted = _tower(), _tower(num_prompt_tokens=4, num_prompt_sets=2)
    plain_params, merged = _shared_params(plain, prompted, image)

    expected, _ = plain.apply({"params": plain_params}, image)
    actual, _ = prompted.apply({"params": merged}, image, prompt_index=0)

    assert expected.shape == (IMAGE_SHAPE[0], NUM_PATCHES, WIDTH)
    assert actual.shape == expected.shape, "prompt token outputs leaked downstream"
    assert not np.allclose(np.asarray(actual), np.asarray(expected)), "tokens had no effect at all"


def test_each_set_is_independent():
    """Per-camera sets only mean something if selecting a different one does something different."""
    image = _image()
    prompted = _tower(num_prompt_tokens=4, num_prompt_sets=2)
    params = prompted.init(jax.random.key(0), image)["params"]

    first, _ = prompted.apply({"params": params}, image, prompt_index=0)
    second, _ = prompted.apply({"params": params}, image, prompt_index=1)

    assert not np.allclose(np.asarray(first), np.asarray(second))


def test_tokens_do_not_disturb_the_cls_token():
    """With pool_type='tok' the tokens sit in front of cls, and the slice has to restore that."""
    image = _image()
    plain = _tower(pool_type="tok")
    prompted = _tower(pool_type="tok", num_prompt_tokens=4, num_prompt_sets=1)
    plain_params, merged = _shared_params(plain, prompted, image)

    expected, _ = plain.apply({"params": plain_params}, image)
    unused, _ = prompted.apply({"params": merged}, image, prompt_index=None)
    used, _ = prompted.apply({"params": merged}, image, prompt_index=0)

    np.testing.assert_array_equal(np.asarray(unused), np.asarray(expected))
    assert used.shape == expected.shape
