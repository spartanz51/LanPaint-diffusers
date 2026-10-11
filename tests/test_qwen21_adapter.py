"""Unit tests for the Qwen-Image 2.1 adapter.

No model weights required: a fake pipe borrows the real QwenImage21Pipeline
pack/unpack helpers from diffusers, and a stub transformer returns a known
velocity per token so the slicing, x0 and CFG math can be checked exactly.

Run with: pytest tests/test_qwen21_adapter.py
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

pytest.importorskip("diffusers.pipelines.qwenimage21.pipeline_qwenimage21")
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline  # noqa: E402
from diffusers.image_processor import VaeImageProcessor  # noqa: E402

from lanpaint_pipeline.adapters.qwen21 import Qwen21Adapter  # noqa: E402
from lanpaint_pipeline.registry import check_adapter_options  # noqa: E402


class StubTransformer:
    """Velocity = token index along the joint sequence (so the target slice is
    recognizable), plus an offset that tells the cond branch (positive embeds)
    from the uncond one (all-zero embeds). Records every kv_cache_mode."""

    dtype = torch.float32
    config = SimpleNamespace(causal_condition=True)

    def __init__(self, blocks=2):
        self.transformer_blocks = [None] * blocks
        self.modes = []
        self.lengths = []

    def cache_context(self, name):
        from contextlib import nullcontext
        return nullcontext()

    def __call__(self, hidden_states, timestep, encoder_hidden_states, encoder_hidden_states_mask,
                 img_shapes, img_mask, kv_cache, kv_cache_mode, return_dict=False):
        self.modes.append(kv_cache_mode)
        self.lengths.append(hidden_states.shape[1])
        b, n, c = hidden_states.shape
        idx = torch.arange(n, dtype=torch.float32).view(1, n, 1).expand(b, n, c)
        offset = 0.5 if float(encoder_hidden_states.abs().sum()) > 0 else 0.25
        return (idx * 0.01 + offset,)


def make_adapter():
    pipe = SimpleNamespace()
    pipe.vae_scale_factor = 16
    pipe.scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True, base_image_seq_len=256,
                                                     max_image_seq_len=8192, base_shift=0.5, max_shift=0.9)
    pipe.transformer = StubTransformer()
    pipe._execution_device = torch.device("cpu")
    pipe._pack_latents = QwenImage21Pipeline._pack_latents
    pipe._unpack_latents = QwenImage21Pipeline._unpack_latents

    adapter = Qwen21Adapter(pipe)
    # state normally filled by encode_and_prepare (64x64 px -> 4x4 latents)
    adapter._latent_height = adapter._latent_width = 4
    adapter._pixel_height = adapter._pixel_width = 64
    adapter._ref_latents = torch.zeros(1, 16, 64)
    adapter._img_shapes = [[(1, 4, 4), (1, 4, 4)]]
    adapter._prompt_embeds = torch.ones(1, 6, 8)
    adapter._prompt_embeds_mask = None
    adapter._img_mask = torch.ones(1, 10, dtype=torch.bool)
    adapter._neg_prompt_embeds = torch.zeros(1, 6, 8)
    adapter._neg_prompt_embeds_mask = None
    adapter._neg_img_mask = torch.ones(1, 10, dtype=torch.bool)
    adapter._new_caches()
    return adapter


class StubVAE:
    """16x RGBA VAE stand-in. Encode: every channel of a latent holds the mean
    RGB of its 16x16 tile. Decode: white RGB under a fully transparent alpha."""

    dtype = torch.float32
    config = SimpleNamespace(z_dim=64, latents_mean=[0.0] * 64, latents_std=[1.0] * 64)

    def encode(self, x):
        assert x.shape[1] == 4 and bool((x[:, 3] == 1).all())  # RGB plus an opaque alpha
        tiles = torch.nn.functional.avg_pool2d(x[:, :3, 0].mean(1, keepdim=True), 16)
        return SimpleNamespace(latents=tiles.unsqueeze(2).expand(-1, 64, -1, -1, -1))

    def decode(self, z, return_dict=False):
        b, _, f, h, w = z.shape
        out = torch.ones(b, 4, f, h * 16, w * 16)
        out[:, 3] = -1.0
        return (out,)


def make_encoding_adapter():
    """make_adapter() plus a VAE, an image processor and a VL encoder that
    records the images it is shown."""
    adapter = make_adapter()
    pipe = adapter.pipe
    pipe.vae = StubVAE()
    pipe.image_processor = VaeImageProcessor(vae_scale_factor=16, vae_latent_channels=64)
    pipe.shown = []

    def encode_prompt(prompt, image, device):
        pipe.shown.append(image)
        return torch.ones(1, 6, 8), None, torch.ones(1, 10, dtype=torch.bool)

    pipe.encode_prompt = encode_prompt
    adapter.encode_prompt("remove the chair", "", torch.device("cpu"))
    return adapter


def test_registry_entry_resolves():
    from lanpaint_pipeline.registry import _import_class, get_model_spec

    spec = get_model_spec("qwen-image-2.1")
    assert spec.requires_ref_at_inference
    assert _import_class(spec.adapter_cls_path) is Qwen21Adapter


def test_reference_is_prepended_and_only_the_target_slice_comes_back():
    adapter = make_adapter()
    x = torch.zeros(1, 16, 64)
    x0, _ = adapter.predict_x0(x, flow_t=0.5, guidance_scale=1.0, cfg_big=1.0)
    assert adapter.pipe.transformer.lengths == [32]  # 16 reference + 16 target tokens
    v_expected = torch.arange(16, 32, dtype=torch.float32).view(1, 16, 1) * 0.01 + 0.5
    assert torch.allclose(x0, -0.5 * v_expected.expand(1, 16, 64))


def test_no_cfg_is_one_pass_and_x0_big_equals_x0():
    adapter = make_adapter()
    x = torch.randn(1, 16, 64)
    x0, x0_big = adapter.predict_x0(x, flow_t=0.8, guidance_scale=1.0, cfg_big=1.0)
    assert len(adapter.pipe.transformer.modes) == 1
    assert torch.equal(x0, x0_big)


def test_cfg_formula_and_cfg_big():
    adapter = make_adapter()
    x = torch.zeros(1, 16, 64)
    x0, x0_big = adapter.predict_x0(x, flow_t=1.0, guidance_scale=4.0, cfg_big=2.0)
    # cond and uncond differ by 0.25 on every token: v = v_uncond + s * 0.25
    tok = torch.arange(16, 32, dtype=torch.float32).view(1, 16, 1) * 0.01
    assert torch.allclose(x0, -(tok + 0.25 + 4.0 * 0.25).expand(1, 16, 64))
    assert torch.allclose(x0_big, -(tok + 0.25 + 2.0 * 0.25).expand(1, 16, 64))


def test_kv_cache_is_extracted_once_per_branch_then_reused():
    adapter = make_adapter()
    x = torch.zeros(1, 16, 64)
    for t in (0.9, 0.9, 0.5):
        adapter.predict_x0(x, flow_t=t, guidance_scale=4.0, cfg_big=1.0)
    assert adapter.pipe.transformer.modes == ["extract", "extract", "cached", "cached", "cached", "cached"]


def test_prompt_first_cfg_big_below_one_is_not_clamped():
    adapter = make_adapter()
    x = torch.zeros(1, 16, 64)
    x0, x0_big = adapter.predict_x0(x, flow_t=1.0, guidance_scale=1.0, cfg_big=-0.5)
    tok = torch.arange(16, 32, dtype=torch.float32).view(1, 16, 1) * 0.01
    assert len(adapter.pipe.transformer.modes) == 2  # the negative branch ran
    assert torch.allclose(x0, -(tok + 0.5).expand(1, 16, 64))
    assert torch.allclose(x0_big, -(tok + 0.25 - 0.5 * 0.25).expand(1, 16, 64))


def test_any_edit_mask_mode_pins_no_masked_pixel():
    keep = torch.ones(1, 1, 64, 64)
    keep[..., 0, 0] = 0              # one edited pixel in latent (0, 0)
    keep[..., 16:24, 0:16] = 0       # latent (1, 0) half edited
    adapter = make_adapter()
    majority = adapter.mask_to_latent_space(keep).view(4, 4)
    adapter.mask_mode = "any_edit"
    any_edit = adapter.mask_to_latent_space(keep).view(4, 4)
    assert majority[0, 0] == 1 and majority[1, 0] == 0
    assert any_edit[0, 0] == 0 and any_edit[1, 0] == 0 and any_edit.sum() == 14


def test_mask_to_latent_space_is_a_16x_majority():
    adapter = make_adapter()
    keep = torch.ones(1, 1, 64, 64)
    keep[..., :16, :16] = 0          # latent (0, 0) fully edited
    keep[..., :16, 16:24] = 0        # latent (0, 1) half edited -> not kept
    keep[..., 16:20, 16:32] = 0      # latent (1, 1) a quarter edited -> kept
    m = adapter.mask_to_latent_space(keep).view(4, 4)
    assert m.shape == (4, 4)
    assert m[0, 0] == 0 and m[0, 1] == 0 and m[1, 1] == 1 and m[3, 3] == 1


def test_pack_unpack_round_trip():
    adapter = make_adapter()
    lat = torch.randn(1, 64, 1, 4, 4)
    packed = adapter.pipe._pack_latents(lat, 1, 64, 4, 4)
    assert packed.shape == (1, 16, 64)
    back = adapter.pipe._unpack_latents(packed, 64, 64, 16)
    assert torch.equal(back, lat)


def test_custom_sigmas_set_the_schedule_length():
    adapter = make_adapter()
    adapter.sigmas = [1.0, 0.9375, 0.875, 0.75, 0.5, 0.25]
    timesteps, flow_ts = adapter.prepare_timesteps(40, torch.device("cpu"))
    assert len(timesteps) == len(flow_ts) == 6
    assert np.all(np.diff(flow_ts.numpy()) < 0)


def test_reference_image_changes_what_the_model_sees_not_what_is_kept():
    gray = torch.zeros(1, 3, 64, 64)  # mid-gray photo, in [-1, 1]
    adapter = make_encoding_adapter()
    kept = adapter.encode_and_prepare(gray, 64, 64, None, torch.device("cpu")).latent
    assert adapter._ref_latents is kept  # by default the model sees the photo itself

    adapter = make_encoding_adapter()
    adapter.reference_image = Image.new("RGB", (32, 32), "white")
    kept = adapter.encode_and_prepare(gray, 64, 64, None, torch.device("cpu")).latent
    assert torch.allclose(kept, torch.zeros_like(kept))  # LanPaint keeps the photo
    assert torch.allclose(adapter._ref_latents, torch.ones_like(kept))  # the model sees the reference
    shown = adapter.pipe.shown[-1][0]
    assert shown.size == (64, 64) and shown.getpixel((0, 0)) == (255, 255, 255)


def test_img_mask_gets_one_slot_per_2x2_group_of_target_latents():
    adapter = make_encoding_adapter()
    adapter.encode_and_prepare(torch.zeros(1, 3, 128, 64), 128, 64, None, torch.device("cpu"))
    # 8x4 target latents = 32 tokens -> 8 slots after the 10 the VL encoder returned
    assert adapter._img_shapes == [[(1, 8, 4), (1, 8, 4)]]
    assert adapter._img_mask.shape == (1, 18)


def test_decode_drops_the_alpha_without_compositing_it():
    adapter = make_encoding_adapter()
    out = adapter.decode_latents(torch.zeros(1, 16, 64))
    assert out.mode == "RGB" and out.size == (64, 64)
    assert np.asarray(out).min() == 255  # transparent white stays white


def test_unknown_mask_mode_is_refused():
    adapter = make_adapter()
    adapter.mask_mode = "dilate"
    with pytest.raises(ValueError, match="mask_mode"):
        adapter.mask_to_latent_space(torch.ones(1, 1, 64, 64))


def test_adapter_options_are_checked_against_the_model():
    check_adapter_options("qwen-image-2.1", ["reference_image", "mask_mode", "sigmas"])
    with pytest.raises(ValueError, match="mask_mode"):
        check_adapter_options("qwen", ["mask_mode"])
