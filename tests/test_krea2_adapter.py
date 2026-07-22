"""Unit tests for the Krea 2 adapter.

No model weights required: a fake pipe borrows the real Krea2Pipeline
pack/unpack/position-id helpers from diffusers, and a stub transformer returns
a known constant velocity so the x0 / CFG math can be checked exactly.

Run with: pytest tests/test_krea2_adapter.py
"""

from types import MethodType, SimpleNamespace

import pytest
import torch

from lanpaint_pipeline.adapters.krea2 import Krea2Adapter


class StubTransformer:
    """Returns a constant velocity; the value depends on whether the text
    embeds are the positive (non-zero) or negative (all-zero) ones, so both
    CFG branches are distinguishable."""

    dtype = torch.float32
    config = SimpleNamespace(in_channels=64)  # 16 latent channels * patch 2*2

    def __init__(self, v_pos=0.5, v_neg=0.25):
        self.v_pos = v_pos
        self.v_neg = v_neg
        self.calls = 0

    def __call__(self, hidden_states, encoder_hidden_states, timestep,
                 position_ids, encoder_attention_mask, return_dict=False):
        self.calls += 1
        val = self.v_pos if float(encoder_hidden_states.abs().sum()) > 0 else self.v_neg
        return (torch.full_like(hidden_states, val),)


def make_adapter(v_pos=0.5, v_neg=0.25):
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.krea2.pipeline_krea2 import Krea2Pipeline

    pipe = SimpleNamespace()
    pipe.patch_size = 2
    pipe.vae_scale_factor = 8
    pipe.config = SimpleNamespace(is_distilled=True)
    pipe.scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)
    pipe.transformer = StubTransformer(v_pos, v_neg)
    pipe._execution_device = torch.device("cpu")
    # real diffusers helpers, bound to the fake pipe (they only use patch_size
    # and vae_scale_factor)
    pipe._pack_latents = MethodType(Krea2Pipeline._pack_latents, pipe)
    pipe._unpack_latents = MethodType(Krea2Pipeline._unpack_latents, pipe)
    pipe.prepare_position_ids = Krea2Pipeline.prepare_position_ids

    adapter = Krea2Adapter(pipe)
    # state normally filled by encode_prompt / encode_and_prepare
    adapter._prompt_embeds = torch.ones(1, 8, 2, 4)
    adapter._prompt_mask = torch.ones(1, 8, dtype=torch.bool)
    adapter._neg_prompt_embeds = torch.zeros(1, 8, 2, 4)
    adapter._neg_prompt_mask = torch.ones(1, 8, dtype=torch.bool)
    adapter._latent_height = 8
    adapter._latent_width = 8
    adapter._pixel_height = 64
    adapter._pixel_width = 64
    return adapter


def test_registry_entry_resolves():
    from lanpaint_pipeline.registry import _import_class, get_model_spec

    spec = get_model_spec("krea2")
    assert spec.default_params["guidance_scale"] == 0.0
    assert _import_class(spec.adapter_cls_path) is Krea2Adapter


def test_predict_x0_no_cfg_is_x_minus_sigma_v():
    adapter = make_adapter(v_pos=0.5)
    x = torch.randn(1, 16, 8, 8)
    x0, x0_big = adapter.predict_x0(x, flow_t=0.8, guidance_scale=0.0, cfg_big=1.0)
    # Flux convention: x0 = x - sigma * v, and pack/unpack must round-trip
    assert torch.allclose(x0, x - 0.8 * 0.5, atol=1e-5)
    assert x0_big is x0
    assert adapter.pipe.transformer.calls == 1  # guidance 0 -> single pass


def test_predict_x0_cfg_formula_and_cfg_big():
    adapter = make_adapter(v_pos=0.5, v_neg=0.25)
    x = torch.randn(1, 16, 8, 8)
    x0, x0_big = adapter.predict_x0(x, flow_t=0.5, guidance_scale=2.0, cfg_big=3.0)
    # Krea formula: v = v_pos + g * (v_pos - v_neg)
    v = 0.5 + 2.0 * (0.5 - 0.25)
    v_big = 0.5 + 3.0 * (0.5 - 0.25)
    assert torch.allclose(x0, x - 0.5 * v, atol=1e-5)
    assert torch.allclose(x0_big, x - 0.5 * v_big, atol=1e-5)
    assert adapter.pipe.transformer.calls == 2  # pos + neg


def test_prepare_timesteps_distilled():
    adapter = make_adapter()
    timesteps, flow_ts = adapter.prepare_timesteps(8, torch.device("cpu"))
    assert len(timesteps) == len(flow_ts) == 8
    ft = flow_ts.tolist()
    assert all(0.0 < s <= 1.0 for s in ft)
    assert all(a > b for a, b in zip(ft, ft[1:]))  # strictly decreasing
    # scheduler timesteps are sigma * num_train_timesteps
    assert torch.allclose(timesteps.float(), flow_ts.float() * 1000, atol=1e-3)


def test_mask_to_latent_space_nearest():
    adapter = make_adapter()
    mask = torch.ones(1, 1, 64, 64)
    mask[:, :, :, :32] = 0.0  # left half is the edit region
    lat = adapter.mask_to_latent_space(mask)
    assert lat.shape == (1, 1, 8, 8)
    assert torch.all(lat[:, :, :, :4] == 0.0)
    assert torch.all(lat[:, :, :, 4:] == 1.0)


def test_flow_time_conversions_are_exact():
    """LanPaint converts flow-x to VP-x via x * (sqrt(abt) + sqrt(1-abt)).
    With abt = (1-t)^2 / ((1-t)^2 + t^2) this equals exact VP normalization
    (factor 1/sqrt((1-t)^2 + t^2)), so the Langevin dynamics run at the right
    time scale for flow models like Krea 2."""
    from lanpaint_pipeline.utils import flow_to_abt, flow_to_ve_sigma

    for t in (0.1, 0.35, 0.5, 0.72, 0.9):
        abt = flow_to_abt(t)
        n = ((1 - t) ** 2 + t ** 2) ** 0.5
        assert abs(abt ** 0.5 - (1 - t) / n) < 1e-4
        assert abs((1 - abt) ** 0.5 - t / n) < 1e-4
        assert abs((abt ** 0.5 + (1 - abt) ** 0.5) - 1.0 / n) < 1e-4
        assert abs(flow_to_ve_sigma(t) - t / (1 - t)) < 1e-6


def test_noise_scaling_endpoints():
    """Flow forward process: sigma=0 gives the clean latent, sigma=1 pure noise."""
    adapter = make_adapter()
    y = torch.randn(1, 16, 8, 8)
    noise = torch.randn_like(y)
    assert torch.allclose(adapter.noise_scaling(torch.tensor(0.0), noise, y), y)
    assert torch.allclose(adapter.noise_scaling(torch.tensor(1.0), noise, y), noise)


def test_set_conditioning_accepts_tuple_and_defaults_negative():
    adapter = make_adapter()
    embeds = torch.randn(1, 16, 2, 4)
    mask = torch.ones(1, 16, dtype=torch.bool)
    adapter.set_conditioning((embeds, mask))
    assert adapter._prompt_embeds is embeds
    assert adapter._neg_prompt_embeds.shape == embeds.shape
    assert float(adapter._neg_prompt_embeds.abs().sum()) == 0.0
    with pytest.raises(ValueError):
        adapter.set_conditioning({"wrong": 1})
