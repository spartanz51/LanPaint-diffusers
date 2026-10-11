"""
Qwen21Adapter — adapts QwenImage21Pipeline (Qwen-Image 2.1) to LanPaint's interface.

What differs from the Qwen-Image-Edit adapter:
  - Latents are unpatched (patch 1, 64 channels, 16x VAE): packing is a plain
    spatial flatten to (B, L, 64)
  - The reference image is PREPENDED to the sequence, and img_shapes lists it
    first; the transformer also takes an img_mask over the joint sequence
  - The VAE is RGBA (4 channels in and out): an opaque alpha is added on encode
    and dropped on decode
  - No CFG by default: guidance_scale <= 1 runs one transformer pass per call
  - Block-causal attention: the text + reference prefix depends neither on the
    timestep nor on the target tokens, so its keys/values are computed on the
    first call and reused by every later LanPaint call (QwenImage21KVCache)

The reference is VAE-encoded at the target size, so its tokens sit on the same
grid as the latent being inpainted. ``reference_image`` swaps the picture the
model is shown (e.g. the photo with the region to remove outlined) without
changing the pixels LanPaint keeps.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from diffusers import QwenImage21Pipeline
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift, retrieve_timesteps

from lanpaint_pipeline.model_adapter import ImageLatents, ModelAdapter, PromptBundle


def _retrieve_latents(encoder_output):
    """Extract latents from VAE encoder output (argmax mode for determinism)."""
    if hasattr(encoder_output, "latent_dist"):
        return encoder_output.latent_dist.mode()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access latents of provided encoder_output")


class Qwen21Adapter(ModelAdapter):
    """
    Concrete adapter for QwenImage21Pipeline.

    Usage::

        pipe = QwenImage21Pipeline.from_pretrained("Qwen/Qwen-Image-2.1", ...)
        adapter = Qwen21Adapter(pipe)
        adapter.sigmas = [1.0, 0.9375, 0.875, 0.75, 0.5, 0.25]  # optional, e.g. a turbo LoRA's schedule
    """

    # Edit-style model: the reference is in the sequence at every step.
    requires_ref_at_inference = True

    # Optional overrides, set by the caller before running the pipeline.
    # Declared on the class so run_lanpaint.py can check its flags before loading the model.
    MASK_MODES = ("majority", "any_edit", "nearest")
    sigmas: Optional[List[float]] = None
    reference_image: Optional[Image.Image] = None
    mask_mode: str = "majority"  # see mask_to_latent_space

    def __init__(self, pipe: QwenImage21Pipeline):
        super().__init__(pipe)
        # Populated by encode_prompt (deferred: the VL encoder needs the image)
        self._prompt_str: Optional[str] = None
        self._neg_prompt_str: str = ""
        # Populated by encode_and_prepare
        self._ref_pil: Optional[Image.Image] = None
        self._prompt_embeds = None
        self._prompt_embeds_mask = None
        self._img_mask = None
        self._neg_prompt_embeds = None
        self._neg_prompt_embeds_mask = None
        self._neg_img_mask = None
        self._ref_latents = None  # packed reference latents (B, L, 64)
        self._img_shapes = None
        self._cond_cache = None
        self._neg_cache = None
        self._cache_primed = {"cond": False, "uncond": False}
        self._latent_height: int = 0
        self._latent_width: int = 0
        self._pixel_height: int = 0
        self._pixel_width: int = 0
        self._conditioning_preloaded: bool = False

    # ---- helpers ----

    def _to_rgba_tensor(self, img: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) in [-1, 1] -> (B, 4, 1, H, W) with an opaque alpha."""
        alpha = torch.ones_like(img[:, :1])
        return torch.cat([img, alpha], dim=1).unsqueeze(2)

    def _vae_encode(self, img: torch.Tensor, device: torch.device) -> torch.Tensor:
        """(B, 3, H, W) in [-1, 1] -> normalized packed latents (B, L, 64), float32."""
        pipe = self.pipe
        vae_input = self._to_rgba_tensor(img).to(device=device, dtype=pipe.vae.dtype)
        with torch.no_grad():
            latents = _retrieve_latents(pipe.vae.encode(vae_input))
        z = pipe.vae.config.z_dim
        mean = torch.tensor(pipe.vae.config.latents_mean).view(1, z, 1, 1, 1).to(latents.device, latents.dtype)
        std = torch.tensor(pipe.vae.config.latents_std).view(1, z, 1, 1, 1).to(latents.device, latents.dtype)
        latents = (latents - mean) / std
        h, w = latents.shape[3], latents.shape[4]
        return pipe._pack_latents(latents, latents.shape[0], z, h, w).to(torch.float32)

    def _encode_text(self, prompt: str, device: torch.device):
        """VL-encode a prompt with the reference; the img_mask gets one slot per
        2x2 group of target latents, as QwenImage21Pipeline does."""
        embeds, mask, img_mask = self.pipe.encode_prompt(prompt=prompt, image=[self._ref_pil], device=device)
        n_target = self._latent_height * self._latent_width
        img_mask = torch.cat([img_mask, img_mask.new_ones(img_mask.shape[0], n_target // 4)], dim=1)
        return embeds, mask, img_mask

    def _new_caches(self):
        blocks = len(self.pipe.transformer.transformer_blocks)
        causal = getattr(self.pipe.transformer.config, "causal_condition", False)
        self._cond_cache = QwenImage21KVCache(blocks) if causal else None
        self._neg_cache = QwenImage21KVCache(blocks) if causal else None
        self._cache_primed = {"cond": False, "uncond": False}

    # ---- ModelAdapter implementation ----

    def set_conditioning(self, positive_conditioning, negative_conditioning=None):
        """
        Inject pre-encoded conditioning for the next run (called with prompt=None):
        dicts with prompt_embeds, prompt_embeds_mask and img_mask (the joint-sequence
        mask, target slots included). Every call replaces the previous request's
        conditioning; without a negative, the empty prompt is encoded if CFG needs one.
        """
        self._prompt_embeds = positive_conditioning["prompt_embeds"]
        self._prompt_embeds_mask = positive_conditioning.get("prompt_embeds_mask")
        self._img_mask = positive_conditioning["img_mask"]
        self._neg_prompt_str = ""
        if negative_conditioning is None:
            self._neg_prompt_embeds = self._neg_prompt_embeds_mask = self._neg_img_mask = None
        else:
            self._neg_prompt_embeds = negative_conditioning["prompt_embeds"]
            self._neg_prompt_embeds_mask = negative_conditioning.get("prompt_embeds_mask")
            self._neg_img_mask = negative_conditioning["img_mask"]
        self._conditioning_preloaded = True
        self._prompt_bundle = PromptBundle(data=dict(positive_conditioning))

    def encode_prompt(self, prompt: str, negative_prompt: str, device: torch.device) -> PromptBundle:
        """Store raw strings: the VL encoder needs the reference, available in encode_and_prepare().
        A text prompt replaces any conditioning injected for an earlier run."""
        self._conditioning_preloaded = False
        self._prompt_str = prompt
        self._neg_prompt_str = negative_prompt or ""
        self._prompt_bundle = PromptBundle(data={"prompt": prompt, "negative_prompt": self._neg_prompt_str})
        return self._prompt_bundle

    def encode_and_prepare(
        self,
        img_tensor: torch.Tensor,
        height: int,
        width: int,
        generator: torch.Generator,
        device: torch.device,
    ) -> ImageLatents:
        """
        VAE-encode the image (what LanPaint keeps) and the reference (what the model
        sees, the image itself unless reference_image is set), VL-encode the prompt.
        """
        pipe = self.pipe
        self._pixel_height, self._pixel_width = height, width
        sf = pipe.vae_scale_factor
        self._latent_height = 2 * (int(height) // (sf * 2))
        self._latent_width = 2 * (int(width) // (sf * 2))

        # The reference, at the target size so both share one token grid
        if self.reference_image is not None:
            ref = self.reference_image.convert("RGB").resize((width, height), Image.BICUBIC)
            ref_tensor = pipe.image_processor.preprocess(ref, height=height, width=width)
        else:
            img = ((img_tensor[0].detach().float().cpu() / 2.0) + 0.5).clamp(0.0, 1.0)
            ref = Image.fromarray((img.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
            ref_tensor = img_tensor
        self._ref_pil = ref

        y_latents = self._vae_encode(img_tensor, device)
        self._ref_latents = y_latents if self.reference_image is None else self._vae_encode(ref_tensor, device)

        self._img_shapes = [[
            (1, self._latent_height, self._latent_width),  # reference
            (1, self._latent_height, self._latent_width),  # target
        ]]

        if not self._conditioning_preloaded:
            self._prompt_embeds, self._prompt_embeds_mask, self._img_mask = self._encode_text(self._prompt_str, device)
            self._neg_prompt_embeds = None  # encoded on first use, only if CFG is on
        self._new_caches()

        self._image_latents = ImageLatents(
            latent=y_latents,
            meta={
                "latent_height": self._latent_height,
                "latent_width": self._latent_width,
                "pixel_height": self._pixel_height,
                "pixel_width": self._pixel_width,
            },
        )
        return self._image_latents

    def mask_to_latent_space(self, mask_pixel_keep: torch.Tensor) -> torch.Tensor:
        """
        Pixel mask (1, 1, H, W) -> packed latent mask (1, L, 1), 1 = keep.

        One latent covers 16x16 pixels. "majority" keeps a latent if most of
        its pixels are kept; "any_edit" edits it as soon as one pixel is edited,
        so no masked pixel is pinned to the photo (up to 15 px of growth);
        "nearest" samples one pixel per latent, as ComfyUI LanPaint does.
        """
        if self.mask_mode not in self.MASK_MODES:
            raise ValueError(f"mask_mode must be one of {self.MASK_MODES}, got {self.mask_mode!r}")
        m = mask_pixel_keep.to(torch.float32)
        size = (self._latent_height, self._latent_width)
        if self.mask_mode == "nearest":
            keep = torch.nn.functional.interpolate(m, size=size, mode="nearest-exact") > 0.5
        else:
            area = torch.nn.functional.interpolate(m, size=size, mode="area")
            keep = area > (1.0 - 1e-4) if self.mask_mode == "any_edit" else area > 0.5
        return keep.float().reshape(1, -1, 1)

    def prepare_timesteps(self, num_steps: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        QwenImage21Pipeline's schedule, shifted by mu = calculate_shift(target tokens).
        The sigmas are self.sigmas, else the checkpoint's sample_sigmas, else
        linspace(1, 1/N, N); the first two set the number of steps.
        """
        sigmas = self.sigmas if self.sigmas is not None else self.pipe.config.sample_sigmas
        if sigmas is not None:
            sigmas = np.array(sigmas, dtype=np.float64)
            num_steps = len(sigmas)
        else:
            sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
        cfg = self.pipe.scheduler.config
        mu = calculate_shift(
            self._latent_height * self._latent_width,
            cfg.get("base_image_seq_len", 256),
            cfg.get("max_image_seq_len", 4096),
            cfg.get("base_shift", 0.5),
            cfg.get("max_shift", 1.15),
        )
        timesteps, _ = retrieve_timesteps(self.pipe.scheduler, num_steps, device, sigmas=sigmas, mu=mu)
        flow_ts = self.pipe.scheduler.sigmas.to(device)[:-1]
        return timesteps[: len(flow_ts)], flow_ts

    def _velocity(self, x: torch.Tensor, flow_t: float, branch: str) -> torch.Tensor:
        model_dtype = self.dtype
        hidden = torch.cat([self._ref_latents.to(x.device, x.dtype), x], dim=1).to(model_dtype)
        timestep = torch.full((x.shape[0],), flow_t, device=x.device, dtype=model_dtype)
        if branch == "cond":
            embeds, emask, imask, cache = self._prompt_embeds, self._prompt_embeds_mask, self._img_mask, self._cond_cache
        else:
            embeds, emask, imask, cache = self._neg_prompt_embeds, self._neg_prompt_embeds_mask, self._neg_img_mask, self._neg_cache
        mode = None if cache is None else ("cached" if self._cache_primed[branch] else "extract")
        with self.pipe.transformer.cache_context(branch):
            v = self.pipe.transformer(
                hidden_states=hidden,
                timestep=timestep,
                encoder_hidden_states=embeds,
                encoder_hidden_states_mask=emask,
                img_shapes=self._img_shapes,
                img_mask=imask,
                kv_cache=cache,
                kv_cache_mode=mode,
                return_dict=False,
            )[0][:, -x.shape[1]:]
        if cache is not None:
            self._cache_primed[branch] = True
        return v.float()

    def predict_x0(
        self,
        x: torch.Tensor,
        flow_t: float,
        guidance_scale: float,
        cfg_big: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x0 = x_t - t * v. Without CFG (guidance_scale and cfg_big <= 1) one pass
        gives both outputs; otherwise v = uncond + scale * (cond - uncond), as in
        QwenImage21Pipeline (no norm rescaling).
        """
        v_cond = self._velocity(x, flow_t, "cond")
        # cfg_big below 1 is valid (ComfyUI "Prompt First" uses -0.5): only skip
        # the negative branch when neither scale needs it
        if guidance_scale <= 1.0 and cfg_big == 1.0:
            x0 = x.float() - flow_t * v_cond
            return x0, x0
        if self._neg_prompt_embeds is None:
            self._neg_prompt_embeds, self._neg_prompt_embeds_mask, self._neg_img_mask = self._encode_text(
                self._neg_prompt_str, x.device)
        v_uncond = self._velocity(x, flow_t, "uncond")
        v = v_cond if guidance_scale <= 1.0 else v_uncond + guidance_scale * (v_cond - v_uncond)
        v_big = v_uncond + cfg_big * (v_cond - v_uncond)
        return x.float() - flow_t * v, x.float() - flow_t * v_big

    def decode_latents(self, latents: torch.Tensor) -> Image.Image:
        """Packed latents -> denormalize -> RGBA VAE decode -> RGB PIL."""
        pipe = self.pipe
        with torch.no_grad():
            unpacked = pipe._unpack_latents(latents, self._pixel_height, self._pixel_width, pipe.vae_scale_factor)
            z = pipe.vae.config.z_dim
            mean = torch.tensor(pipe.vae.config.latents_mean).view(1, z, 1, 1, 1).to(unpacked.device, unpacked.dtype)
            std = torch.tensor(pipe.vae.config.latents_std).view(1, z, 1, 1, 1).to(unpacked.device, unpacked.dtype)
            unpacked = unpacked * std + mean
            img = pipe.vae.decode(unpacked.to(pipe.vae.dtype), return_dict=False)[0][:, :, 0]
            pil = pipe.image_processor.postprocess(img, output_type="pil")
            pil = pil[0] if isinstance(pil, list) else pil
            return pil.convert("RGB")
