"""
Krea2Adapter — adapts Krea2Pipeline to LanPaint's interface.

Handles Krea 2-specific details (see diffusers Krea 2 pipeline):
  - Latents live in packed patch space inside the transformer (B, seq, C*p*p with p=2);
    the adapter keeps them spatial (B, C, H_lat, W_lat) for LanPaint masking and
    packs/unpacks around each transformer call
  - Prompt embeds are multi-layer Qwen3-VL hidden states (B, 512, num_layers, dim)
    with a boolean validity mask passed as encoder_attention_mask
  - Timestep for the transformer is the noise level sigma directly (t / 1000)
  - Flow matching, Flux convention: model output is velocity v, x_0 = x_t - sigma * v
  - Timesteps via retrieve_timesteps(sigmas=linspace, mu=1.15 fixed for the distilled
    Turbo checkpoint, resolution-aware calculate_shift for Raw)
  - VAE: Qwen-Image autoencoder, (latent - latents_mean) / latents_std (encode),
    inverse for decode; 5D tensors with a singleton frame dim
"""

from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

from diffusers import Krea2Pipeline
from diffusers.pipelines.krea2.pipeline_krea2 import calculate_shift, retrieve_timesteps

from lanpaint_pipeline.model_adapter import ImageLatents, ModelAdapter, PromptBundle


class Krea2Adapter(ModelAdapter):
    """
    Concrete adapter for Krea2Pipeline (Turbo and Raw checkpoints).

    Usage::

        pipe = Krea2Pipeline.from_pretrained("krea/Krea-2-Turbo", torch_dtype=torch.bfloat16)
        adapter = Krea2Adapter(pipe)

    The transformer can be swapped for a GGUF-quantized one before wrapping
    (``Krea2Transformer2DModel.from_single_file(..., quantization_config=GGUFQuantizationConfig(...))``),
    which brings the 24.5 GB bf16 checkpoint down to 5-13 GB.
    """

    def __init__(self, pipe: Krea2Pipeline):
        super().__init__(pipe)
        self._prompt_embeds: Optional[torch.Tensor] = None
        self._prompt_mask: Optional[torch.Tensor] = None
        self._neg_prompt_embeds: Optional[torch.Tensor] = None
        self._neg_prompt_mask: Optional[torch.Tensor] = None
        self._y_latent: Optional[torch.Tensor] = None
        self._position_ids: Optional[torch.Tensor] = None
        self._latent_height: int = 0
        self._latent_width: int = 0
        self._pixel_height: int = 0
        self._pixel_width: int = 0

    # ---- conditioning ----

    def set_conditioning(self, positive_conditioning, negative_conditioning=None):
        """
        Inject pre-encoded Krea 2 conditioning.

        Accepts a ``(prompt_embeds, prompt_embeds_mask)`` tuple/list or a dict with
        those keys, matching ``Krea2Pipeline.encode_prompt`` output:
        embeds (B, 512, num_layers, dim), mask (B, 512).
        """
        def _unwrap(cond):
            if isinstance(cond, dict) and "prompt_embeds" in cond and "prompt_embeds_mask" in cond:
                return cond["prompt_embeds"], cond["prompt_embeds_mask"]
            if isinstance(cond, (tuple, list)) and len(cond) == 2:
                return cond[0], cond[1]
            raise ValueError(
                "Krea2Adapter.set_conditioning expects (prompt_embeds, prompt_embeds_mask) "
                "or {'prompt_embeds': ..., 'prompt_embeds_mask': ...}"
            )

        self._prompt_embeds, self._prompt_mask = _unwrap(positive_conditioning)
        if negative_conditioning is not None:
            self._neg_prompt_embeds, self._neg_prompt_mask = _unwrap(negative_conditioning)
        else:
            self._neg_prompt_embeds = torch.zeros_like(self._prompt_embeds)
            self._neg_prompt_mask = self._prompt_mask
        self._prompt_bundle = PromptBundle(data={
            "prompt_embeds": self._prompt_embeds,
            "prompt_embeds_mask": self._prompt_mask,
            "neg_prompt_embeds": self._neg_prompt_embeds,
            "neg_prompt_embeds_mask": self._neg_prompt_mask,
        })

    def encode_prompt(self, prompt: str, negative_prompt: str, device: torch.device) -> PromptBundle:
        self._prompt_embeds, self._prompt_mask = self.pipe.encode_prompt(prompt=prompt, device=device)
        self._neg_prompt_embeds, self._neg_prompt_mask = self.pipe.encode_prompt(
            prompt=negative_prompt or "", device=device,
        )
        self._prompt_bundle = PromptBundle(data={
            "prompt_embeds": self._prompt_embeds,
            "prompt_embeds_mask": self._prompt_mask,
            "neg_prompt_embeds": self._neg_prompt_embeds,
            "neg_prompt_embeds_mask": self._neg_prompt_mask,
        })
        return self._prompt_bundle

    # ---- latents ----

    def _latents_mean_std(self):
        vae = self.pipe.vae
        mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1)
        return mean, std

    def encode_and_prepare(
        self,
        img_tensor: torch.Tensor,
        height: int,
        width: int,
        generator: torch.Generator,
        device: torch.device,
    ) -> ImageLatents:
        """VAE-encode image; Krea 2 uses (latent - latents_mean) / latents_std."""
        vae = self.pipe.vae
        p = self.pipe.patch_size

        self._pixel_height = p * (int(height) // (self.vae_scale_factor * p)) * self.vae_scale_factor
        self._pixel_width = p * (int(width) // (self.vae_scale_factor * p)) * self.vae_scale_factor
        self._latent_height = self._pixel_height // self.vae_scale_factor
        self._latent_width = self._pixel_width // self.vae_scale_factor

        with torch.no_grad():
            px = img_tensor.to(device=device, dtype=vae.dtype)
            px = px.unsqueeze(2)  # (B, C, 1, H, W) — Qwen-Image VAE is video-shaped
            latent = vae.encode(px).latent_dist.mode()  # (B, z, 1, lh, lw)
            mean, std = self._latents_mean_std()
            latent = (latent - mean.to(latent.device, latent.dtype)) / std.to(latent.device, latent.dtype)
            latent = latent[:, :, 0]  # (B, z, lh, lw)

        self._y_latent = latent.to(torch.float32)
        self._position_ids = None  # rebuilt lazily (depends on grid + text seq len)
        self._image_latents = ImageLatents(latent=self._y_latent, meta={})
        return self._image_latents

    def mask_to_latent_space(self, mask_pixel_keep: torch.Tensor) -> torch.Tensor:
        """Pixel mask (1, 1, H, W) → latent mask (1, 1, H_lat, W_lat)."""
        return torch.nn.functional.interpolate(
            mask_pixel_keep,
            size=(self._latent_height, self._latent_width),
            mode="nearest",
        ).to(mask_pixel_keep.device, torch.float32)

    # ---- timesteps ----

    def prepare_timesteps(
        self, num_steps: int, device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Krea 2: sigmas = linspace(1, 1/N) with mu = 1.15 (Turbo, distilled) or the
        resolution-aware calculate_shift (Raw). flow_ts = sigma (1 = noisy, 0 = clean)."""
        p = self.pipe.patch_size
        image_seq_len = (self._latent_height // p) * (self._latent_width // p)

        if self.pipe.config.is_distilled:
            mu = 1.15
        else:
            mu = calculate_shift(
                image_seq_len,
                self.pipe.scheduler.config.get("base_image_seq_len", 256),
                self.pipe.scheduler.config.get("max_image_seq_len", 6400),
                self.pipe.scheduler.config.get("base_shift", 0.5),
                self.pipe.scheduler.config.get("max_shift", 1.15),
            )

        sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
        timesteps, _ = retrieve_timesteps(
            self.pipe.scheduler,
            num_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        timesteps = timesteps.to(device)

        # flow_ts = scheduler sigmas (sigma: 1 = noisy, 0 = clean). sigmas has a
        # trailing 0.0 for the final jump; take [:-1] to match the loop length.
        flow_ts = self.pipe.scheduler.sigmas.to(device)[:-1]
        timesteps = timesteps[: len(flow_ts)]
        return timesteps, flow_ts

    # ---- prediction ----

    def _get_position_ids(self, device: torch.device) -> torch.Tensor:
        if self._position_ids is None:
            p = self.pipe.patch_size
            self._position_ids = self.pipe.prepare_position_ids(
                self._prompt_embeds.shape[1],
                self._latent_height // p,
                self._latent_width // p,
                device,
            )
        return self._position_ids

    def _velocity(self, packed: torch.Tensor, timestep: torch.Tensor,
                  embeds: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.pipe.transformer(
            hidden_states=packed,
            encoder_hidden_states=embeds.to(self.dtype),
            timestep=timestep,
            position_ids=self._get_position_ids(packed.device),
            encoder_attention_mask=mask,
            return_dict=False,
        )[0]

    def predict_x0(
        self,
        x: torch.Tensor,
        flow_t: float,
        guidance_scale: float,
        cfg_big: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Krea 2 transformer works on packed patch tokens; pack the spatial latent,
        predict velocity, unpack. Flux convention: x_0 = x_t - sigma * v.

        CFG uses the Krea 2 formula ``v = v_pos + g * (v_pos - v_neg)``, active
        when ``guidance_scale > 0`` (matching ``do_classifier_free_guidance`` in
        the official pipeline — the Turbo checkpoint runs guidance-free at 0.0).
        """
        model_dtype = self.dtype
        batch_size = x.shape[0]

        packed = self.pipe._pack_latents(
            x.to(model_dtype), batch_size, x.shape[1], self._latent_height, self._latent_width,
        )
        timestep = torch.full((batch_size,), float(flow_t), device=x.device, dtype=model_dtype)

        v_pos = self._velocity(packed, timestep, self._prompt_embeds, self._prompt_mask)

        if guidance_scale > 0:
            v_neg = self._velocity(packed, timestep, self._neg_prompt_embeds, self._neg_prompt_mask)
            v = v_pos + guidance_scale * (v_pos - v_neg)
            v_big = v if cfg_big == guidance_scale else v_pos + cfg_big * (v_pos - v_neg)
        else:
            v = v_big = v_pos

        def _unpack(t: torch.Tensor) -> torch.Tensor:
            out = self.pipe._unpack_latents(t, self._pixel_height, self._pixel_width)
            return out[:, :, 0].float()  # (B, z, lh, lw)

        x0 = x.float() - flow_t * _unpack(v)
        x0_big = x0 if v_big is v else x.float() - flow_t * _unpack(v_big)
        return x0, x0_big

    # ---- decode ----

    def decode_latents(self, latents: torch.Tensor) -> Image.Image:
        """Krea 2: latents * latents_std + latents_mean → VAE decode → postprocess."""
        vae = self.pipe.vae
        with torch.no_grad():
            mean, std = self._latents_mean_std()
            lat = latents.unsqueeze(2).to(vae.dtype)  # (B, z, 1, lh, lw)
            lat = lat * std.to(lat.device, lat.dtype) + mean.to(lat.device, lat.dtype)
            image = vae.decode(lat, return_dict=False)[0][:, :, 0]
            out = self.pipe.image_processor.postprocess(image, output_type="pil")
            return out[0] if isinstance(out, list) else out
