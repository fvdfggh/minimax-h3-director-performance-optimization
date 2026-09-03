"""Fast Video VAE node for MiniMax H3.

This is a ``VAE``-typed node: it takes a MiniMax H3 video VAE (the same kind you
would load with the core ``VAELoader``) plus a ``batch`` parameter, and returns a
**wrapped VAE** whose decode path uses the fast tiled decoder
(:func:`comfy.ldm.minimax.vae.h3_fast_decode`).

Downstream, you use the output exactly like a normal video VAE — plug it into a
standard ``VAEDecode`` / ``VAEEncode`` / any node that accepts a ``VAE`` input.
The only difference from a plain video VAE is the extra ``batch`` parameter that
controls the per-tile batch size of the fast decoder.
"""

from __future__ import annotations

import logging
import time

import comfy.ldm.minimax.vae
import comfy.sd


class _FastMiniMaxH3VideoVAE(comfy.sd.VAE):
    """A ``VAE`` subclass that routes decode through ``h3_fast_decode``.

    Everything except ``.decode()`` is inherited / delegated from the wrapped
    VAE. ``h3_fast_decode`` already performs the input normalization and output
    de-normalization internally, so we must NOT call ``self.process_output`` on
    its result (that would double-process).
    """

    def __init__(self, vae: comfy.sd.VAE, tile_batch_size: int = 4):
        # Copy the wrapped VAE wholesale, then override decode behavior.
        self.__dict__.update(vae.__dict__)
        self._inner_vae = vae
        self._tile_batch_size = int(tile_batch_size)
        if not isinstance(vae.first_stage_model, comfy.ldm.minimax.vae.MiniMaxH3VideoVAE):
            raise ValueError("Fast Video VAE requires the MiniMax H3 video VAE")

    def decode(self, samples_in, vae_options=None):
        self.throw_exception_if_invalid()

        if isinstance(samples_in, dict):
            samples_in = samples_in["samples"]

        # h3_fast_decode handles device / dtype / normalization / process_output
        # internally; it expects a (possibly nested) latent tensor.
        latent = samples_in
        if getattr(latent, "is_nested", False):
            latent = latent.unbind()[0]

        with comfy.model_management.cuda_device_context(self.device):
            comfy.model_management.load_models_gpu(
                [self.patcher],
                memory_required=self.memory_used_decode(latent.shape, self.vae_dtype),
                force_full_load=self.disable_offload,
            )
            start_time = time.perf_counter()
            images = comfy.ldm.minimax.vae.h3_fast_decode(self, latent, self._tile_batch_size)
            logging.info(
                "Fast Video VAE decode (batch %d) finished in %.2fs",
                self._tile_batch_size,
                time.perf_counter() - start_time,
            )

        # h3_fast_decode returns images in [B,T,H,W,C] (5D) or [B,H,W,C] (4D);
        # flatten temporal batches to match comfy VAE output convention.
        if images.ndim == 5:  # Combine batches
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return images.to(self.output_device).movedim(-1, 1)

    def decode_tiled(self, samples, tile_x=None, tile_y=None, overlap=None, tile_t=None, overlap_t=None):
        # Fall back to the standard (non-fast) tiled decode so downstream
        # VAEDecodeTiled-style usage keeps working.
        return self._inner_vae.decode_tiled(samples, tile_x=tile_x, tile_y=tile_y,
                                            overlap=overlap, tile_t=tile_t, overlap_t=overlap_t)

    def encode(self, pixel_samples):
        return self._inner_vae.encode(pixel_samples)

    def encode_tiled(self, pixel_samples, tile_x=None, tile_y=None, overlap=None, tile_t=None, overlap_t=None):
        return self._inner_vae.encode_tiled(pixel_samples, tile_x=tile_x, tile_y=tile_y,
                                            overlap=overlap, tile_t=tile_t, overlap_t=overlap_t)

    def __getattr__(self, name):
        # Delegate anything not explicitly overridden to the wrapped VAE.
        return getattr(self._inner_vae, name)


class MiniMaxH3FastVideoVAE:
    """Fast Video VAE loader/wrapper.

    Wraps a MiniMax H3 video VAE so that decoding uses the fast tiled path.
    Output is a ``VAE`` you can use exactly like the original video VAE — just
    with an extra ``batch`` control for decode speed/memory.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE", {"tooltip": "The MiniMax H3 video VAE to wrap with the fast decoder."}),
                "batch": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "Tile batch size for h3_fast_decode. Larger values are faster "
                        "but use more memory.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    OUTPUT_TOOLTIPS = ("The wrapped MiniMax H3 video VAE that decodes via the fast tiled path.",)
    FUNCTION = "wrap"

    CATEGORY = "MiniMaxH3"
    DESCRIPTION = (
        "Wraps a MiniMax H3 video VAE so its decode uses the fast tiled decoder. "
        "Use the output exactly like a normal video VAE; the only extra control is "
        "the batch (tile batch size) parameter."
    )

    def wrap(self, vae, batch=4):
        return (_FastMiniMaxH3VideoVAE(vae, tile_batch_size=int(batch)),)
