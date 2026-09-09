"""Fast Video VAE node for MiniMax H3.

This is a ``VAE``-typed node: it takes a MiniMax H3 video VAE (the same kind you
would load with the core ``VAELoader``) plus a ``batch`` parameter, and returns a
**wrapped VAE** whose decode path uses the same batched-tile decoder as the
official ``MiniMax H3 Fast VAE Decode`` node (``h3_fast_*`` helpers below).

Downstream, you use the output exactly like a normal video VAE — plug it into a
standard ``VAEDecode`` / ``VAEEncode`` / any node that accepts a ``VAE`` input.
The only difference from a plain video VAE is the extra ``batch`` parameter that
controls the per-tile batch size of the fast decoder.
"""

from __future__ import annotations

import logging
import time

import torch

import comfy.ldm.minimax.vae
import comfy.model_management
import comfy.sd


# ---------------------------------------------------------------------------
# Batched-tile H3 VAE decoder.
#
# These helpers are copied verbatim from the official ``MiniMax H3 Fast VAE
# Decode`` node (ComfyUI-MiniMax-H3-MotionCache/fast_vae_decode.py) so the
# wrapped VAE decodes identically to that node, including the real
# ``tile_batch_size`` control and the OOM-retry-to-1 fallback.
# ---------------------------------------------------------------------------


def h3_fast_tiled_decode(model, z, tile_batch_size):
    height, width = z.shape[-2] * model.vae_ratio, z.shape[-1] * model.vae_ratio
    y_idx, y_len, y_overlap = model.split_tiles(height)
    x_idx, x_len, x_overlap = model.split_tiles(width)

    canvas = None
    row_tails = []
    out_y = 0
    for i, (i_pos, i_len) in enumerate(zip(y_idx, y_len)):
        zi, zl = i_pos // model.vae_ratio, i_len // model.vae_ratio
        new_tails = []
        left_tail = None
        out_x = 0
        row_height = 0
        for batch_start in range(0, len(x_idx), tile_batch_size):
            batch_end = min(batch_start + tile_batch_size, len(x_idx))
            latent_tiles = []
            for j in range(batch_start, batch_end):
                zj = x_idx[j] // model.vae_ratio
                zw = x_len[j] // model.vae_ratio
                latent_tiles.append(z[..., zi : zi + zl, zj : zj + zw])
            decoded = model._decode_pixels(torch.cat(latent_tiles, dim=0))
            tiles = decoded.split(z.shape[0], dim=0)

            for j, tile in zip(range(batch_start, batch_end), tiles):
                if i < len(y_idx) - 1:
                    new_tails.append(tile[..., -y_overlap[i] :, :].clone())
                next_left_tail = (
                    tile[..., :, -x_overlap[j] :].clone()
                    if j < len(x_idx) - 1
                    else None
                )
                if i > 0:
                    tile = model.blend(row_tails[j], tile, y_overlap[i - 1], dim=-2)
                if j > 0:
                    tile = model.blend(left_tail, tile, x_overlap[j - 1], dim=-1)
                left_tail = next_left_tail
                if i < len(y_idx) - 1:
                    tile = tile[..., : -y_overlap[i], :]
                if j < len(x_idx) - 1:
                    tile = tile[..., :, : -x_overlap[j]]
                if canvas is None:
                    canvas = torch.empty(
                        *tile.shape[:-2],
                        height,
                        width,
                        dtype=tile.dtype,
                        device=tile.device,
                    )
                canvas[
                    ..., out_y : out_y + tile.shape[-2], out_x : out_x + tile.shape[-1]
                ].copy_(tile)
                out_x += tile.shape[-1]
                row_height = tile.shape[-2]
        row_tails = new_tails
        out_y += row_height
    return canvas


def h3_fast_adaptive_decode(model, z, tile_batch_size):
    if model.tiling:
        return h3_fast_tiled_decode(model, z, tile_batch_size)
    return model._decode_pixels(z)


def h3_fast_decode_temporal(model, z, tile_batch_size):
    chunk_dec = model.tokens_chunk_size * model.vae_ratio_t
    split_count = int(model.token_drop > 0) + 1
    pseudo_total_tokens = z.shape[2] + model.token_drop

    pad_tokens = 0
    remainder = pseudo_total_tokens % model.tokens_chunk_size
    if remainder != 0:
        pad_tokens = model.tokens_chunk_size - remainder
        pseudo_total_tokens += pad_tokens

    num_chunks = pseudo_total_tokens // model.tokens_chunk_size - int(
        model.token_drop > 0
    )
    if num_chunks < 1:
        pad_tokens += model.tokens_chunk_size
        num_chunks += 1

    if pad_tokens > 0:
        pad_z = z[:, :, -1:, :, :].repeat(1, 1, pad_tokens, 1, 1)
        z = torch.cat([z, pad_z], dim=2)

    output_frames = model._decode_temporal_frame_plan(
        z.shape[2], num_chunks, pad_tokens
    )
    dec = None
    dec_overlap = None
    write_pos = 0

    def write_part(part):
        nonlocal dec, write_pos
        part_frames = part.shape[2]
        if part_frames <= 0:
            return
        if dec is None:
            out_shape = list(part.shape)
            out_shape[2] = output_frames
            dec = torch.empty(out_shape, dtype=part.dtype, device=part.device)
        copy_frames = min(part_frames, max(0, dec.shape[2] - write_pos))
        if copy_frames > 0:
            dec[:, :, write_pos : write_pos + copy_frames, :, :].copy_(
                part[:, :, :copy_frames, :, :]
            )
            write_pos += copy_frames

    for i in range(num_chunks):
        t_start_idx = i * model.tokens_chunk_size
        t_end_idx = t_start_idx + model.tokens_chunk_size + model.token_overlap
        clip_z = z[:, :, t_start_idx:t_end_idx, :, :]
        clip_dec = h3_fast_adaptive_decode(model, clip_z, tile_batch_size)

        for j in range(split_count):
            f_start_idx = j * chunk_dec
            f_end_idx = min(f_start_idx + chunk_dec, clip_dec.shape[2])
            clip_dec_chunk = clip_dec[:, :, f_start_idx:f_end_idx, :, :]
            clip_dec_chunk = clip_dec_chunk[:, :, model.frame_pre_padding :, :, :]
            if j == 0:
                if dec_overlap is not None:
                    clip_dec_chunk = model.blend(
                        dec_overlap, clip_dec_chunk, model.frame_overlap, dim=-3
                    )
                    dec_overlap = None
                write_part(clip_dec_chunk)
            else:
                dec_overlap = clip_dec_chunk.contiguous()

        if i == num_chunks - 1 and dec_overlap is not None:
            write_part(dec_overlap)
            dec_overlap = None

    return dec


def h3_fast_decode_model(model, z, tile_batch_size):
    latents_mean = model.latents_mean.view(1, -1, 1, 1, 1).to(z)
    latents_std = model.latents_std.view(1, -1, 1, 1, 1).to(z)
    z = z * latents_std + latents_mean

    if z.shape[2] == 1:
        dec = h3_fast_adaptive_decode(model, z, tile_batch_size)
        dec = dec[:, :, -1:, :, :]
    else:
        dec = h3_fast_decode_temporal(model, z, tile_batch_size)

    # NOTE: output must be float32 pixels in [0, 1] — identical to the stock
    # MiniMaxH3VideoVAE.decode contract used everywhere in the MiniMax H3
    # Director ecosystem. The MiniMaxH3 VAE's ``process_output`` is an identity
    # mapping, so the downstream VAEDecode / director consumers expect [0, 1].
    # We must NOT apply the standard ComfyUI ``*2 - 1`` transform here; doing so
    # produced output that looked like a color-inverted / negative image when
    # this fast wrapper was plugged into MiniMaxH3DirectorOpt.
    dec = dec.float()
    dec.mul_(model.pixel_std.to(dec)).add_(model.pixel_mean.to(dec)).clamp_(0.0, 1.0)
    return dec


def h3_fast_decode(vae, latent, tile_batch_size):
    tile_batch_size = max(1, int(tile_batch_size))
    vae.throw_exception_if_invalid()
    memory_used = vae.memory_used_decode(latent.shape, vae.vae_dtype)
    with comfy.model_management.cuda_device_context(vae.device):
        comfy.model_management.load_models_gpu(
            [vae.patcher],
            memory_required=memory_used,
            force_full_load=vae.disable_offload,
        )
        latent = latent.to(device=vae.device, dtype=vae.vae_dtype)
        retry = False
        try:
            images = h3_fast_decode_model(
                vae.first_stage_model, latent, tile_batch_size
            )
        except Exception as error:
            comfy.model_management.raise_non_oom(error)
            logging.warning(
                "MiniMax H3 Fast VAE Decode ran out of memory at tile batch %d; retrying with 1",
                tile_batch_size,
            )
            retry = True
        if retry:
            comfy.model_management.soft_empty_cache()
            images = h3_fast_decode_model(vae.first_stage_model, latent, 1)

        images = images.to(
            device=vae.output_device, dtype=vae.vae_output_dtype(), copy=True
        )
        vae.process_output(images)
    return images.movedim(1, -1)


class _FastMiniMaxH3VideoVAE(comfy.sd.VAE):
    """A ``VAE`` subclass that routes decode through the official batched-tile
    H3 decoder (``h3_fast_decode``).

    This produces results identical to the ``MiniMax H3 Fast VAE Decode`` node:
    the ``batch`` (tile batch size) parameter is honored, and an OOM during
    decode is retried with ``tile_batch_size=1``. Everything except ``.decode()``
    is inherited / delegated from the wrapped VAE.
    """

    def __init__(self, vae: comfy.sd.VAE, tile_batch_size: int = 4):
        # Copy the wrapped VAE wholesale, then override decode behavior.
        self.__dict__.update(vae.__dict__)
        self._inner_vae = vae
        self._tile_batch_size = int(tile_batch_size)
        if not isinstance(vae.first_stage_model, comfy.ldm.minimax.vae.MiniMaxH3VideoVAE):
            raise ValueError("Fast Video VAE requires the MiniMax H3 video VAE")

    def decode(self, samples_in, vae_options=None):
        if isinstance(samples_in, dict):
            samples_in = samples_in["samples"]

        latent = samples_in
        if getattr(latent, "is_nested", False):
            latent = latent.unbind()[0]

        start_time = time.perf_counter()
        # h3_fast_decode returns images in [B, T, H, W, C] (channels-last, the
        # comfy IMAGE convention). Flatten the temporal batch dim into the
        # leading frame dim so the result is [F, H, W, C] — exactly what a
        # stock MiniMax H3 VAEDecode produces, so downstream consumers
        # (segment frame cache, concat_chunks_lazy, VAEDecode) need no special
        # casing.
        images = h3_fast_decode(self, latent, self._tile_batch_size)
        if images.ndim == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        logging.info(
            "Fast Video VAE decode (batch %d) finished in %.2fs",
            self._tile_batch_size,
            time.perf_counter() - start_time,
        )
        return images

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


class MiniMaxH3FastVideoVAEOpt:
    """Fast Video VAE loader/wrapper.

    Wraps a MiniMax H3 video VAE so that decoding uses the same batched-tile
    decoder as the official ``MiniMax H3 Fast VAE Decode`` node. Output is a
    ``VAE`` you can use exactly like the original video VAE — just with an extra
    ``batch`` control for decode speed/memory.
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

    CATEGORY = "MiniMaxH3 Opt"
    DESCRIPTION = (
        "Wraps a MiniMax H3 video VAE so its decode uses the same batched-tile decoder "
        "as the official MiniMax H3 Fast VAE Decode node. "
        "Use the output exactly like a normal video VAE; the only extra control is "
        "the batch (tile batch size) parameter."
    )

    def wrap(self, vae, batch=4):
        return (_FastMiniMaxH3VideoVAE(vae, tile_batch_size=int(batch)),)
