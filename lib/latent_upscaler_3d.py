"""MiniMax H3 latent upscaler (3D) packaged as a reusable ``model`` object.

This module is a self-contained port of the ``Minimax H3 Latent Upscaler (3D)``
inference node (``Comfyui_Minimax_h3_latent_Upscaler``), refactored so the
upscaler behaves like a *model* instead of a one-shot node:

* :class:`MiniMaxH3LatentUpscaleModel` is a plain Python object that can be
  passed around, stored, and **called from anywhere** (other nodes, the
  Director pipeline, scripts) via ``model.upscale(latent, ...)`` /
  ``model(latent)``.
* It is returned by the ``MiniMax H3 Latent Upscale Model Loader`` node as a
  ``LATENT_UPSCALE_MODEL`` output and consumed by the
  ``MiniMax H3 Latent Upscale With Model`` node, so the weights are loaded once
  and can be reused by as many apply-nodes as you like.

Behaviour (sizing math, pixel alignment, temporal chunking, normalization,
precision / VRAM handling) is byte-for-byte the same as the original node.
"""

from __future__ import annotations

import gc
import glob  # noqa: F401  (kept for parity with the original module)
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

import folder_paths

try:
    import comfy.model_management as mm

    HAS_COMFY_MM = True
except ImportError:  # pragma: no cover - only outside ComfyUI
    HAS_COMFY_MM = False

from einops import rearrange


# ==========================================
# Register model folder
# ==========================================
_LATENT_UPSCALE_FOLDER = "latent_upscale_models"
if _LATENT_UPSCALE_FOLDER not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path(
        _LATENT_UPSCALE_FOLDER,
        os.path.join(folder_paths.models_dir, _LATENT_UPSCALE_FOLDER),
    )

VAE_DOWNSAMPLE = 16

# ==========================================
# Minimax H3 latent normalization stats (24 channels)
# ==========================================
LATENTS_MEAN = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264
]
LATENTS_STD = [
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877,
    2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264,
    3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523
]

MODE_SCALE = "scale by multiplier"
MODE_TARGET_DIMENSIONS = "target dimensions"
MODE_MEGAPIXELS = "megapixels"
UPSCALE_MODES = [MODE_SCALE, MODE_TARGET_DIMENSIONS, MODE_MEGAPIXELS]

_UPSCALE_MODEL_TYPE = "LATENT_UPSCALE_MODEL"


def _make_norm_tensors(device, dtype):
    mean = torch.tensor(LATENTS_MEAN, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(LATENTS_STD, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    return mean, std


# ==========================================
# ROCm / Device helper functions
# ==========================================
def _is_rocm_build():
    return getattr(torch.version, "hip", None) is not None


def _resolve_device(backend):
    if backend == "cpu":
        return torch.device("cpu")
    if backend == "rocm":
        if not _is_rocm_build():
            raise RuntimeError("ROCm was selected, but this PyTorch build has no HIP/ROCm support.")
        if not torch.cuda.is_available():
            raise RuntimeError("ROCm was selected, but PyTorch cannot access an AMD GPU.")
        return torch.device("cuda")
    if backend == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"Unsupported device backend: {backend}")


def _backend_label(device):
    if device.type == "cuda" and _is_rocm_build():
        return f"ROCm/HIP {torch.version.hip}"
    if device.type == "cuda":
        return f"CUDA {getattr(torch.version, 'cuda', None) or 'unknown'}"
    return "CPU"


# ==========================================
# 3D network components
# ==========================================
def normalization(channels):
    return nn.GroupNorm(32, channels)


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


class AttnBlock3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = normalization(in_channels)
        self.q = nn.Conv3d(in_channels, in_channels, 1)
        self.k = nn.Conv3d(in_channels, in_channels, 1)
        self.v = nn.Conv3d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv3d(in_channels, in_channels, 1)

    def forward(self, x):
        h = self.norm(x)
        q = rearrange(self.q(h), "b c t h w -> b 1 (t h w) c")
        k = rearrange(self.k(h), "b c t h w -> b 1 (t h w) c")
        v = rearrange(self.v(h), "b c t h w -> b 1 (t h w) c")
        h = F.scaled_dot_product_attention(q, k, v)
        h = rearrange(h, "b 1 (t h w) c -> b c t h w", t=x.shape[2], h=x.shape[3], w=x.shape[4])
        return x + self.proj_out(h)


class ResBlockEmb3D(nn.Module):
    def __init__(self, channels, emb_channels, dropout=0, out_channels=None):
        super().__init__()
        self.out_channels = out_channels or channels
        self.in_layers = nn.Sequential(
            normalization(channels), nn.SiLU(),
            nn.Conv3d(channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(), nn.Linear(emb_channels, 2 * self.out_channels),
        )
        self.out_norm = normalization(self.out_channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(), nn.Dropout(p=dropout),
            zero_module(nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1)),
        )
        self.skip = (
            nn.Conv3d(channels, self.out_channels, 1)
            if self.out_channels != channels else nn.Identity()
        )

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return self.skip(x) + h


class TemporalConv(nn.Module):
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        padding = kernel_size // 2
        self.norm = normalization(channels)
        self.dwconv = nn.Conv3d(channels, channels,
                                kernel_size=(kernel_size, 1, 1),
                                padding=(padding, 0, 0),
                                groups=channels)
        self.pwconv = nn.Conv3d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.pwconv.weight)
        nn.init.zeros_(self.pwconv.bias)

    def forward(self, x):
        identity = x
        h = self.norm(x)
        h = F.silu(h)
        h = self.dwconv(h)
        h = self.pwconv(h)
        return identity + h


# ==========================================
# Pure-3D backbone with Temporal Chunking
# ==========================================
class LatentResizer3D(nn.Module):
    def __init__(self, in_channels=24, in_blocks=12, out_blocks=12,
                 channels=512, dropout=0.1, attn=False,
                 temporal_every=2, temporal_kernel=5):
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels, channels, 3, padding=1)
        embed_dim = 64
        self.embed = nn.Sequential(
            nn.Linear(1, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))

        self.in_blocks = nn.ModuleList()
        for b in range(in_blocks):
            if (b == 1 or b == in_blocks - 1) and attn:
                self.in_blocks.append(AttnBlock3D(channels))
            self.in_blocks.append(ResBlockEmb3D(channels, embed_dim, dropout))
            if temporal_every > 0 and b % temporal_every == 0:
                self.in_blocks.append(TemporalConv(channels, temporal_kernel))

        self.out_blocks = nn.ModuleList()
        for b in range(out_blocks):
            if (b == 1 or b == out_blocks - 1) and attn:
                self.out_blocks.append(AttnBlock3D(channels))
            self.out_blocks.append(ResBlockEmb3D(channels, embed_dim, dropout))
            if temporal_every > 0 and b % temporal_every == 0:
                self.out_blocks.append(TemporalConv(channels, temporal_kernel))

        self.norm_out = normalization(channels)
        self.conv_out = nn.Conv3d(channels, in_channels, 3, padding=1)

    def forward(self, x, scale=None, target_size=None, enable_chunking=True):
        if target_size is not None:
            size = target_size
        elif scale is not None:
            size = tuple(int(round(s * scale)) for s in x.shape[-3:])
        else:
            return x

        if size == x.shape[-3:]:
            return x

        B, C, T, H, W = x.shape

        tk = 0
        for b in self.in_blocks:
            if isinstance(b, TemporalConv):
                tk = b.dwconv.weight.shape[2]
                break

        overlap = tk
        chunk = 32

        if not enable_chunking or T <= chunk:
            return self._forward_seg(x, scale, size)

        print(f"[MinimaxH3-3D] temporal chunking: T={T} chunks={(T + chunk - 1) // chunk} overlap={overlap}")

        x_padded = F.pad(x, (0, 0, 0, 0, overlap, overlap), mode='replicate')

        out_full = torch.zeros(B, C, T, size[-2], size[-1], device=x.device, dtype=x.dtype)
        weight_full = torch.zeros(1, 1, T, 1, 1, device=x.device, dtype=x.dtype)

        start = 0
        while start < T:
            seg_start = start
            seg_end = min(T, start + chunk)
            out_start = max(0, seg_start - overlap)
            out_end = min(T, seg_end + overlap)
            lo = max(0, out_start - overlap)
            hi = min(T + 2 * overlap, out_end + overlap)

            seg = x_padded[:, :, lo:hi].contiguous()
            seg_size = (hi - lo, size[-2], size[-1])
            seg_out = self._forward_seg(seg, scale, seg_size)

            s0 = (out_start + overlap) - lo
            s1 = s0 + (out_end - out_start)
            valid_out = seg_out[:, :, s0:s1]
            n_valid = out_end - out_start

            weight = torch.ones(n_valid, device=x.device, dtype=x.dtype)
            if seg_start > out_start:
                blend_len = seg_start - out_start
                weight[:blend_len] = torch.arange(1, blend_len + 1, device=x.device, dtype=x.dtype) / (blend_len + 1)
            if out_end > seg_end:
                blend_len = out_end - seg_end
                weight[-blend_len:] = torch.arange(blend_len, 0, -1, device=x.device, dtype=x.dtype) / (blend_len + 1)

            out_full[:, :, out_start:out_end] += valid_out * weight.view(1, 1, n_valid, 1, 1)
            weight_full[:, :, out_start:out_end] += weight.view(1, 1, n_valid, 1, 1)

            start += chunk
            del seg, seg_out, valid_out
            if start % (chunk * 4) == 0:
                gc.collect()

        out_full = out_full / weight_full.clamp(min=1e-8)
        return out_full

    def _forward_seg(self, x, scale, size):
        scale_emb = torch.tensor(
            [scale - 1 if scale is not None else 0.0],
            dtype=x.dtype, device=x.device).unsqueeze(0)
        emb = self.embed(scale_emb)

        x = self.conv_in(x)
        for b in self.in_blocks:
            if isinstance(b, ResBlockEmb3D):
                emb_t = emb.expand(x.shape[0], -1)
                x = b(x, emb_t)
            else:
                x = b(x)

        x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)

        for b in self.out_blocks:
            if isinstance(b, ResBlockEmb3D):
                emb_t = emb.expand(x.shape[0], -1)
                x = b(x, emb_t)
            else:
                x = b(x)

        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        return x


# ==========================================
# Model loading
# ==========================================
MODEL_CACHE: dict[str, "LatentResizer3D"] = {}


def get_models_dir():
    return folder_paths.get_folder_paths(_LATENT_UPSCALE_FOLDER)[0]


def scan_models():
    names = [
        name for name in folder_paths.get_filename_list(_LATENT_UPSCALE_FOLDER)
        if os.path.splitext(name)[1].lower() in (".pth", ".safetensors")
    ]
    return names if names else [f"(place models in: {get_models_dir()})"]


def _load_raw_sd(path):
    if path.endswith('.safetensors'):
        try:
            from safetensors import safe_open
            with safe_open(path, framework="pt", device="cpu") as f:
                sd = {k: f.get_tensor(k) for k in f.keys()}
        except ImportError:
            from safetensors.torch import load_file
            sd = load_file(path, device='cpu')
    else:
        sd = torch.load(path, map_location='cpu', weights_only=False)

    if isinstance(sd, dict) and 'model' in sd:
        sd = sd['model']
    sd = {k: v.to(torch.float16) if v.dtype == torch.float8_e4m3fn else v
          for k, v in sd.items()}
    return sd


def _extract_upscaler_sd(sd):
    if any(k.startswith("upscaler.") for k in sd):
        return {k[len("upscaler."):]: v for k, v in sd.items() if k.startswith("upscaler.")}
    return sd


def _detect_arch(sd):
    cfg = {
        "in_channels": 24, "in_blocks": 12, "out_blocks": 12, "channels": 512,
        "dropout": 0.1, "attn": False, "temporal_every": 2, "temporal_kernel": 5,
    }
    conv_key = 'conv_in.weight'
    if conv_key in sd:
        cfg["in_channels"] = sd[conv_key].shape[1]
        cfg["channels"] = sd[conv_key].shape[0]

    in_ids, out_ids = set(), set()
    temporal_in_indices, temporal_out_indices = set(), set()
    for k in sd.keys():
        m = re.match(r'in_blocks\.(\d+)\.in_layers\.', k)
        if m: in_ids.add(int(m.group(1)))
        m = re.match(r'out_blocks\.(\d+)\.in_layers\.', k)
        if m: out_ids.add(int(m.group(1)))
        m = re.match(r'in_blocks\.(\d+)\.dwconv\.weight', k)
        if m: temporal_in_indices.add(int(m.group(1)))
        m = re.match(r'out_blocks\.(\d+)\.dwconv\.weight', k)
        if m: temporal_out_indices.add(int(m.group(1)))

    if in_ids: cfg["in_blocks"] = len(in_ids)
    if out_ids: cfg["out_blocks"] = len(out_ids)

    if temporal_in_indices or temporal_out_indices:
        cfg["temporal_every"] = 2
        for k in sd.keys():
            if 'dwconv.weight' in k and k.endswith('dwconv.weight'):
                cfg["temporal_kernel"] = sd[k].shape[2]
                break
    else:
        cfg["temporal_every"] = 0

    cfg["attn"] = False
    return cfg


def load_model(name, device, precision):
    backend_lbl = _backend_label(device)
    cache_key = f"{name}::{backend_lbl}::{precision}"
    if cache_key in MODEL_CACHE:
        model = MODEL_CACHE[cache_key]
        return model.to(device, non_blocking=True)

    try:
        path = folder_paths.get_full_path_or_raise(_LATENT_UPSCALE_FOLDER, name)
    except Exception as e:
        raise FileNotFoundError(f"Model file not found: {name}") from e

    raw_sd = _load_raw_sd(path)
    up_sd = _extract_upscaler_sd(raw_sd)
    cfg = _detect_arch(up_sd)

    model = LatentResizer3D(
        in_channels=cfg["in_channels"], in_blocks=cfg["in_blocks"], out_blocks=cfg["out_blocks"],
        channels=cfg["channels"], dropout=cfg["dropout"], attn=cfg["attn"],
        temporal_every=cfg["temporal_every"], temporal_kernel=cfg["temporal_kernel"],
    )
    model.load_state_dict(up_sd, strict=True)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}.get(precision, torch.float32)
    model = model.to(device).eval().requires_grad_(False)
    if dtype != torch.float32:
        model = model.to(dtype)

    MODEL_CACHE[cache_key] = model
    print(f"[MinimaxH3-3D] Loaded upscale model: {name}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,} | "
          f"Attn: forced off | Temporal: {'on' if cfg['temporal_every'] > 0 else 'off'} "
          f"(every={cfg['temporal_every']}, kernel={cfg['temporal_kernel']}) | "
          f"Backend: {backend_lbl} | Precision: {precision}")
    return model


def _normalize_mode(mode):
    """Accept ``"megapixels"`` / ``{"mode": "megapixels", ...}`` / ``UP`` enum."""
    if isinstance(mode, dict):
        return mode.get("mode") or MODE_SCALE, mode
    return mode, {}


class MiniMaxH3LatentUpscaleModel:
    """A *model* wrapper around the MiniMax H3 3D latent upscaler.

    Instances are produced by the ``MiniMax H3 Latent Upscale Model Loader``
    node (``LATENT_UPSCALE_MODEL`` output) and can be consumed by the
    ``MiniMax H3 Latent Upscale With Model`` node — or called directly from
    any other Python code:

    ````python
    model = MiniMaxH3LatentUpscaleModel("my_upscaler.safetensors",
                                        device="cuda", precision="fp16")
    upscaled = model(latent, mode="megapixels", megapixels=1.0)
    ````
    """

    #: Socket type used in ComfyUI graphs.
    TYPE = _UPSCALE_MODEL_TYPE

    def __init__(self, model_name: str, device: str = "cuda", precision: str = "fp16",
                 align: int = 32, enable_temporal_chunking: bool = True,
                 force_unload: bool = True,
                 mode: str = MODE_SCALE, scale: float = 2.0,
                 width: int = 1280, height: int = 704, megapixels: float = 1.0,
                 lazy: bool = False):
        if model_name.startswith('('):
            raise ValueError("Please place model files into the latent_upscale_models directory")

        self.model_name = model_name
        self.device = device
        self.precision = precision
        # Defaults so `model(latent)` works without re-specifying everything.
        self.align = align
        self.enable_temporal_chunking = enable_temporal_chunking
        self.force_unload = force_unload
        self.mode = mode
        self.scale = scale
        self.width = width
        self.height = height
        self.megapixels = megapixels

        self._model = None
        if not lazy:
            self.load()

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------
    def load(self):
        """Load (or fetch from cache) the underlying ``LatentResizer3D``."""
        self._model = load_model(self.model_name, _resolve_device(self.device), self.precision)
        return self._model

    @property
    def model(self):
        if self._model is None:
            self.load()
        return self._model

    @property
    def torch_device(self):
        return _resolve_device(self.device)

    def to(self, device):
        """Move the weights to ``device`` (``"cpu"`` to offload)."""
        if self._model is not None:
            self._model = self._model.to(device)
        return self

    def unload(self):
        """Offload weights back to CPU and release the VRAM cache."""
        if self._model is not None:
            self._model = self._model.to("cpu", non_blocking=True)
        self._release_vram(self.torch_device)
        return self

    @staticmethod
    def _release_vram(dev):
        if dev.type != "cuda":
            return
        if HAS_COMFY_MM:
            mm.soft_empty_cache()
        else:
            torch.cuda.empty_cache()
        gc.collect()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def upscale(self, latent, mode=None, scale=None, width=None, height=None,
                megapixels=None, align=None, enable_temporal_chunking=None,
                force_unload=None):
        """Upscale ``latent`` (dict ``{"samples": ...}`` or raw tensor).

        Returns the same container type it received.
        """
        selected_mode, mode_cfg = _normalize_mode(mode if mode is not None else self.mode)
        if selected_mode is None:
            selected_mode = self.mode
        scale = self.scale if scale is None else scale
        width = self.width if width is None else width
        height = self.height if height is None else height
        megapixels = self.megapixels if megapixels is None else megapixels
        align = self.align if align is None else align
        enable_temporal_chunking = (
            self.enable_temporal_chunking if enable_temporal_chunking is None else enable_temporal_chunking
        )
        force_unload = self.force_unload if force_unload is None else force_unload
        # A DynamicCombo-style dict carries the sub-values.
        scale = mode_cfg.get("scale", scale)
        width = mode_cfg.get("width", width)
        height = mode_cfg.get("height", height)
        megapixels = mode_cfg.get("megapixels", megapixels)

        is_dict = isinstance(latent, dict)
        src = latent["samples"] if is_dict else latent
        orig_dtype = src.dtype
        was_4d = (src.dim() == 4)

        dev = _resolve_device(self.device)
        compute_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[self.precision]

        s = src.to(device=dev, dtype=compute_dtype, copy=True)
        if was_4d:
            s = s.unsqueeze(2)

        b, c, t, h_in, w_in = s.shape
        downsample = VAE_DOWNSAMPLE

        # 1. Calculate target size
        if selected_mode == MODE_SCALE:
            w_pixel_target = w_in * downsample * scale
            h_pixel_target = h_in * downsample * scale
            effective_scale = scale
        elif selected_mode == MODE_TARGET_DIMENSIONS:
            w_pixel_target = float(width)
            h_pixel_target = float(height)
            effective_scale = (w_pixel_target / (w_in * downsample) + h_pixel_target / (h_in * downsample)) / 2.0
        elif selected_mode == MODE_MEGAPIXELS:
            target_pixels = megapixels * 1024 * 1024
            aspect_ratio = w_in / h_in
            h_pixel_target = (target_pixels / aspect_ratio) ** 0.5
            w_pixel_target = h_pixel_target * aspect_ratio
            effective_scale = (w_pixel_target / (w_in * downsample) + h_pixel_target / (h_in * downsample)) / 2.0
        else:
            raise ValueError(f"Unsupported mode: {selected_mode}")

        # 2. Pixel-space alignment
        alignment = max(1, align)
        w_pixel_aligned = round(w_pixel_target / alignment) * alignment
        h_pixel_aligned = round(h_pixel_target / alignment) * alignment

        w_pixel_final = round(w_pixel_aligned / downsample) * downsample
        h_pixel_final = round(h_pixel_aligned / downsample) * downsample

        w_out = max(1, int(w_pixel_final // downsample))
        h_out = max(1, int(h_pixel_final // downsample))

        if effective_scale < 1.0 and (w_out < w_in or h_out < h_in):
            raise ValueError("This model only supports upscaling (effective scale >= 1.0).")

        if w_out == w_in and h_out == h_in:
            return latent

        print(f"[MinimaxH3-3D] Latent {w_in}x{h_in} -> {w_out}x{h_out} | "
              f"Pixels {w_out * downsample}x{h_out * downsample} | scale={effective_scale:.3f}")

        # 3. Inference
        net = self.model.to(dev, non_blocking=True)
        norm_mean, norm_std = _make_norm_tensors(dev, compute_dtype)

        with torch.inference_mode():
            s_norm = (s - norm_mean) / norm_std
            del s

            out = net(s_norm, scale=effective_scale, target_size=(t, h_out, w_out),
                      enable_chunking=enable_temporal_chunking)

            del s_norm
            out = out * norm_std + norm_mean

        if was_4d:
            out = out.squeeze(2)

        out = out.to(device="cpu", dtype=orig_dtype, non_blocking=True)

        # 4. VRAM management
        if dev.type == "cuda":
            if force_unload:
                net.to("cpu", non_blocking=True)
                print("[MinimaxH3-3D] Model offloaded to CPU. VRAM released.")
            self._release_vram(dev)

        if is_dict:
            return {"samples": out}
        return out

    # Convenience aliases so the object can be used "like a model".
    def __call__(self, latent, **kwargs):
        return self.upscale(latent, **kwargs)

    def resample_latent(self, samples, **kwargs):
        """Alias used by ComfyUI's generic latent-upscale-model consumers."""
        return self.upscale(samples, **kwargs)

    def __repr__(self):
        return (f"MiniMaxH3LatentUpscaleModel(name={self.model_name!r}, device={self.device!r}, "
                f"precision={self.precision!r})")


__all__ = [
    "MiniMaxH3LatentUpscaleModel",
    "LatentResizer3D",
    "load_model",
    "scan_models",
    "get_models_dir",
    "UPSCALE_MODES",
    "MODE_SCALE",
    "MODE_TARGET_DIMENSIONS",
    "MODE_MEGAPIXELS",
    "LATENT_UPSCALE_MODEL_TYPE",
]

LATENT_UPSCALE_MODEL_TYPE = _UPSCALE_MODEL_TYPE
