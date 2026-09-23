"""Image → base64 helpers for the live-preview and thumbnail payloads.

The preview paths (TAE sampling preview, per-frame segment thumbnails) both need
"PIL image as a base64 JPEG string", and the tensor variant is just that plus a
float→uint8 conversion, so both live here.
"""

from __future__ import annotations

import base64
import io

import numpy as np
import torch
from PIL import Image


def pil_to_jpeg_b64(pil: Image.Image, *, quality: int = 80) -> str:
    """Encode a PIL image as a base64 JPEG string."""
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=int(quality))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def tensor_to_pil(frame: torch.Tensor) -> Image.Image:
    """``[H,W,C]`` tensor in 0..1 → 8-bit PIL image."""
    arr = (frame.detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def tensor_frame_to_jpeg_b64(frame: torch.Tensor, *, quality: int = 88) -> str:
    """``[H,W,C]`` tensor → base64 JPEG (per-frame thumbnails)."""
    return pil_to_jpeg_b64(tensor_to_pil(frame), quality=quality)
