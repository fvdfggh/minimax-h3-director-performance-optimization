"""Persisted vision-tower (ViT) output cache for the MiniMax H3 text encoder.

Why this exists
---------------
MiniMax H3's text encoder is Qwen3-VL. Reference images *and* videos are fed
into it as **vision entries** (``comfy/text_encoders/minimax.py`` wraps each one
in ``<Picture N>: <|vision_start|> … <|vision_end|>``), so each entry runs
through the ViT inside::

    MiniMaxQwen3VL.preprocess_embed(embed, device)   # text_encoders/minimax.py:88
        -> Qwen3VL.preprocess_embed(...)             # text_encoders/qwen3vl.py:62
            -> self.visual(image, grid)              #   the expensive pass

A video is chunked into 2-frame blocks (``minimax.py:188``) and **each block
gets its own ViT pass**, so one reference clip costs many. That pass — not the
VAE — dominates conditioning cost, and the same reference media is re-encoded
for every segment that mentions it (and on every run).

Why caching it is safe
----------------------
``preprocess_embed`` is called per image with only ``(embed, device)``; the
position/``<Picture N>`` numbering is computed *afterwards*
(``comfy/sd1_clip.py:236-243``) and only ever flows downstream into the
sequence, never back into the ViT. So its output is a pure function of the
pixels, the shape (which fixes ``grid``), the model variant and the
image/video flag. DeepStack features are consumed out-of-place
(``text_encoders/llama.py:930`` does ``x[m] = x[m] + d``), so handing out a
cached copy is bit-equivalent to recomputing.

The cache is keyed by **content hash**, never ``id()`` — ``id()`` both misses
every re-loaded duplicate and can collide after GC reuse.

Layout
------
``output/minimax_director_opt_cache/_vit/vit_<hash>.pt`` — global rather than
per-workflow, because the output depends on the media and model only, so the
same file is reusable across segments *and* workflows.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path

import torch

from . import cache_layout

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.vision_cache")

#: Set ``MINIMAX_VIT_CACHE=0`` to keep the encoder untouched.
_ENABLED_ENV = "MINIMAX_VIT_CACHE"

_orig_preprocess = None
_installed = False


def enabled() -> bool:
    return os.environ.get(_ENABLED_ENV, "1").strip().lower() not in ("0", "false", "no", "off")


def _vit_dir() -> Path | None:
    try:
        import folder_paths

        out = folder_paths.get_output_directory()
    except Exception:
        return None
    if not out:
        return None
    return Path(out) / cache_layout.CACHE_ROOT / cache_layout.VIT_CACHE_DIRNAME


def _content_digest(data: torch.Tensor) -> str:
    """Full-MD5 of the pixel payload.

    Deliberately not sampled: a collision here would silently bind the wrong
    visual features to a reference. The tensors are small (an image is a few MB,
    a video block is 2 frames) and a couple of ms is negligible next to a ViT
    forward pass.
    """
    t = data.detach().to("cpu")
    if t.dtype == torch.bfloat16 or t.dtype == torch.float16:
        # numpy has no bfloat16; widening is exact so the digest stays faithful.
        t = t.to(torch.float32)
    if not t.is_contiguous():
        t = t.contiguous()
    return hashlib.md5(t.numpy().tobytes()).hexdigest()


def _key_for(model_type: str, embed: dict, data: torch.Tensor) -> str:
    video_block = bool(embed.get("minimax_video_block", False))
    payload = "|".join(
        (
            str(model_type),
            "video" if video_block else "image",
            "x".join(str(d) for d in tuple(data.shape)),
            str(data.dtype).replace("torch.", ""),
            _content_digest(data),
        )
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:32]


def _write_atomic(path: Path, payload: dict) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
        return True
    except Exception as exc:
        log.debug("ViT cache write failed for %s: %s", path.name, exc)
        try:
            if tmp.is_file():
                tmp.unlink()
        except Exception:
            pass
        return False


def _load_entry(path: Path, device) -> tuple[torch.Tensor, dict] | None:
    try:
        entry = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        log.debug("ViT cache read failed for %s: %s", path.name, exc)
        return None
    if not isinstance(entry, dict) or "merged" not in entry or "extra" not in entry:
        return None
    merged = entry["merged"]
    extra = entry["extra"]
    if not torch.is_tensor(merged):
        return None
    merged = merged.to(device=device, dtype=torch.float32)
    # grid stays on its original device (CPU, per qwen_vl.py:61 device=data.device)
    # — the real preprocess_embed returns it untouched, so we must too. Only
    # merged/deepstack need to land on the compute device.
    grid = extra.get("grid")
    deepstack = extra.get("deepstack")
    if isinstance(deepstack, (list, tuple)):
        deepstack = [d.to(device=device, dtype=torch.float32) if torch.is_tensor(d) else d for d in deepstack]
    return merged, {"grid": grid, "deepstack": deepstack}


def prune(max_bytes: int | None = None, cache_dir: Path | None = None) -> int:
    """Drop oldest entries until the cache fits ``max_bytes``. Returns bytes freed."""
    d = cache_dir or _vit_dir()
    if d is None or not d.is_dir():
        return 0
    limit = int(max_bytes if max_bytes is not None else cache_layout.VIT_CACHE_MAX_BYTES)
    try:
        files = [(p, p.stat().st_size) for p in d.glob(f"{cache_layout.VIT_PREFIX}_*{cache_layout.VIT_SUFFIX}") if p.is_file()]
    except Exception:
        return 0
    total = sum(sz for _, sz in files)
    if total <= limit:
        return 0
    files.sort(key=lambda it: it[0].stat().st_mtime)
    freed = 0
    for p, sz in files:
        if total <= limit:
            break
        try:
            p.unlink()
            total -= sz
            freed += sz
        except Exception:
            continue
    if freed:
        log.info("ViT cache trimmed by %s (limit %s)", _human(freed), _human(limit))
    return freed


def _human(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if f < 1024 or unit == "GB":
            return f"{f:.1f}{unit}"
        f /= 1024
    return f"{f:.1f}GB"


def stats() -> dict:
    d = _vit_dir()
    count = 0
    total = 0
    if d is not None and d.is_dir():
        try:
            for p in d.glob(f"{cache_layout.VIT_PREFIX}_*{cache_layout.VIT_SUFFIX}"):
                if p.is_file():
                    count += 1
                    total += p.stat().st_size
        except Exception:
            pass
    return {"dir": str(d) if d else None, "entries": count, "bytes": total, "human": _human(total), "installed": _installed}


def _make_patched(orig):
    """Wrap ``preprocess_embed`` so each vision entry is memoised to disk."""

    def preprocess_embed(self, embed, device):
        if not isinstance(embed, dict) or embed.get("type") != "image":
            return orig(self, embed, device)
        data = embed.get("data")
        if not torch.is_tensor(data):
            return orig(self, embed, device)

        key = None
        path = None
        d = None
        try:
            d = _vit_dir()
            if d is not None:
                key = _key_for(getattr(self, "model_type", "unknown"), embed, data)
                path = d / f"{cache_layout.VIT_PREFIX}_{key}{cache_layout.VIT_SUFFIX}"
                if path.is_file():
                    hit = _load_entry(path, device)
                    if hit is not None:
                        try:
                            os.utime(path, None)  # LRU touch
                        except Exception:
                            pass
                        log.debug("ViT cache hit %s", path.name)
                        return hit
        except Exception as exc:
            log.debug("ViT cache lookup skipped: %s", exc)
            key = None

        merged, extra = orig(self, embed, device)

        if key is not None and path is not None and d is not None:
            try:
                grid = (extra or {}).get("grid")
                deepstack = (extra or {}).get("deepstack")
                payload = {
                    "merged": merged.detach().to("cpu"),
                    "extra": {
                        "grid": grid.detach().cpu() if torch.is_tensor(grid) else grid,
                        "deepstack": [x.detach().cpu() for x in deepstack]
                        if isinstance(deepstack, (list, tuple))
                        else deepstack,
                    },
                    "meta": {
                        "model_type": getattr(self, "model_type", "unknown"),
                        "video_block": bool(embed.get("minimax_video_block", False)),
                        "shape": list(data.shape),
                        "ts": time.time(),
                    },
                }
                if _write_atomic(path, payload):
                    if _puts[0] >= 10:
                        _puts[0] = 0
                        prune(cache_dir=d)
                    else:
                        _puts[0] += 1
            except Exception as exc:
                log.debug("ViT cache store skipped: %s", exc)
        return merged, extra

    return preprocess_embed


_puts = [0]


def install() -> bool:
    """Patch the encoder's ViT entry point. Idempotent; safe to call repeatedly."""
    global _orig_preprocess, _installed
    if _installed:
        return True
    if not enabled():
        log.info("ViT cache disabled via %s", _ENABLED_ENV)
        return False
    try:
        from comfy.text_encoders.minimax import MiniMaxQwen3VL
    except Exception as exc:
        log.debug("ViT cache not installed (MiniMaxQwen3VL unavailable): %s", exc)
        return False
    try:
        # Must patch the *override* in minimax.py, not the base in qwen3vl.py:
        # the override handles video blocks itself and never reaches the base.
        _orig_preprocess = MiniMaxQwen3VL.preprocess_embed
        MiniMaxQwen3VL.preprocess_embed = _make_patched(_orig_preprocess)
        _installed = True
        d = _vit_dir()
        log.info("ViT cache installed at %s (%s)", d, stats()["human"])
        return True
    except Exception as exc:
        log.warning("ViT cache install failed: %s", exc)
        _orig_preprocess = None
        _installed = False
        return False


def uninstall() -> None:
    """Restore the original encoder. Idempotent."""
    global _orig_preprocess, _installed
    if not _installed or _orig_preprocess is None:
        return
    try:
        from comfy.text_encoders.minimax import MiniMaxQwen3VL

        MiniMaxQwen3VL.preprocess_embed = _orig_preprocess
    except Exception as exc:
        log.debug("ViT cache uninstall failed: %s", exc)
    finally:
        _orig_preprocess = None
        _installed = False
