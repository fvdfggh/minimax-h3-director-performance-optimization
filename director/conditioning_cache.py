"""Conditioning cache — save/load CLIP-encoded tensors to avoid repeated encoding."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import torch

log = logging.getLogger("ComfyUI-MiniMaxH3-Director-Cached.conditioning_cache")

# Cache directory relative to ComfyUI output
CACHE_SUBDIR = "minimax_conditioning_cache"


def _get_cache_dir(node_id: str | None = None) -> Path:
    """Get or create the conditioning cache directory."""
    from folder_paths import output_directory
    
    base = Path(output_directory) / CACHE_SUBDIR
    if node_id:
        base = base / f"node_{node_id}"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _hash_ref_images(ref_images: Any) -> str:
    """Generate a hash for reference images to include in cache key.
    
    This ensures cache invalidation when reference images change.
    """
    if ref_images is None:
        return "none"
    
    try:
        # Handle different ref_images formats
        if isinstance(ref_images, dict):
            # Dict format: {"ref_image_0": tensor, ...}
            hash_parts = []
            for key in sorted(ref_images.keys()):
                tensor = ref_images[key]
                if isinstance(tensor, torch.Tensor):
                    # Hash the tensor data
                    tensor_bytes = tensor.cpu().numpy().tobytes()
                    hash_parts.append(f"{key}:{hashlib.md5(tensor_bytes).hexdigest()[:8]}")
                else:
                    hash_parts.append(f"{key}:{hash(str(tensor))}")
            return "|".join(hash_parts) if hash_parts else "empty_dict"
        
        elif isinstance(ref_images, (list, tuple)):
            # List/tuple format
            hash_parts = []
            for i, item in enumerate(ref_images):
                if isinstance(item, torch.Tensor):
                    tensor_bytes = item.cpu().numpy().tobytes()
                    hash_parts.append(f"img_{i}:{hashlib.md5(tensor_bytes).hexdigest()[:8]}")
                elif isinstance(item, dict):
                    # Nested dict
                    for key in sorted(item.keys()):
                        tensor = item[key]
                        if isinstance(tensor, torch.Tensor):
                            tensor_bytes = tensor.cpu().numpy().tobytes()
                            hash_parts.append(f"{key}:{hashlib.md5(tensor_bytes).hexdigest()[:8]}")
            return "|".join(hash_parts) if hash_parts else "empty_list"
        
        else:
            return f"other:{hash(str(ref_images))}"
    
    except Exception as exc:
        log.warning("Failed to hash ref_images: %s", exc)
        return f"error:{hash(str(ref_images))}"


def _prompt_hash(prompt: str, width: int, height: int, length: int, task_key: str, 
                 ref_image_size: str = "match", ref_images: Any = None) -> str:
    """Generate a hash key for the prompt + dimensions + reference images.
    
    Including ref_images hash ensures cache invalidation when reference images change.
    """
    ref_hash = _hash_ref_images(ref_images)
    content = f"{prompt}|{width}|{height}|{length}|{task_key}|{ref_image_size}|{ref_hash}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def save_conditioning_cache(
    node_id: str | None,
    segment_index: int,
    positive: Any,
    negative: Any,
    latent: Any,
    prompt: str,
    width: int,
    height: int,
    length: int,
    task_key: str,
    ref_image_size: str = "match",
    ref_images: Any = None,
) -> Path | None:
    """Save conditioning tensors to disk cache.
    
    Returns the cache file path, or None if saving failed.
    """
    try:
        cache_dir = _get_cache_dir(node_id)
        prompt_key = _prompt_hash(prompt, width, height, length, task_key, ref_image_size, ref_images)
        cache_file = cache_dir / f"seg_{segment_index:03d}_{prompt_key}.pt"
        
        # Prepare conditioning for serialization
        # ComfyUI conditioning is a list of [tensor, dict] pairs
        def prepare_for_save(cond):
            if cond is None:
                return None
            if isinstance(cond, (list, tuple)):
                prepared = []
                for item in cond:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        tensor, config = item[0], item[1]
                        # Ensure tensor is on CPU for storage
                        if hasattr(tensor, 'cpu'):
                            tensor = tensor.cpu()
                        prepared.append([tensor, config])
                    else:
                        prepared.append(item)
                return prepared
            return cond
        
        cache_data = {
            "positive": prepare_for_save(positive),
            "negative": prepare_for_save(negative),
            "latent": prepare_for_save(latent) if latent is not None else None,
            "metadata": {
                "segment_index": segment_index,
                "prompt_hash": prompt_key,
                "width": width,
                "height": height,
                "length": length,
                "task_key": task_key,
                "ref_image_size": ref_image_size,
                "prompt_preview": prompt[:200] + "..." if len(prompt) > 200 else prompt,
            }
        }
        
        torch.save(cache_data, cache_file)
        log.info("Conditioning cached: seg #%d → %s", segment_index + 1, cache_file.name)
        return cache_file
        
    except Exception as exc:
        log.warning("Failed to cache conditioning for seg #%d: %s", segment_index + 1, exc)
        return None


def load_conditioning_cache(
    node_id: str | None,
    segment_index: int,
    prompt: str,
    width: int,
    height: int,
    length: int,
    task_key: str,
    ref_image_size: str = "match",
    ref_images: Any = None,
) -> dict | None:
    """Load cached conditioning tensors from disk.
    
    Returns dict with 'positive', 'negative', 'latent' keys, or None if cache miss.
    Tensors are moved to CUDA (GPU) for use by the model.
    """
    try:
        cache_dir = _get_cache_dir(node_id)
        prompt_key = _prompt_hash(prompt, width, height, length, task_key, ref_image_size, ref_images)
        
        # Search for matching cache file
        pattern = f"seg_{segment_index:03d}_{prompt_key}.pt"
        cache_file = cache_dir / pattern
        
        if not cache_file.exists():
            # Try to find any cache file for this segment (prompt might have changed)
            log.debug("No cache hit for seg #%d (key=%s)", segment_index + 1, prompt_key)
            return None
        
        cache_data = torch.load(cache_file, map_location="cpu", weights_only=False)
        
        # Validate cache metadata
        meta = cache_data.get("metadata", {})
        if (meta.get("width") != width or 
            meta.get("height") != height or 
            meta.get("length") != length or
            meta.get("task_key") != task_key):
            log.info("Cache invalidated for seg #%d (dimensions changed)", segment_index + 1)
            return None
        
        # Move tensors to CUDA (GPU) for model use
        def move_to_device(cond):
            """Recursively move all tensors in conditioning to CUDA."""
            if cond is None:
                return None
            if isinstance(cond, torch.Tensor):
                return cond.cuda()
            if isinstance(cond, (list, tuple)):
                return type(cond)(move_to_device(item) for item in cond)
            if isinstance(cond, dict):
                return {k: move_to_device(v) for k, v in cond.items()}
            return cond
        
        positive = move_to_device(cache_data["positive"])
        negative = move_to_device(cache_data["negative"])
        latent = move_to_device(cache_data["latent"])
        
        log.info("Conditioning cache HIT: seg #%d from %s (moved to CUDA)", segment_index + 1, cache_file.name)
        return {
            "positive": positive,
            "negative": negative,
            "latent": latent,
        }
        
    except Exception as exc:
        log.warning("Failed to load conditioning cache for seg #%d: %s", segment_index + 1, exc)
        return None


def clear_conditioning_cache(node_id: str | None = None, segment_index: int | None = None) -> int:
    """Clear conditioning cache files.
    
    Returns number of files deleted.
    """
    try:
        cache_dir = _get_cache_dir(node_id)
        if not cache_dir.exists():
            return 0
        
        deleted = 0
        if segment_index is not None:
            # Clear specific segment
            for f in cache_dir.glob(f"seg_{segment_index:03d}_*.pt"):
                f.unlink()
                deleted += 1
        else:
            # Clear all
            for f in cache_dir.glob("seg_*.pt"):
                f.unlink()
                deleted += 1
        
        log.info("Cleared %d conditioning cache files", deleted)
        return deleted
        
    except Exception as exc:
        log.warning("Failed to clear conditioning cache: %s", exc)
        return 0


def get_cache_stats(node_id: str | None = None) -> dict:
    """Get statistics about the conditioning cache."""
    try:
        cache_dir = _get_cache_dir(node_id)
        if not cache_dir.exists():
            return {"exists": False, "files": 0, "total_size_mb": 0}
        
        files = list(cache_dir.glob("seg_*.pt"))
        total_size = sum(f.stat().st_size for f in files)
        
        return {
            "exists": True,
            "files": len(files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_dir": str(cache_dir),
        }
    except Exception:
        return {"exists": False, "files": 0, "total_size_mb": 0}


def clear_all_conditioning_cache() -> int:
    """Clear ALL conditioning cache files across all nodes.
    
    Returns total number of files deleted.
    """
    try:
        from folder_paths import output_directory
        
        base = Path(output_directory) / CACHE_SUBDIR
        if not base.exists():
            return 0
        
        deleted = 0
        for node_dir in base.iterdir():
            if node_dir.is_dir():
                for f in node_dir.glob("seg_*.pt"):
                    f.unlink()
                    deleted += 1
        
        log.info("Cleared ALL conditioning cache files: %d total", deleted)
        return deleted
        
    except Exception as exc:
        log.warning("Failed to clear all conditioning cache: %s", exc)
        return 0


def get_all_cache_stats() -> dict:
    """Get statistics about ALL conditioning cache files across all nodes."""
    try:
        from folder_paths import output_directory
        
        base = Path(output_directory) / CACHE_SUBDIR
        if not base.exists():
            return {"exists": False, "files": 0, "total_size_mb": 0, "nodes": 0}
        
        total_files = 0
        total_size = 0
        node_count = 0
        
        for node_dir in base.iterdir():
            if node_dir.is_dir():
                files = list(node_dir.glob("seg_*.pt"))
                total_files += len(files)
                total_size += sum(f.stat().st_size for f in files)
                if files:
                    node_count += 1
        
        return {
            "exists": True,
            "files": total_files,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "nodes": node_count,
            "cache_dir": str(base),
        }
    except Exception:
        return {"exists": False, "files": 0, "total_size_mb": 0, "nodes": 0}
