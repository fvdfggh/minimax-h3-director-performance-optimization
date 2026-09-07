"""Batch mode executor: three-phase processing to minimize model loading.

Phase 1 (prepare):  Pre-encode text/audio/ref → disk
Phase 2 (sample):   UNet stays loaded, sequential sampling → latent to disk
Phase 3 (decode):   VAE stays loaded, sequential decode → mp4 export

Key constraint: segment N+1's conditioning depends on segment N's tail latent
(motion context), so sampling is sequential but models stay resident.
"""

from __future__ import annotations

import gc
import json
import logging
import os
from pathlib import Path
from typing import Any

import torch

from ..lib.image_prep import assert_minimax_canvas, fit_canvas, fit_video_long_edge
from ..lib.task_modes import SUPPORTED_TASK_KEYS
from . import cache_layout
from . import segment_slots
from .conditioning_cache import (
    clear_conditioning_cache,
    load_conditioning_cache,
    save_conditioning_cache,
    slugify_workflow_name,
    text_cache_key,
)
from .core_sampling import sample_single_stage
from .frame_align import (
    minimax_align_frame_count, minimax_phase_aligned_export_frames, pad_or_trim_frames,
)
from .audio_export import (
    AUDIO_MODE_GENERATE, AUDIO_MODE_MUTE, AUDIO_MODE_SOURCE,
    empty_audio_dict, resolve_audio_mode,
)
from .segment_runtime import frames_label, resolve_segment_raw_clip, segment_passthrough_chunk, tensor_frame_to_jpeg_b64
from .plan import (
    DirectorPlan, prepare_segment_clip, resolve_ref_image_size,
    ref_audios_to_dict, ref_video_audios_to_dict, ref_videos_to_dict,
    reference_video_for_segment,
    refs_to_kwargs_for_context, reinforce_r2v_prompt, reinforce_rv2v_prompt, reinforce_v2v_prompt,
)
from .progress import report_director_finish, report_director_progress, report_director_segment_preview
from .h3_motion_context import (
    DEFAULT_AUDIO_CONTEXT_FRAMES, apply_motion_context,
    generation_frame_budget, handoff_end_frame,
    resolve_tail_context_length,
    snap_context_frames, trim_context_prefix, trim_export_tail,
    video_from_latent,
)
from .segment_cache import (
    build_run_selection_clips, continuous_export_runs,
    load_next_segment_av_latent,
    load_segment_audio, load_segment_av_latent,
    load_segment_handoff_meta, probe_segment_cache_shape,
    save_segment_cache,
    sync_segment_slots,
)
from .segment_mp4_export import (
    maybe_export_segment_mp4, new_segment_mp4_run_dir, export_run_mp4, run_mp4_path,
)
from .segment_continuity import concat_chunks_lazy, is_continuity_active, resolve_prev_segment_output
from .vram_cleanup import cleanup_segment_vram

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.batch")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unpack_node_output(out):
    if hasattr(out, "args"):
        args = out.args
        if args:
            return args
    if isinstance(out, (tuple, list)):
        return out
    raise RuntimeError(f"Unexpected node output: {type(out)!r}")


def _decode_av_latent(samples, vae, audio_vae, *, decode_audio=True):
    from comfy_extras.nodes_lt import LTXVSeparateAVLatent
    from nodes import VAEDecode

    sep = LTXVSeparateAVLatent.execute(samples)
    video_latent, audio_latent = _unpack_node_output(sep)[:2]
    images, = VAEDecode().decode(vae, video_latent)
    if not decode_audio or audio_vae is None:
        return images, empty_audio_dict()
    try:
        from comfy_extras.nodes_audio import VAEDecodeAudio
    except ImportError:
        from comfy_extras.nodes_lt import VAEDecodeAudio
    audio_out = VAEDecodeAudio.execute(audio_vae, audio_latent)
    audio = _unpack_node_output(audio_out)[0]
    return images, audio


def _trim_decoded_to_export(decoded, audio_dict, *, trim_frames, export_len, plan):
    if trim_frames > 0:
        decoded, audio_dict = trim_context_prefix(
            decoded, audio_dict, trim_frames,
            fps=float(plan.frame_rate or 24), match_tail=True,
        )
    if decoded.shape[0] > export_len:
        decoded = decoded[:export_len]
        if isinstance(audio_dict, dict) and audio_dict.get("waveform") is not None:
            sr = int(audio_dict.get("sample_rate") or 32000)
            want = int(round((export_len / float(plan.frame_rate or 24)) * sr))
            wf = audio_dict["waveform"]
            if int(wf.shape[-1]) > want:
                audio_dict = {"waveform": wf[..., :want], "sample_rate": sr}
    return decoded, audio_dict


def _ref_tensor_from_seg_refs(refs, index):
    for ref in refs or []:
        if int(getattr(ref, "index", -1)) == index and ref.tensor is not None:
            t = ref.tensor
            if t.shape[0] > 0:
                return t[:1]
    return None


def _build_minimax_inputs(plan, seg, *, clip_frames, ctx_w, ctx_h, prev_tail):
    task_key = seg.task_key
    first_frame = last_frame = ref_images = ref_videos = ref_audios = ref_video_audios = None

    if task_key == "fl2v":
        first_frame = _ref_tensor_from_seg_refs(seg.refs, 0)
        last_frame = _ref_tensor_from_seg_refs(seg.refs, 1)
        if first_frame is not None and last_frame is None and clip_frames is not None:
            if clip_frames.shape[0] >= 2:
                last_frame = clip_frames[-1:].clone()
    elif task_key == "i2v":
        if clip_frames is not None and clip_frames.shape[0] > 0:
            first_frame = clip_frames[:1]
        else:
            first_frame = _ref_tensor_from_seg_refs(seg.refs, 0)
        del prev_tail
    elif task_key == "r2v":
        ref_kwargs = refs_to_kwargs_for_context(task_key, seg.refs)
        ref_images = {}
        for key, tensor in ref_kwargs.items():
            ref_images[key] = tensor
        ref_videos_dict = ref_videos_to_dict(getattr(seg, "ref_videos", None) or [])
        if ref_videos_dict:
            ref_videos = ref_videos_dict
        ref_audios_dict = ref_audios_to_dict(getattr(seg, "ref_audios", None) or [])
        if ref_audios_dict:
            ref_audios = ref_audios_dict
        # Read seg.ref_video_audios (SegmentRefVideo has no `.audio`) and emit the
        # official `ref_video_audio_<N>` keys, so soundtracks actually pair up.
        ref_video_audios = ref_video_audios_to_dict(getattr(seg, "ref_video_audios", None))
    elif task_key == "v2v":
        ref_videos_dict = ref_videos_to_dict(getattr(seg, "ref_videos", None) or [])
        if ref_videos_dict:
            ref_videos = ref_videos_dict
    elif task_key == "rv2v":
        ref_kwargs = refs_to_kwargs_for_context(task_key, seg.refs)
        ref_images = {}
        for key, tensor in ref_kwargs.items():
            ref_images[key] = tensor
        ref_videos_dict = ref_videos_to_dict(getattr(seg, "ref_videos", None) or [])
        if ref_videos_dict:
            ref_videos = ref_videos_dict
        ref_audios_dict = ref_audios_to_dict(getattr(seg, "ref_audios", None) or [])
        if ref_audios_dict:
            ref_audios = ref_audios_dict
        ref_video_audios = ref_video_audios_to_dict(getattr(seg, "ref_video_audios", None))

    return first_frame, last_frame, ref_images, ref_videos, ref_audios, ref_video_audios


# ---------------------------------------------------------------------------
# Cache paths for batch mode
# ---------------------------------------------------------------------------

def _batch_cache_dir(node_id: int, workflow_name: str | None = None) -> Path:
    """Directory holding this run's intermediates.

    Shares the one Director cache folder with the durable segment artefacts, so
    scratch files are marked ``seg_XXXX_scratch_<kind>.pt`` and told apart by
    that prefix alone — see :mod:`cache_layout`.

    Namespaced by workflow name just like the encoding cache: node ids are
    per-graph and get reused across workflow files, so without this two
    workflows would overwrite each other's scratched latents.

    File names stay fixed per segment, so a re-run overwrites in place rather
    than accumulating.
    """
    return cache_layout.node_cache_dir(str(node_id), workflow_name)


def _clear_batch_cache(cache_dir: Path, reports: list[str]) -> None:
    """Delete this run's scratch cache (Phase 1/2/3 intermediates).

    Scratch files now live beside the durable segment artefacts, so this removes
    only the ``seg_*_scratch_*`` files — never the whole directory. Deleting the
    directory would take the rendered frames / AV latents with it, which is what
    motion context and「全部导出」read back.

    When the option is off, files keep fixed names and are simply overwritten by
    the next run.
    """
    try:
        if not cache_dir.is_dir():
            reports.append("Batch cache: nothing to clear")
            return
        removed = 0
        failed = 0
        for path in cache_layout.iter_scratch_files(cache_dir):
            try:
                path.unlink()
                removed += 1
            except OSError:
                failed += 1
        if failed:
            reports.append(f"Batch cache: removed {removed} file(s), {failed} locked")
        else:
            reports.append(f"Batch cache cleared: {removed} scratch file(s) in {cache_dir}")
    except Exception as exc:  # never fail the run over cache cleanup
        log.warning("Director batch: failed to clear batch cache %s: %s", cache_dir, exc)
        reports.append(f"Batch cache: cleanup failed ({exc})")


# --------------------------------------------------------------------------
# Phase 1 (prepare) — model-grouped encoding
#
# The official conditioning nodes touch CLIP, the video VAE and the audio VAE
# inside a single call, so all three stay resident for the whole prepare loop.
# These helpers split that work by model instead:
#
#   step 0  prepare   — resize / canvas / frame-grid alignment (no models)
#   step 1  text      — CLIP tokenize+encode every prompt, then unload CLIP
#   step 2  video vae — encode every image/video pixel, then unload the VAE
#   step 3  audio vae — encode every soundtrack, then unload the audio VAE
#   step 4  assemble  — attach the finished latent blocks to the conditioning
#
# The split is only possible because the CLIP side consumes *pixels*: Qwen sees
# the resized refs (``ref_items`` / ``images``), while the latents that ride in
# ``minimax_refs`` are produced afterwards by the VAEs from those same pixels.
# --------------------------------------------------------------------------


def _minimax_h3_official():
    """Official MiniMax H3 helpers, re-used so the batch path can't drift."""
    from comfy_extras.nodes_minimax_h3 import (
        MiniMaxH3ReferenceToVideo, _empty_av_latent, _resize, adapt_canvas,
    )
    return MiniMaxH3ReferenceToVideo, _empty_av_latent, _resize, adapt_canvas


def _ref_audio_encode(audio_vae, audio):
    """``MiniMaxH3ReferenceToVideo._encode_ref_audio`` as a plain function."""
    cls, _unused_empty, _unused_resize, _unused_canvas = _minimax_h3_official()
    return cls._encode_ref_audio(audio_vae, audio)


def _job(media, container, key, raw=None):
    """Describe one VAE encode.

    ``cache_key`` is ``(id(raw), shape, dtype)``. In global edit mode every
    segment shares the *same* reference tensor (``list(global_refs)`` is a shallow
    copy), so identical media yields an identical key and the VAE runs once
    instead of once per segment. ``raw`` is pinned in the cache so ``id()`` cannot
    be recycled onto a different tensor while the key is still live.

    Audio arrives as a dict, so the tensor under ``"waveform"`` is what identifies
    it — two dicts wrapping the same waveform still dedupe to one encode.
    """
    tensor = media["waveform"] if isinstance(media, dict) else media
    rt = raw if raw is not None else media
    if isinstance(rt, dict):
        rt = rt.get("waveform")
    return {
        "pixel": media, "container": container, "key_name": key,
        "cache_key": (id(rt), tuple(tensor.shape), str(tensor.dtype)),
        "raw": rt,
    }


def _extract_referenced_picture_indices(prompt: str) -> set[int]:
    """Extract all <Picture N> indices referenced in the prompt.
    
    Searches for patterns like:
    - <Picture 1>, <Picture 2>, etc.
    - 图片1, 图片2, etc. (Chinese variant)
    
    Returns a set of 0-based indices (so <Picture 1> → {0}).
    """
    import re
    
    if not prompt:
        return set()
    
    indices = set()
    
    # English pattern: <Picture 1>, <Picture 2>, etc.
    for match in re.finditer(r'<Picture\s+(\d+)>', prompt):
        idx = int(match.group(1)) - 1  # Convert to 0-based
        if idx >= 0:
            indices.add(idx)
    
    # Chinese pattern: 图片1, 图片2, etc.
    for match in re.finditer(r'图片\s*(\d+)', prompt):
        idx = int(match.group(1)) - 1  # Convert to 0-based
        if idx >= 0:
            indices.add(idx)
    
    return indices


def _filter_ref_images_by_prompt(
    ref_images: dict[str, Any] | None,
    prompt: str,
) -> dict[str, Any] | None:
    """Filter reference images to only those referenced in the prompt.
    
    ref_images format: {"ref_image_0": tensor, "ref_image_1": tensor, ...}
    
    This reduces VRAM usage and encoding time by skipping unreferenced materials.
    """
    if not ref_images:
        return None
    
    referenced_indices = _extract_referenced_picture_indices(prompt)
    if not referenced_indices:
        # No references found in prompt — keep all for backward compatibility
        return ref_images
    
    filtered = {}
    for key, value in ref_images.items():
        if value is None:
            continue
        # Extract index from "ref_image_0" → 0
        try:
            idx_str = key.split("_")[-1]
            idx = int(idx_str)
            if idx in referenced_indices:
                filtered[key] = value
        except (ValueError, IndexError):
            # Malformed key — skip
            continue
    
    return filtered if filtered else None


def prepare_segment_materials(
    *, prompt, width, height, length, task_key,
    first_frame=None, last_frame=None, ref_images=None, ref_videos=None,
    ref_video_audios=None, ref_audios=None, ref_image_size="match",
):
    """Model-free half of the official conditioning nodes.

    Mirrors ``MiniMaxH3ImageToVideo.execute`` / ``MiniMaxH3ReferenceToVideo.execute``
    up to (but not including) the first model call. Returns a dict holding the
    empty AV latent, the CLIP-side pixel presentation, and two job queues for the
    VAE passes to fill in.
    """
    _cls, empty_av_latent, resize, adapt = _minimax_h3_official()

    ref_images = ref_images or {}
    ref_videos = ref_videos or {}
    ref_video_audios = ref_video_audios or {}
    ref_audios = ref_audios or {}
    
    # Filter ref_images to only those referenced in the prompt (R2V optimization)
    if task_key in {"r2v", "v2v", "rv2v"} and ref_images:
        orig_count = len(ref_images)
        filtered = _filter_ref_images_by_prompt(ref_images, prompt)
        if filtered:
            ref_images = filtered
            log.debug(
                "R2V ref_images filtered by prompt: %d → %d items",
                orig_count,
                len(filtered),
            )

    use_reference = (
        task_key in {"r2v", "v2v", "rv2v"}
        or bool(ref_images) or bool(ref_videos)
        or bool(ref_audios) or bool(ref_video_audios)
    )

    latent, frame_count = empty_av_latent(width, height, length)
    out = {
        "prompt": prompt,
        "task_key": task_key,
        "latent": latent,
        "frame_count": frame_count,
        "use_reference": use_reference,
        "images": [],       # CLIP image tokens (fl2v / i2v keyframes)
        "keyframes": [],    # DiT keyframe payloads, gain "latent" from the VAE
        "ref_items": [],    # tokenizer presentation (pixels only)
        "ref_blocks": [],   # DiT payload, gains "latent"/"audio_latent"
        "image_jobs": [],   # video-VAE jobs: _job(pixel, container, key, raw)
        "audio_jobs": [],   # audio-VAE jobs: _job(audio, container, key, raw)
    }

    if not use_reference:
        # --- MiniMaxH3ImageToVideo geometry -------------------------------------
        if first_frame is not None:
            img = resize(first_frame[:1], width, height, "disabled")
            out["images"].append(img)
            kf = {"resolved_frame_index": 0, "image": img}
            out["keyframes"].append(kf)
            out["image_jobs"].append(_job(img, kf, "latent", first_frame))
        if last_frame is not None:
            img = resize(last_frame[:1], width, height, "center")
            out["images"].append(img)
            kf = {"resolved_frame_index": frame_count - 1, "image": img}
            out["keyframes"].append(kf)
            out["image_jobs"].append(_job(img, kf, "latent", last_frame))
        return out

    # --- MiniMaxH3ReferenceToVideo geometry -----------------------------------
    # Order matters: <Picture i> images, then per-video <Audio j> + <Video k>,
    # then standalone <Audio j>. Duplicating the official iteration order keeps
    # the prompt tags aligned.
    from comfy_extras.nodes_minimax_h3 import CANVAS_MULTIPLE, FPS, REF_IMAGE_SHORT_EDGE
    import math

    for img in ref_images.values():
        if img is None:
            continue
        h, w = img.shape[1], img.shape[2]
        if ref_image_size == "match":
            scale = min(1.0, math.sqrt((width * height) / (w * h)))
        else:
            scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))
        tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        resized = resize(img[:1], tw, th, "disabled")
        blk = {"kind": "image", "latent_h": th // 16, "latent_w": tw // 16, "latent": None}
        out["ref_items"].append({"type": "image", "data": resized})
        out["ref_blocks"].append(blk)
        out["image_jobs"].append(_job(resized, blk, "latent", img))

    for name, video_frames in ref_videos.items():
        if video_frames is None:
            continue
        soundtrack = ref_video_audios.get("ref_video_audio_" + name.rsplit("_", 1)[-1])
        vh, vw = video_frames.shape[1], video_frames.shape[2]
        cw, ch = adapt(vw, vh)
        if vw * vh < cw * ch:
            cw = max(CANVAS_MULTIPLE, round(vw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            ch = max(CANVAS_MULTIPLE, round(vh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        frames = resize(video_frames, cw, ch, "disabled")
        if frames.shape[0] > frame_count:
            frames = frames[:frame_count]
        n = frames.shape[0]
        if n < 5:
            raise ValueError("MiniMax H3 reference videos need at least 5 frames (~0.2s at 24 fps)")
        while n % 17 != 5:
            n -= 1
        frames = frames[:n]

        blk = {
            "kind": None,               # resolved once the soundtrack is encoded
            "_has_soundtrack": soundtrack is not None,
            "latent_t": None, "latent_h": ch // 16, "latent_w": cw // 16,
            "ref_audio_t": 0, "latent": None, "audio_latent": None,
        }
        out["image_jobs"].append(_job(frames, blk, "latent", video_frames))

        if soundtrack is not None:
            ref_items_audio = {"type": "audio"}
            out["ref_items"].append(ref_items_audio)
            out["audio_jobs"].append(_job(soundtrack, blk, "audio_latent", soundtrack))

        # Qwen sees the video at 2 fps with timestamps
        sample_idx = list(range(0, frames.shape[0], FPS // 2))
        out["ref_items"].append({
            "type": "video", "data": frames[sample_idx],
            "timestamps": [i / 2.0 for i in range(len(sample_idx))],
        })
        out["ref_blocks"].append(blk)

    for audio in ref_audios.values():
        if audio is None:
            continue
        blk = {"kind": "audio", "ref_audio_t": 0, "audio_latent": None}
        out["ref_items"].append({"type": "audio"})
        out["ref_blocks"].append(blk)
        out["audio_jobs"].append(_job(audio, blk, "audio_latent", audio))

    return out


def encode_text_batch(clip, prepared: list[dict]) -> int:
    """Step 1 — CLIP: tokenize + encode every *distinct* prompt in one residency.

    Segments whose text-side inputs hash identically share a single encoding. The
    hash covers prompt, canvas, length, task key, reference size and the reference
    pixels, so two segments share only when their text inputs really are
    interchangeable — the expensive 32B Qwen prefill then runs once for the group
    instead of once per segment.

    Returns how many segments reused an encoding instead of paying for a new one.
    """
    # This path calls ``clip.tokenize`` directly and bypasses
    # ``run_minimax_conditioning``, so the ViT cache has to be installed here too
    # (idempotent, no-op if already installed).
    try:
        from . import vision_cache

        vision_cache.install()
    except Exception:  # pragma: no cover - optimisation only
        pass
    by_key: dict[str, Any] = {}
    reused = 0
    for item in prepared:
        key = item.get("text_key")
        if key is not None and key in by_key:
            # assemble_conditioning pairs this with the segment's own ref_blocks
            # and only shallow-copies, so sharing the tensor is safe.
            item["cond"] = by_key[key]
            item["text_reused"] = True
            reused += 1
            continue
        if item["use_reference"]:
            tokens = clip.tokenize(item["prompt"], minimax_ref_items=item["ref_items"])
        else:
            tokens = clip.tokenize(item["prompt"], images=item["images"])
        item["cond"] = clip.encode_from_tokens_scheduled(tokens)
        item["text_reused"] = False
        if key is not None:
            by_key[key] = item["cond"]
    return reused


def encode_video_vae_batch(vae, prepared: list[dict], cache: dict | None = None) -> int:
    """Step 2 — video VAE: encode every prepared image / video pixel.

    Reference media reused across segments (the usual case in global edit mode)
    is encoded once and shared via ``cache``. Latents are treated as read-only
    downstream — the DiT concatenates them — so sharing one tensor is safe and
    avoids both a repeat encode and a repeat GPU upload.
    """
    cache = {} if cache is None else cache
    done = 0
    for item in prepared:
        for job in item["image_jobs"]:
            ck = job["cache_key"]
            if ck in cache:
                job["container"][job["key_name"]] = cache[ck]
            else:
                z = vae.encode(job["pixel"])
                cache[ck] = z
                cache[(ck, "raw")] = job["raw"]   # pin: keep id() stable
                job["container"][job["key_name"]] = z
                done += 1
        # Downstream only needs the latent, so drop the pixel after encoding.
        for kf in item["keyframes"]:
            kf.pop("image", None)
        item["image_jobs"].clear()
    return done


def encode_audio_vae_batch(audio_vae, prepared: list[dict], cache: dict | None = None) -> int:
    """Step 3 — audio VAE: encode every prepared soundtrack.

    Shares results the same way as :func:`encode_video_vae_batch`.
    """
    cache = {} if cache is None else cache
    done = 0
    for item in prepared:
        for job in item["audio_jobs"]:
            ck = job["cache_key"]
            if ck in cache:
                z, t = cache[ck]
            else:
                z, t = _ref_audio_encode(audio_vae, job["pixel"])
                t = int(t)
                cache[ck] = (z, t)
                cache[(ck, "raw")] = job["raw"]
                done += 1
            job["container"][job["key_name"]] = z
            job["container"]["ref_audio_t"] = int(t)
        item["audio_jobs"].clear()
    return done


def assemble_conditioning(prepared: list[dict]) -> list[dict]:
    """Step 4 — attach the latent blocks to each conditioning payload."""
    from node_helpers import conditioning_set_values

    out = []
    for item in prepared:
        cond = item["cond"]
        for blk in item["ref_blocks"]:
            # A video block is labelled "video_audio" only once its soundtrack
            # actually produced latent frames.
            if blk["kind"] is None:
                blk["kind"] = "video_audio" if int(blk.get("ref_audio_t") or 0) > 0 else "video"
                blk.pop("_has_soundtrack", None)
            # Key-presence check, not a None check: only video blocks carry
            # latent_t, and an image block also has a non-None latent by now.
            if "latent_t" in blk and blk["latent_t"] is None:
                blk["latent_t"] = int(blk["latent"].shape[2])
        if item["ref_blocks"]:
            cond = conditioning_set_values(cond, {"minimax_refs": item["ref_blocks"]})
        if item["keyframes"]:
            # minimax_frame_count is the aligned *total* frame count, not the
            # number of keyframes — the DiT needs it to place each keyframe.
            cond = conditioning_set_values(cond, {
                "minimax_keyframes": item["keyframes"],
                "minimax_frame_count": item["frame_count"],
            })
        out.append({"positive": cond, "negative": [], "latent": item["latent"]})
    return out


def unload_model_group(*models, reports: list[str] | None = None, label: str = "") -> None:
    """Release a model group from VRAM before the next one is loaded.

    Only models that are actually resident are freed; ``soft_empty_cache`` then
    returns the freed blocks to the caching allocator.
    """
    import comfy.model_management as mm

    for m in models:
        if m is None:
            continue
        patcher = getattr(m, "patcher", None)
        if patcher is None:
            continue
        try:
            mm.unload_model_and_clones(patcher)
        except Exception as exc:  # unloading is an optimisation, never fatal
            log.warning("Director batch: failed to unload %s: %s", label or type(m).__name__, exc)
    mm.soft_empty_cache()
    if reports is not None and label:
        reports.append(f"  unloaded {label}")


def _latent_for_cache(node_id, seg_index, completed_av_latents, cache_dir):
    """Return the AV latent to persist into seg_cache, back-filling from disk.

    The next segment pins its motion context from seg_cache's av latent, so this
    must never return None for a segment we just decoded. When Phase 3 loaded the
    latent from the batch scratch dir (fallback path), it was never inserted into
    ``completed_av_latents`` — back-fill it so save_segment_cache persists it.

    This also covers the final segment: nothing follows it in this run, but a
    later run may append segments after it, and those will pin from here.
    """
    lat = completed_av_latents.get(seg_index)
    if lat is not None:
        return lat
    lat = _load_batch_latent(node_id, seg_index, cache_dir)
    if lat is not None:
        completed_av_latents[seg_index] = lat
    return lat


def _save_batch_conditioning(node_id, seg_index, positive, negative, latent, cache_dir):
    """Save pre-encoded conditioning to disk (per-run scratch)."""
    path = cache_layout.scratch_path(cache_dir, seg_index, "cond")
    torch.save({
        "positive": positive,
        "negative": negative,
        "latent": latent,
    }, path, _use_new_zipfile_serialization=True)
    return path


def _load_batch_conditioning(node_id, seg_index, cache_dir):
    """Load pre-encoded conditioning from disk (per-run scratch)."""
    path = cache_layout.scratch_path(cache_dir, seg_index, "cond")
    if not path.exists():
        return None
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data


def _save_batch_ref(node_id, seg_index, ref_data, cache_dir):
    """Save pre-processed reference data to disk (per-run scratch)."""
    path = cache_layout.scratch_path(cache_dir, seg_index, "ref")
    torch.save(ref_data, path, _use_new_zipfile_serialization=True)
    return path


def _load_batch_ref(node_id, seg_index, cache_dir):
    """Load pre-processed reference data from disk (per-run scratch)."""
    path = cache_layout.scratch_path(cache_dir, seg_index, "ref")
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def _save_batch_latent(node_id, seg_index, samples, cache_dir):
    """Persist the sampled AV latent to the durable, content-addressed segment cache.

    Writes ``seg_<hash>_latent.pt`` (the same slot that ``segment_cache.save_segment_cache``
    fills in Phase 3), NOT a per-run scratch file. This keeps a single on-disk copy of the
    latent and lets the next segment — same run or a later run — pin its motion context
    straight from the segment cache via ``load_segment_av_latent`` instead of from a scratch
    file that ``iter_scratch_files`` would delete.

    A latent-only file (no ``.meta.json`` yet) is deliberately *not* a "finished" segment:
    strict readers (``allow_stale=False``) still require the meta and ignore it, so it never
    spoofs a complete cache that continuity /「全部导出」would trust. ``save_segment_cache``
    in Phase 3 then completes the group with frames + meta.
    """
    stem = segment_slots.resolve_stem(cache_dir, seg_index)
    if not stem:
        log.warning("Director batch: no cache slot for segment %d, latent not persisted", seg_index + 1)
        return None
    path = cache_layout.segment_paths(cache_dir, stem)["latent"]
    # Move to CPU for disk storage
    cpu_samples = {}
    for k, v in samples.items():
        if isinstance(v, torch.Tensor):
            cpu_samples[k] = v.cpu()
        else:
            cpu_samples[k] = v
    torch.save(cpu_samples, path, _use_new_zipfile_serialization=True)
    return path


def _load_batch_latent(node_id, seg_index, cache_dir):
    """Load a sampled AV latent from the durable segment cache.

    Position-addressed (mirrors :func:`segment_cache.load_segment_av_latent`) and without the
    fingerprint gate, so a latent-only file written by ``_save_batch_latent`` in Phase 2 —
    before the meta exists — still loads. Used as the fallback when the stricter
    ``load_segment_av_latent`` did not return one.
    """
    stem = segment_slots.resolve_stem(cache_dir, seg_index)
    if not stem:
        return None
    path = cache_layout.segment_paths(cache_dir, stem)["latent"]
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def _av_latent_canvas_matches(av_latent, width: int, height: int) -> bool:
    """True when apply_motion_context would take its *latent* path for this AV latent.

    Must mirror the check inside apply_motion_context exactly, otherwise the two
    disagree: the probe would report "pinnable", Phase 2 would fall through to
    the pixel path, and resolve_prev_segment_output would then raise because
    that path needs a frame cache a latent-only re-run does not have.

    Hence the reuse of video_from_latent — the AV latent is stored under
    "samples" (a NestedTensor), not "video"; indexing it by hand is what made
    an earlier version of this return False unconditionally. Any failure here
    degrades to False (treat as unpinnable) rather than propagating, since the
    probe only needs a conservative answer.
    """
    try:
        src = video_from_latent(av_latent)
        return (
            int(src.shape[4]) * 16 == int(width)
            and int(src.shape[3]) * 16 == int(height)
        )
    except Exception:
        return False


def _prev_context_available(
    node_id,
    plan: DirectorPlan,
    all_segments: list,
    seg_index: int,
    run_indices: set,
    completed_av_latents: dict,
    width: int = 0,
    height: int = 0,
    workflow_name: str | None = None,
) -> bool:
    """Can segment ``seg_index`` pin its timeline predecessor?

    Phase 1 must already know: ``sample_len`` is baked into the conditioning
    here, so the motion-context decision cannot be revisited in Phase 2. Without
    this probe,「选择运行」with a gap (e.g. run seg 1, then seg 4) reaches
    ``apply_motion_context`` with nothing to pin and raises
    "need previous segment latent or decoded frames". Degrade to a no-pin
    segment there instead of failing the run.

    The predecessor's AV latent is reachable in one of two ways:
      * it is sampled earlier in this very run (``run_list`` keeps timeline
        order, so any predecessor inside ``run_indices`` comes first), or
      * it is already on disk from a previous run.

    On a disk hit the latent is kept in ``completed_av_latents`` — Phase 2 would
    load it anyway, so this costs no extra I/O.
    """
    prev_idx = int(seg_index) - 1
    if prev_idx < 0:
        return False
    if prev_idx in completed_av_latents:
        return True
    if prev_idx in run_indices:
        return True
    prev_seg = next((s for s in all_segments if s.index == prev_idx), None)
    if prev_seg is None:
        return False
    prev_av = load_segment_av_latent(node_id, prev_seg, plan, allow_stale=True, workflow_name=workflow_name)
    if prev_av is None:
        return False
    # A canvas mismatch would send apply_motion_context down its pixel path,
    # which now has nothing to pin (see _av_latent_canvas_matches).
    if width > 0 and height > 0 and not _av_latent_canvas_matches(prev_av, width, height):
        return False
    completed_av_latents[prev_idx] = prev_av
    return True


def _assemble_export_list(
    node_id,
    plan: DirectorPlan,
    all_segments: list,
    *,
    decoded_frames: dict[int, int],
    completed_audios: dict[int, dict],
    reports: list[str],
    workflow_name: str | None = None,
) -> tuple[list, list[dict], list[int], dict[int, torch.Tensor]]:
    """Timeline-ordered export list for「全部导出」.

    Batch mode only samples「选择运行」, but the merge must still cover the whole
    timeline — unselected slots are restored from the segment cache (exact
    fingerprint first, then stale) or a source passthrough.
    Returns ``(segments, audios, frame_counts, memory_chunks)``
    aligned 1:1.

    ``memory_chunks`` holds the passthrough fills: those exist **only** in RAM,
    so they must be handed to ``concat_chunks_lazy`` as overrides instead of
    being dropped — dropping them makes the merge fail with a cache miss on the
    very segment that was just accepted. Stale cache hits are not stored here;
    they stay on disk and the merge re-reads them with the same stale policy.
    """
    segments: list = []
    audios: list[dict] = []
    frame_counts: list[int] = []
    memory_chunks: dict[int, torch.Tensor] = {}
    skipped: list[int] = []

    for seg in all_segments:
        if seg.index in decoded_frames:
            segments.append(seg)
            frame_counts.append(int(decoded_frames[seg.index]))
            audios.append(completed_audios.get(seg.index) or {})
            continue

        # Probe the header only: this branch used to load the whole clip just to
        # read ``shape[0]`` and then throw the pixels away, so every unselected
        # segment was read twice per run (once here, once by the merge).
        cached_shape = probe_segment_cache_shape(node_id, seg, plan, workflow_name=workflow_name)
        used_stale = False
        if cached_shape is None:
            cached_shape = probe_segment_cache_shape(
                node_id, seg, plan, allow_stale=True, workflow_name=workflow_name
            )
            used_stale = cached_shape is not None
        if cached_shape is not None:
            n_frames = int(cached_shape[0])
            cached_audio = load_segment_audio(node_id, seg, plan, allow_stale=used_stale, workflow_name=workflow_name)
            if not isinstance(cached_audio, dict):
                cached_audio = {}
            segments.append(seg)
            frame_counts.append(n_frames)
            audios.append(cached_audio)
            reports.append(
                f"  Seg #{seg.index + 1}: cache fill for「全部导出」({n_frames}f"
                f"{', +audio' if cached_audio else ', no audio cache'}"
                f"{', stale fingerprint' if used_stale else ''})"
            )
            continue

        fill = segment_passthrough_chunk(plan, seg)
        if fill is None:
            skipped.append(seg.index + 1)
            reports.append(
                f"  Seg #{seg.index + 1}: skipped — no cache "
                "(outside run selection; omitted from「全部导出」merge)"
            )
            continue
        segments.append(seg)
        frame_counts.append(int(fill.shape[0]))
        audios.append({})
        # Not on disk — keep it alive for the merge (see docstring).
        memory_chunks[int(seg.index)] = fill
        reports.append(
            f"  Seg #{seg.index + 1}: source passthrough ({int(fill.shape[0])}f, "
            "not sampled — outside run selection)"
        )

    if skipped:
        reports.append(
            f"Omitted from「全部导出」(no cache): segment(s) {skipped} — "
            "勾选重跑或先全跑可补上。"
        )
    return segments, audios, frame_counts, memory_chunks


# ---------------------------------------------------------------------------
# Main batch executor
# ---------------------------------------------------------------------------

def execute_director_batch(
    plan: DirectorPlan,
    *,
    node_id: int,
    model,
    vae,
    audio_vae,
    clip,
    cfg: float,
    seed: int,
    steps: int,
    sampler: str,
    scheduler: str,
    shift_video: float,
    shift_audio: float,
    use_conditioning_cache: bool = False,
    clear_conditioning_cache_on_run: bool = False,
    clear_vram_between_segments: bool = True,
    workflow_name: str | None = None,
    progress_cb=None,
) -> tuple:
    """Three-phase batch execution.

    This is the only execution path. Returns
    ``(combined, segment_outputs, segment_audios, report, export_frame_counts)``.
    """
    all_segments = plan.segments
    # Reconcile cache files with the current timeline first: a group deleted in
    # the middle takes its own files, everyone else keeps their own render.
    sync_segment_slots(node_id, plan, workflow_name=workflow_name)
    # ``ext_meta`` is the raw externalGroups payload — a plain dict, NOT a plan.
    # Audio mode must be resolved from the plan, which owns raw["output"]["audioMode"].
    ext_meta = (plan.raw or {}).get("externalGroups") or {}
    audio_mode = resolve_audio_mode(plan)
    decode_audio = audio_mode == AUDIO_MODE_GENERATE
    raw_live = (plan.raw or {}).get("liveTaePreview", (plan.raw or {}).get("live_tae_preview", True))
    live_tae_preview = False if raw_live in (False, 0, "0", "false", "False", "off") else True

    # UI card count drives progress labels; timeline rows are the floor because
    # unselected slots still fill「全部导出」from cache/source.
    timeline_seg_total = len(all_segments)
    try:
        timeline_seg_total = max(timeline_seg_total, int(ext_meta.get("count") or 0))
    except (TypeError, ValueError):
        pass

    run_indices = set(plan.run_indices) if plan.run_indices is not None else set(range(len(all_segments)))

    # 「分段导出」: the checked segments already live in the cache, so there is
    # NOTHING to sample here. Sampling would re-run the UNet on the whole
    # timeline (and apply seam luma on every segment), which is exactly what an
    # export-only run must avoid. We sample zero segments; the export reads
    # frames / latents from disk and decodes with the already-loaded VAE.
    seg_export = getattr(plan, "segment_export", None)
    # Trigger on the one-shot ``enabled`` flag AND the checked ``indices``. The
    #「分段导出」button sets both; a plain 运行 must not export.
    #
    # ``indices`` alone is NOT a valid trigger: it is persisted in the timeline so
    # the picker reopens with the last selection, so once anything was ever checked
    # every subsequent run became an export-only pass (run_indices emptied → zero
    # sampling). The flag is safe to trust: ``app.queuePrompt`` is patched to flush
    # every Director timeline synchronously *before* the prompt is built, so the
    # widget snapshot always carries the flag as it was when the button was pressed.
    seg_export_active = (
        seg_export is not None and bool(seg_export.enabled) and bool(seg_export.indices)
    )
    if seg_export is not None and seg_export.indices and not seg_export.enabled:
        # Checked but not triggered: a normal generation run. Log it so an
        # accidental no-op export is visible rather than silent.
        log.info(
            "分段导出: %d 个片段已勾选，但未由「分段导出」按钮触发 — 按正常生成运行。",
            len(seg_export.indices),
        )
    if seg_export_active:
        run_indices = set()

    run_list = [seg for seg in all_segments if seg.index in run_indices]

    if not run_list and not seg_export_active:
        raise ValueError("Batch mode: no segments to run.")

    # Lives from Phase 1 on: the motion-context probe in Phase 1 may back-fill a
    # predecessor's AV latent here, which Phase 2 then reuses instead of re-reading.
    completed_av_latents: dict[int, dict] = {}

    reports = [
        f"Batch mode: {len(run_list)}/{timeline_seg_total} segments",
        f"Phase 1: Pre-encode all conditioning → disk",
        f"Phase 2: Sequential sampling (UNet stays loaded)",
        f"Phase 3: Sequential decode (VAE stays loaded)",
    ]
    if audio_mode == AUDIO_MODE_MUTE:
        reports.append("Audio: muted — skip audio VAE decode, silent AUDIO output.")
    elif audio_mode == AUDIO_MODE_SOURCE:
        reports.append("Audio: source — skip audio VAE decode, use original timeline audio.")
    else:
        reports.append("Audio: generate — decode MiniMax H3 AV latent audio.")

    cache_dir = _batch_cache_dir(node_id, workflow_name)

    # Report current memory
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        reports.append(f"GPU memory: {alloc:.1f}GB allocated, {reserved:.1f}GB reserved")

    # ===================================================================
    # PHASE 1: Pre-encode all conditioning → disk
    # ===================================================================
    # Opt-in wipe before anything is encoded: when a prompt/canvas changed but
    # the on-disk hash still matches an old variant, the run would otherwise be
    # silently served from that stale cache.
    if clear_conditioning_cache_on_run:
        try:
            removed = clear_conditioning_cache(node_id, workflow_name=workflow_name)
            reports.append(f"Conditioning cache cleared on run: {removed} file(s) removed")
        except Exception as exc:
            log.warning("Conditioning cache clear skipped (%s).", exc)

    reports.append("=" * 50)
    reports.append("PHASE 1: Preparing materials, then encoding by model group...")

    # Step 0 runs with no model resident: canvas fit, resize and frame-grid
    # alignment are plain tensor work. Only after every segment's pixels are
    # staged do we load CLIP, then the video VAE, then the audio VAE — one at a
    # time, released before the next is loaded.
    pending: list[dict] = []        # segments still needing an encoder
    pending_meta: list[dict] = []   # per-segment context for cache + reporting
    cache_hits = 0
    # Every text hash this run consumes — cache-served and freshly encoded alike.
    # Anything on disk outside this set is stale and gets pruned after Step 1.
    used_text_keys: set[str] = set()
    # Shared across all segments: in global edit mode every segment points at the
    # same reference tensors, so identical media encodes once instead of N times.
    vae_cache: dict = {}
    aud_cache: dict = {}

    for seg in run_list:
        ui_idx = seg.timeline_index
        seg_pos = run_list.index(seg)
        report_director_progress(
            node_id, segment_index=seg_pos, segment_total=len(run_list),
            phase="batch_prepare", phase_value=0, phase_max=1,
            frames_label=frames_label(seg), task_key=seg.task_key,
            timeline_segment_index=ui_idx, timeline_segment_total=timeline_seg_total,
        )

        target_len = max(1, int(seg.frame_count or plan.total_frames or 124))
        raw_clip = resolve_segment_raw_clip(plan, seg)

        if seg.source_clip is not None:
            body_raw = seg.source_clip
            target_len = max(target_len, int(body_raw.shape[0]))
        else:
            body_raw = raw_clip[:target_len] if int(raw_clip.shape[0]) > target_len else raw_clip

        if body_raw is not None and body_raw.shape[0] > 0:
            if plan.output_mode == "fixed":
                clip_frames = fit_canvas(body_raw, plan.width, plan.height)
            else:
                clip_frames = fit_video_long_edge(body_raw, plan.ref_max_size)
                if int(clip_frames.shape[1]) != int(plan.height) or int(clip_frames.shape[2]) != int(plan.width):
                    clip_frames = fit_canvas(clip_frames, plan.width, plan.height)
        else:
            clip_frames = None

        num_frames = minimax_align_frame_count(target_len)
        if clip_frames is not None:
            clip_frames, _ = prepare_segment_clip(clip_frames, num_frames)

        continuity_active = is_continuity_active(plan, seg)

        ctx_w = int(plan.width)
        ctx_h = int(plan.height)
        if clip_frames is not None and clip_frames.shape[0] > 0:
            ctx_h, ctx_w = int(clip_frames.shape[1]), int(clip_frames.shape[2])
        assert_minimax_canvas(ctx_w, ctx_h)

        positive_prompt = seg.prompt
        if seg.task_key == "fl2v":
            from .fl2v_timeline import reinforce_fl2v_prompt
            has_start = any(getattr(r, "index", None) == 0 for r in (seg.refs or []))
            has_end = any(getattr(r, "index", None) == 1 for r in (seg.refs or []))
            if not has_start and not has_end and seg.refs:
                has_start = True
                has_end = len(seg.refs) >= 2
            positive_prompt = reinforce_fl2v_prompt(positive_prompt, has_end_frame=has_end, has_start_frame=has_start)
        elif seg.task_key == "r2v":
            ref_idxs = [int(getattr(r, "index", 0)) for r in (seg.refs or []) if r is not None]
            vid_idxs = [int(getattr(v, "index", 0)) for v in (getattr(seg, "ref_videos", None) or []) if v is not None]
            audio_idxs = [int(getattr(a, "index", 0)) for a in (seg.ref_audios or []) if a is not None]
            positive_prompt = reinforce_r2v_prompt(positive_prompt, ref_indices=ref_idxs, video_indices=vid_idxs, audio_indices=audio_idxs)
        elif seg.task_key == "v2v":
            positive_prompt = reinforce_v2v_prompt(positive_prompt)
        elif seg.task_key == "rv2v":
            ref_idxs = [int(getattr(r, "index", 0)) for r in (seg.refs or []) if r is not None]
            audio_idxs = [int(getattr(a, "index", 0)) for a in (seg.ref_audios or []) if a is not None]
            positive_prompt = reinforce_rv2v_prompt(positive_prompt, ref_indices=ref_idxs, audio_indices=audio_idxs)

        # Build inputs (without prev_tail — we don't have it yet in phase 1)
        first_frame, last_frame, ref_images, ref_videos, ref_audios, ref_video_audios = _build_minimax_inputs(
            plan, seg, clip_frames=clip_frames, ctx_w=ctx_w, ctx_h=ctx_h, prev_tail=None,
        )

        i2v_new_anchor = seg.task_key == "i2v" and first_frame is not None
        # Probe availability here, not in Phase 2: sample_len is baked into the
        # conditioning below, so the pin decision cannot be revised later.
        prev_context_available = _prev_context_available(
            node_id, plan, all_segments, seg.index, run_indices, completed_av_latents,
            width=ctx_w, height=ctx_h, workflow_name=workflow_name,
        )
        use_motion_context = (
            continuity_active
            and not i2v_new_anchor
            and seg.index > 0
            and prev_context_available
        )
        if use_motion_context:
            first_frame = None  # context owns the head
        if continuity_active and seg.index > 0 and not use_motion_context:
            reports.append(
                f"  Seg #{seg.index + 1}: 无上段可引用（前段未采样且无缓存），"
                "本段不继承运动——请按顺序重跑该段以恢复接缝连贯。"
            )

        # Depends on use_motion_context: pinning extends the sample by context_n.
        context_n = (
            snap_context_frames(plan.continuity_overlap_frames) if use_motion_context else 0
        )
        sample_len, _planned_trim = generation_frame_budget(num_frames, context_n)

        if seg.task_key in {"r2v", "v2v", "rv2v"} and (ref_images or ref_videos or ref_audios or ref_video_audios) and audio_vae is None:
            raise ValueError("r2v/v2v/rv2v requires audio_vae input.")

        # Text-side identity of this segment. Segments sharing a key produce
        # interchangeable encodings, so the expensive Qwen prefill runs once per
        # distinct key rather than once per segment.
        ref_image_size = resolve_ref_image_size(seg, plan)
        text_key = text_cache_key(
            positive_prompt, ctx_w, ctx_h, sample_len, seg.task_key,
            ref_image_size, ref_images,
        )
        used_text_keys.add(text_key)

        # Try conditioning cache first
        cached_conditioning = None
        if use_conditioning_cache:
            cached_conditioning = load_conditioning_cache(
                node_id=node_id, segment_index=seg.index, prompt=positive_prompt,
                width=ctx_w, height=ctx_h, length=sample_len, task_key=seg.task_key,
                ref_image_size=ref_image_size, ref_images=ref_images,
                workflow_name=workflow_name,
            )

        # Ref data feeds Phase 2; write it before any cache short-circuit.
        ref_data = {
            "first_frame": first_frame,
            "last_frame": last_frame,
            "ref_images": ref_images,
            "ref_videos": ref_videos,
            "ref_audios": ref_audios,
            "ref_video_audios": ref_video_audios,
            "clip_frames": clip_frames,
            "num_frames": num_frames,
            "target_len": target_len,
            "sample_len": sample_len,
            "ctx_w": ctx_w,
            "ctx_h": ctx_h,
            "positive_prompt": positive_prompt,
            "context_n": context_n,
            "use_motion_context": use_motion_context,
        }
        _save_batch_ref(node_id, seg.index, ref_data, cache_dir)

        if cached_conditioning is not None:
            # Cache hit: nothing to encode, so this segment never touches a model.
            _save_batch_conditioning(
                node_id, seg.index,
                cached_conditioning["positive"], cached_conditioning["negative"],
                cached_conditioning["latent"], cache_dir,
            )
            reports.append(f"  Seg #{seg.index + 1}: conditioning CACHE HIT")
            cache_hits += 1
            continue

        # Stage the pixels now; the encoders consume them in the grouped passes.
        staged = prepare_segment_materials(
            prompt=positive_prompt, width=ctx_w, height=ctx_h,
            length=sample_len, task_key=seg.task_key,
            first_frame=first_frame, last_frame=last_frame,
            ref_images=ref_images, ref_videos=ref_videos,
            ref_video_audios=ref_video_audios, ref_audios=ref_audios,
            ref_image_size=ref_image_size,
        )
        staged["text_key"] = text_key
        pending.append(staged)
        pending_meta.append({
            "seg": seg, "positive_prompt": positive_prompt,
            "ctx_w": ctx_w, "ctx_h": ctx_h, "sample_len": sample_len,
            "ref_images": ref_images, "ref_image_size": ref_image_size,
            "text_key": text_key,
        })

    # ---- Step 1: text encoder -------------------------------------------------
    if pending:
        n_deduped = len(pending) - len(used_text_keys)
        reports.append(
            f"  text encoder: {len(used_text_keys)} encode(s) for {len(pending)} segment(s)"
            + (f", {n_deduped} deduped" if n_deduped > 0 else "")
            + (f", {cache_hits} cache hit(s)" if cache_hits else "")
        )
        n_reused = encode_text_batch(clip, pending)
        if n_reused:
            reports.append(f"  text encoder: {n_reused} segment(s) shared an encoding")
        unload_model_group(clip, reports=reports, label="text encoder (CLIP)")

        # ---- Step 2: video VAE ----------------------------------------------
        # One cache across every segment: shared reference media is encoded once.
        n_img_jobs = sum(len(p["image_jobs"]) for p in pending)
        n_aud_jobs = sum(len(p["audio_jobs"]) for p in pending)
        n_img = encode_video_vae_batch(vae, pending, vae_cache)
        if n_img:
            unload_model_group(vae, reports=reports, label="video VAE")
        shared = n_img_jobs - n_img
        reports.append(f"  video VAE: {n_img} encode(s)"
                       + (f", {shared} shared" if shared else ""))

        # ---- Step 3: audio VAE ----------------------------------------------
        n_aud = (encode_audio_vae_batch(audio_vae, pending, aud_cache)
                 if audio_vae is not None else 0)
        if n_aud:
            unload_model_group(audio_vae, reports=reports, label="audio VAE")
        shared = n_aud_jobs - n_aud
        reports.append(f"  audio VAE: {n_aud} encode(s)"
                       + (f", {shared} shared" if shared else ""))

        # ---- Step 4: assemble + persist --------------------------------------
        for prepared, meta in zip(pending, pending_meta):
            seg = meta["seg"]
            result = assemble_conditioning([prepared])[0]
            positive, negative, latent = result["positive"], result["negative"], result["latent"]
            _save_batch_conditioning(node_id, seg.index, positive, negative, latent, cache_dir)
            if use_conditioning_cache and not prepared.get("text_reused"):
                # Only one segment per key writes the shared encoding; the
                # dedupe path above skips the redundant rewrite.
                save_conditioning_cache(
                    node_id=node_id, segment_index=seg.index,
                    positive=positive, negative=negative, latent=latent,
                    prompt=meta["positive_prompt"], width=meta["ctx_w"], height=meta["ctx_h"],
                    length=meta["sample_len"], task_key=seg.task_key,
                    ref_image_size=meta["ref_image_size"], ref_images=meta["ref_images"],
                    workflow_name=workflow_name,
                )
            reports.append(f"  Seg #{seg.index + 1}: conditioning encoded → disk")
            del positive, negative, latent
    elif cache_hits:
        reports.append(f"  all {cache_hits} segment(s) served from conditioning cache")

    # NOTE: unused conditioning files are NOT pruned automatically after a run.
    # The "Clear cache" button in the UI handles cleanup on demand
    # (minimax_clear_cache → clear_conditioning_cache), which is sufficient.

    # Release the staged pixels/captions; Phase 2 only needs the latents now.
    pending.clear()
    pending_meta.clear()
    gc.collect()

    reports.append(f"Phase 1 complete: {len(run_list)} segments prepared")
    gc.collect()

    # ===================================================================
    # PHASE 2: Sequential sampling (UNet stays loaded)
    # ===================================================================
    reports.append("=" * 50)
    reports.append("PHASE 2: Sequential sampling with UNet resident...")

    completed_av_handoff: dict[int, dict] = {}
    completed_audios: dict[int, dict] = {}
    # Phase-align gap per segment: how many trailing frames must be dropped from
    # the *previous* export before concat (h3_motion_context.apply_motion_context
    # returns it as prev_export_trim_tail). Applied in Phase 3 once the previous
    # segment has been decoded.
    pending_prev_trim: dict[int, int] = {}
    # Highest timeline index; its tail must never be phase-trimmed.
    last_timeline_index = max((s.index for s in all_segments), default=None)

    # new_segment_mp4_run_dir already returns None unless export_mode == "segments".
    # Do not invert the condition here: both branches would yield None and the
    # per-segment mp4 would never be written.
    mp4_run_dir = new_segment_mp4_run_dir(plan)
    if mp4_run_dir is not None:
        reports.append(f"Segment mp4 export dir: {mp4_run_dir}")

    for seg in run_list:
        ui_idx = seg.timeline_index
        seg_pos = run_list.index(seg)

        report_director_progress(
            node_id, segment_index=seg_pos, segment_total=len(run_list),
            phase="batch_sample", phase_value=0, phase_max=1,
            frames_label=frames_label(seg), task_key=seg.task_key,
            timeline_segment_index=ui_idx, timeline_segment_total=timeline_seg_total,
        )

        # Load pre-encoded conditioning from disk
        cond_data = _load_batch_conditioning(node_id, seg.index, cache_dir)
        if cond_data is None:
            raise RuntimeError(f"Batch mode: conditioning cache miss for segment {seg.index}")
        positive = cond_data["positive"]
        negative = cond_data["negative"]
        latent = cond_data["latent"]
        del cond_data

        # Load ref data from disk
        ref_data = _load_batch_ref(node_id, seg.index, cache_dir)
        if ref_data is None:
            raise RuntimeError(f"Batch mode: ref cache miss for segment {seg.index}")

        sample_len = ref_data["sample_len"]
        num_frames = ref_data["num_frames"]
        target_len = ref_data["target_len"]
        ctx_w = ref_data["ctx_w"]
        ctx_h = ref_data["ctx_h"]
        use_motion_context = ref_data["use_motion_context"]
        context_n = ref_data["context_n"]

        # Load previous segment's tail for motion context
        trim_frames = 0
        if use_motion_context and seg.index > 0:
            prev_idx = seg.index - 1
            # Motion context always reads the *timeline* predecessor (seg.index - 1).
            # Lookup order: this run's memory -> seg_cache av latent -> durable segment
            # cache left by an earlier run. The last hop keeps a partial「选择运行」
            # working even when the predecessor was never re-sampled this run.
            prev_av = completed_av_latents.get(prev_idx)
            if prev_av is None:
                prev_seg = next((s for s in all_segments if s.index == prev_idx), None)
                if prev_seg is not None:
                    prev_av = load_segment_av_latent(node_id, prev_seg, plan, allow_stale=True, workflow_name=workflow_name)
            if prev_av is None:
                prev_av = _load_batch_latent(node_id, prev_idx, cache_dir)
                if prev_av is not None:
                    log.info(
                        "Director batch: seg #%d uses seg #%d latent from the durable segment cache; "
                        "if its prompt changed, re-run that segment to refresh.",
                        seg.index + 1, prev_idx + 1,
                    )
            if prev_av is not None:
                completed_av_latents[prev_idx] = prev_av

            # Pixel tail is a fallback, not a prerequisite: apply_motion_context
            # prefers context_latent and then never touches context_frames.
            # Resolving it eagerly made a missing *frame* cache fatal even though
            # the latent was sitting right there — resolve_prev_segment_output
            # raises whenever the predecessor is unavailable and continuity is on.
            # Only needed when the latent is missing, or when its canvas does not
            # match and the pin would fall through to the pixel path anyway.
            prev_tail = None
            if prev_av is None or not _av_latent_canvas_matches(prev_av, ctx_w, ctx_h):
                # Passed an empty dict on purpose: Phase 2 runs entirely before Phase 3,
                # so no segment is decoded to pixels yet — the predecessor's tail can
                # only come from the segment cache on disk.
                prev_tail = resolve_prev_segment_output(
                    plan, all_segments, seg.index, {}, node_id, workflow_name=workflow_name
                )

            prev_audio = completed_audios.get(prev_idx)
            if prev_audio is None:
                prev_seg = next((s for s in all_segments if s.index == prev_idx), None)
                if prev_seg is not None:
                    prev_audio = load_segment_audio(node_id, prev_seg, plan, allow_stale=True, workflow_name=workflow_name)
                    if prev_audio is not None:
                        completed_audios[prev_idx] = prev_audio

            prev_handoff = completed_av_handoff.get(prev_idx)
            if prev_handoff is None:
                prev_seg = next((s for s in all_segments if s.index == prev_idx), None)
                if prev_seg is not None:
                    prev_handoff = load_segment_handoff_meta(node_id, prev_seg, plan, allow_stale=True, workflow_name=workflow_name)
                    if prev_handoff is not None:
                        completed_av_handoff[prev_idx] = prev_handoff

            prev_end_frame = None
            if prev_handoff:
                prev_end_frame = handoff_end_frame(
                    trim_frames=int(prev_handoff.get("trim_frames") or 0),
                    export_frames=int(prev_handoff.get("export_frames") or 0),
                )
                sample_f = int(prev_handoff.get("sample_frames") or 0)
                if sample_f > 0 and prev_end_frame >= sample_f:
                    prev_end_frame = None

            pin_audio = audio_mode != AUDIO_MODE_MUTE and (prev_av is not None or prev_audio is not None)
            # ------------------------------------------------------------------
            # 对齐下段 (align-to-next): pin the next segment's opening into this
            # segment's tail. Cache-driven middle-out mode — only runs when that
            # neighbour already holds an AV latent, otherwise it silently no-ops.
            # ------------------------------------------------------------------
            tail_context_latent = None
            tail_context_length = 0
            if bool(getattr(seg, "continuity_to_next", False)):
                tail_n = resolve_tail_context_length(
                    latent, int(seg.frame_count), context_n=context_n
                )
                if tail_n > 0:
                    next_latent = load_next_segment_av_latent(node_id, seg.index, workflow_name=workflow_name)
                    if next_latent is None:
                        log.info(
                            "Seg #%d: 对齐下段 skipped — next segment has no cached AV latent.",
                            seg.index + 1,
                        )
                    else:
                        tail_context_latent = next_latent
                        tail_context_length = tail_n
                        log.info(
                            "Seg #%d: 对齐下段 active — pinning next segment head %df.",
                            seg.index + 1,
                            tail_n,
                        )
            positive, trim_frames, prev_export_trim = apply_motion_context(
                positive, latent, vae=vae,
                context_length=context_n,
                context_latent=prev_av,
                context_frames=prev_tail,
                context_audio=prev_audio,
                audio_vae=audio_vae,
                continue_audio=pin_audio,
                keep_existing_keyframes=(seg.task_key == "fl2v"),
                context_end_frame=prev_end_frame,
                audio_context_length=DEFAULT_AUDIO_CONTEXT_FRAMES,
                tail_context_latent=tail_context_latent,
                tail_context_length=tail_context_length,
            )
            if prev_export_trim > 0:
                pending_prev_trim[seg.index] = int(prev_export_trim)
            reports.append(
                f"  Seg #{seg.index + 1}: motion context ON — pin {trim_frames}f from seg #{seg.index}"
                + (f", prev tail trim {prev_export_trim}f" if prev_export_trim else "")
            )

        # UNet sampling
        def _report_sample_phase(phase, value):
            report_director_progress(
                node_id, segment_index=seg_pos, segment_total=len(run_list),
                phase=phase, phase_value=value, phase_max=1,
                frames_label=frames_label(seg), task_key=seg.task_key,
                timeline_segment_index=ui_idx, timeline_segment_total=timeline_seg_total,
            )

        def _report_step_preview(step: int, total_steps: int, x0) -> None:
            # Live frame for the batch-card preview slot.
            try:
                from .tae_preview import pil_to_jpeg_b64, x0_to_preview_pil
                pil = x0_to_preview_pil(x0, max_side=512)
                if pil is None:
                    return
                report_director_segment_preview(
                    node_id,
                    segment_index=ui_idx,
                    image_b64=pil_to_jpeg_b64(pil),
                    width=pil.width,
                    height=pil.height,
                    live=True,
                    step=step + 1,
                    total_steps=total_steps,
                )
            except Exception as exc:
                log.debug("Live TAE preview skipped: %s", exc)

        samples = sample_single_stage(
            model=model, positive=positive, negative=negative,
            latent=latent, seed=seed, cfg=cfg, steps=steps,
            sampler_name=sampler, scheduler=scheduler,
            shift_video=shift_video, shift_audio=shift_audio,
            on_phase=_report_sample_phase,
            on_step_preview=_report_step_preview if live_tae_preview else None,
            preview_every=1,
        )

        # Save AV latent to disk
        _save_batch_latent(node_id, seg.index, samples, cache_dir)

        # Build handoff for next segment.
        # export_frames must be the real export length, not num_frames: the next
        # segment derives its pin end limit from (trim + export_frames), and that
        # end must sit on the 17-frame VAE cycle grid or the pin window stops
        # short — the remainder is then either trimmed (lost) or replayed as a
        # seam echo. See minimax_phase_aligned_export_frames.
        export_len = (
            minimax_phase_aligned_export_frames(num_frames)
            if trim_frames > 0
            else int(num_frames)
        )
        handoff = {
            "trim_frames": int(trim_frames),
            "export_frames": int(export_len),
            "sample_frames": int(sample_len),
            "official_mc_length": False,
        }
        completed_av_latents[seg.index] = samples
        completed_av_handoff[seg.index] = handoff

        # Free conditioning from GPU (UNet stays!)
        del positive, negative, latent, samples
        if clear_vram_between_segments and torch.cuda.is_available():
            torch.cuda.empty_cache()

        reports.append(
            f"  Seg #{seg.index + 1}: sampled {sample_len}f → latent saved to disk"
        )

    reports.append(f"Phase 2 complete: {len(run_list)} latents saved")
    gc.collect()

    # ===================================================================
    # PHASE 3: Sequential decode (VAE stays loaded)
    # ===================================================================
    # Unload UNet to free VRAM for VAE (UNet ~20GB + VAE ~5GB > 22GB GPU)
    reports.append("=" * 50)
    reports.append("PHASE 3: Sequential decoding with VAE resident...")
    cleanup_segment_vram(enabled=True, unload_models=True)
    reports.append("  UNet unloaded, loading VAE...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        reports.append(f"  VRAM after UNet unload: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    segment_outputs: list[torch.Tensor] = []
    segment_audios: list[dict[str, Any]] = []
    export_frame_counts: list[int] = []
    # Per-segment bookkeeping keyed by timeline index, so the merge can walk the
    # whole timeline (including slots this run did not sample) without index math.
    # Frames are NOT kept here — only lengths — to avoid pinning every chunk.
    decoded_segments: list = []
    decoded_frames: dict[int, int] = {}
    # Only the immediate previous export is kept: the next iteration may need to
    # trim its tail (phase-align), and holding every chunk would double peak RAM.
    prev_export_seg = None
    prev_export_chunk: torch.Tensor | None = None
    prev_export_audio: dict | None = None
    # segment_outputs is ordered by run_list position; map timeline index -> pos
    # so a tail trim can patch the already-finished previous segment in place.
    run_pos_map: dict[int, int] = {}

    for seg in run_list:
        ui_idx = seg.timeline_index
        seg_pos = run_list.index(seg)

        report_director_progress(
            node_id, segment_index=seg_pos, segment_total=len(run_list),
            phase="batch_decode", phase_value=0, phase_max=1,
            frames_label=frames_label(seg), task_key=seg.task_key,
            timeline_segment_index=ui_idx, timeline_segment_total=timeline_seg_total,
        )

        # Use AV latent from Phase 2 (in memory) instead of reloading from disk
        samples = completed_av_latents.get(seg.index)
        if samples is None:
            # Fallback: load from disk. Back-fill it too — save_segment_cache below
            # writes the latent into seg_cache, and the next segment pins from
            # there rather than from this dict.
            samples = _load_batch_latent(node_id, seg.index, cache_dir)
            if samples is not None:
                completed_av_latents[seg.index] = samples
        if samples is None:
            raise RuntimeError(f"Batch mode: latent cache miss for segment {seg.index}")

        # No manual device move here: ``samples["samples"]`` is a
        # comfy.nested_tensor.NestedTensor, which is NOT a torch.Tensor subclass,
        # so an isinstance guard silently skipped it. ComfyUI's VAEDecode already
        # places inputs on the right device internally.

        # Load ref data for trim/export info
        ref_data = _load_batch_ref(node_id, seg.index, cache_dir)
        num_frames = ref_data["num_frames"]
        target_len = ref_data["target_len"]
        context_n = ref_data["context_n"]
        use_motion_context = ref_data["use_motion_context"]
        del ref_data

        # Determine trim_frames / export_len from the Phase 2 handoff: that is
        # the same pair the next segment uses to place its pin window, so the two
        # must never disagree. Recompute only when the handoff is missing.
        handoff = completed_av_handoff.get(seg.index, {})
        trim_frames = int(handoff.get("trim_frames") or 0)
        export_len = int(handoff.get("export_frames") or 0)
        if export_len <= 0:
            export_len = (
                minimax_phase_aligned_export_frames(num_frames)
                if trim_frames > 0
                else int(num_frames)
            )

        # VAE decode
        decoded, audio_dict = _decode_av_latent(samples, vae, audio_vae, decode_audio=decode_audio)
        del samples

        decoded, audio_dict = _trim_decoded_to_export(
            decoded, audio_dict, trim_frames=trim_frames, export_len=export_len, plan=plan,
        )

        chunk = decoded.cpu().float()

        # The head/tail window and the segment's clip.mp4 are written together by
        # save_segment_cache, so a piecewise export is byte-equivalent to this
        # trimmed chunk — no re-decode, no re-trim, and no chance of the two
        # files describing different renders.
        save_segment_cache(
            node_id, seg, plan, chunk,
            av_latent=_latent_for_cache(node_id, seg.index, completed_av_latents, cache_dir),
            handoff=handoff,
            audio=audio_dict if isinstance(audio_dict, dict) else None,
            workflow_name=workflow_name,
        )

        # Export mp4
        if mp4_run_dir is not None:
            mp4_path = maybe_export_segment_mp4(
                mp4_run_dir, plan, seg, chunk,
                audio_dict if isinstance(audio_dict, dict) else None,
            )
            if mp4_path:
                reports.append(f"  Segment {ui_idx + 1}/{timeline_seg_total}: mp4 → {mp4_path}")

        # Preview
        if seg.task_key in {"t2v", "i2v", "r2v", "fl2v", "v2v", "rv2v"} and chunk.shape[0] >= 1:
            try:
                frames_b64 = [tensor_frame_to_jpeg_b64(chunk[i]) for i in range(min(int(chunk.shape[0]), 10))]
                h, w = int(chunk.shape[1]), int(chunk.shape[2])
                report_director_segment_preview(
                    node_id, segment_index=ui_idx,
                    image_b64=frames_b64[0], width=w, height=h,
                    frames=frames_b64, fps=float(plan.frame_rate or 24),
                )
            except Exception as exc:
                log.debug("Segment preview skipped: %s", exc)

        run_pos_map[seg.index] = len(segment_outputs)
        segment_outputs.append(chunk)
        audio_out = audio_dict if isinstance(audio_dict, dict) else {}
        segment_audios.append(audio_out)
        decoded_segments.append(seg)
        decoded_frames[seg.index] = int(chunk.shape[0])
        completed_audios[seg.index] = audio_out
        export_frame_counts.append(int(chunk.shape[0]))

        # Phase-align orphaned tail: drop gap_after_pin frames from the *previous*
        # export so this segment's pin window abuts it exactly. Skipping this
        # replays the gap at every seam (visible stutter) and leaves the merge
        # longer by (seams x gap).
        # With the 17-frame-grid export length above, gap_after_pin is normally 0
        # and this whole block is dead code. It stays as a safety net for the
        # edges where the pin window can still fall short (pixel-pin fallback,
        # non-standard context lengths, hand-off from an older cache).
        # ``pending_prev_trim[seg.index]`` must be applied to timeline segment
        # ``seg.index - 1``.  ``prev_export_seg`` is merely the previous *iterated*
        # segment, so with a sparse「选择运行」(e.g. 0,3,7) it would shave segment
        # 0's tail to pay for segment 3's gap.  Only trim when the predecessor was
        # actually decoded in this run.
        pending_trim = int(pending_prev_trim.get(seg.index) or 0)
        if (
            pending_trim > 0
            and prev_export_seg is not None
            and prev_export_seg.index != seg.index - 1
        ):
            log.warning(
                "Director batch: skip %df phase-align trim for seg #%d — "
                "timeline predecessor #%d was not decoded in this run "
                "(its cache is untrimmed, so this seam keeps the gap).",
                pending_trim, seg.index + 1, seg.index,
            )
            pending_trim = 0
        # Never trim a segment with no successor on the timeline. With no following
        # segment to pin it, those trailing frames are genuine content rather than
        # a handoff gap, so dropping them would be a real loss. The final segment's
        # AV latent is still persisted above, so a later run can append a segment
        # after it and pin normally.
        if (
            pending_trim > 0
            and prev_export_seg is not None
            and last_timeline_index is not None
            and prev_export_seg.index >= last_timeline_index
        ):
            log.warning(
                "Director batch: skip %df phase-align trim on seg #%d — it is the "
                "last timeline segment, so its tail is content, not a gap.",
                pending_trim, prev_export_seg.index + 1,
            )
            pending_trim = 0
        if pending_trim > 0 and prev_export_seg is not None and prev_export_chunk is not None:
            fps = float(plan.frame_rate or 24)
            if int(prev_export_chunk.shape[0]) > pending_trim:
                new_chunk, new_audio = trim_export_tail(
                    prev_export_chunk, prev_export_audio, pending_trim, fps=fps
                )
                prev_export_chunk = new_chunk
                prev_export_audio = new_audio
                decoded_frames[prev_export_seg.index] = int(new_chunk.shape[0])
                new_audio_dict = new_audio if isinstance(new_audio, dict) else {}
                completed_audios[prev_export_seg.index] = new_audio_dict
                # Patch the already-finished lists (trimmed chunk must replace
                # the untrimmed one already appended above).
                run_pos = run_pos_map.get(prev_export_seg.index)
                if run_pos is not None and run_pos < len(segment_outputs):
                    segment_outputs[run_pos] = new_chunk
                    segment_audios[run_pos] = new_audio_dict
                    export_frame_counts[run_pos] = int(new_chunk.shape[0])
                # concat_chunks_lazy reads from disk — the cache must match.
                # save_segment_cache rewrites the head/tail window *and* the
                # segment's clip.mp4, so both follow the shortened export.
                ph = dict(completed_av_handoff.get(prev_export_seg.index) or {})
                ph["export_frames"] = int(new_chunk.shape[0])
                ph["phase_align_trim"] = pending_trim
                completed_av_handoff[prev_export_seg.index] = ph
                # Re-persist with the same AV latent: the rewritten export must
                # not lose the latent the following segment pins from.
                save_segment_cache(
                    node_id, prev_export_seg, plan, new_chunk,
                    av_latent=_latent_for_cache(
                        node_id, prev_export_seg.index, completed_av_latents, cache_dir
                    ),
                    handoff=ph,
                    audio=new_audio if isinstance(new_audio, dict) else None,
                    replace_audio=False,
                    workflow_name=workflow_name,
                )
                if mp4_run_dir is not None:
                    maybe_export_segment_mp4(
                        mp4_run_dir, plan, prev_export_seg, new_chunk,
                        new_audio_dict,
                    )
                reports.append(
                    f"  Seg #{prev_export_seg.index + 1}: phase-align trim — dropped "
                    f"{pending_trim}f orphaned tail (export {int(new_chunk.shape[0])}f)"
                )
                del new_chunk
            else:
                log.warning(
                    "Director batch: skip %df phase-align trim on seg #%d "
                    "(%df export is too short).",
                    pending_trim, prev_export_seg.index + 1,
                    int(prev_export_chunk.shape[0]),
                )

        prev_export_seg = seg
        prev_export_chunk = chunk
        prev_export_audio = audio_out

        # Free decoded frames from GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        reports.append(
            f"  Seg #{seg.index + 1}: decoded {chunk.shape[0]}f → mp4 exported"
            f" (VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB)"
        )

    reports.append(f"Phase 3 complete: {len(run_list)} segments decoded and exported")

    # Free AV latents (no longer needed after decode)
    completed_av_latents.clear()
    completed_av_handoff.clear()
    gc.collect()

    # ===================================================================
    # MERGE
    # ===================================================================
    report_director_finish(node_id, len(run_list))

    # Trigger on the checked indices (``seg_export_active``), consistent with the
    # Phase-1 decision above — NOT on ``seg_export.enabled``. The picker clears
    # ``enabled`` right after queueing, so a run that already skipped sampling
    # (run_indices = set()) used to fall through to the「全部导出」branch here and
    # crash on the unbound ``combined``.
    # ``merge_done`` marks a branch that already produced ``combined``; the generic
    # merge below then stands down instead of stitching the timeline a second time.
    merge_done = False
    if seg_export_active:
        # 「分段导出」: only the checked segments participate — no full-timeline
        # merge. This keeps an export-only run fast (it never renders or encodes
        # the other 30+ segments) and returns a combined clip of just the checked
        # segments as the node output; the per-segment mp4 files come from
        # ``run_segment_export`` below.
        #
        # Unified decode first: latent-only segments (no frame cache) are decoded
        # once and written back to ``seg_XXXX.pt``, so ``_load_segment_export_source``
        # below reads frames directly — no repeated per-consumer VAE decode.
        from .segment_cache import predecode_latent_segments as _predecode

        _predecode(
            node_id,
            plan,
            list(seg_export.indices),
            vae=(vae, audio_vae) if (vae is not None or audio_vae is not None) else None,
            workflow_name=workflow_name,
        )
        wanted_segs = [s for s in all_segments if int(s.index) in set(seg_export.indices)]
        export_segments_list, segment_audios, export_frame_counts, merge_overrides = (
            _assemble_export_list(
                node_id, plan, wanted_segs,
                decoded_frames=decoded_frames,
                completed_audios=completed_audios,
                reports=reports,
                workflow_name=workflow_name,
            )
        )
        # 「分段导出」must NOT merge the checked segments into one video on the
        # images output — each checked segment (piecewise) / each contiguous run
        # (continuous) is a separate clip. Route through the segments layout so
        # the node emits one IMAGE per clip instead of a stitched merge. The
        # per-segment mp4 files come from ``run_segment_export`` below.
        plan.export_mode = "segments"
        # ``export_segments_list`` is a list of SegmentPlan objects (not frames).
        # Load each checked segment's frames from cache / latent for the images
        # output, reusing the exact source the export uses so IMAGE == mp4.
        from .segment_cache import _load_segment_export_source as _seg_src

        seg_by_index = {int(s.index): s for s in wanted_segs}
        frame_by_index = {}
        audio_by_index = {}
        for idx in sorted(set(int(i) for i in seg_export.indices)):
            s = seg_by_index.get(idx)
            if s is None:
                continue
            src = _seg_src(node_id, s, plan, vae=(vae, audio_vae) if (vae is not None or audio_vae is not None) else None, workflow_name=workflow_name)
            if src is not None:
                frame_by_index[idx] = src[0]
                audio_by_index[idx] = src[1] if isinstance(src[1], dict) else {}
        # ``segment_audios`` from the export list is aligned with
        # ``export_segments_list``; fall back to it when the frame source carried
        # no audio (e.g. a stale frame cache without an audio sidecar).
        for s, a in zip(export_segments_list or [], segment_audios or []):
            idx = int(getattr(s, "index", -1))
            if idx >= 0 and not audio_by_index.get(idx) and isinstance(a, dict):
                audio_by_index[idx] = a

        _seg_mode = seg_export.normalized_mode()
        log.info(
            "分段导出: mode=%s checked=%s frames_loaded=%s",
            _seg_mode, sorted(set(int(i) for i in seg_export.indices)), sorted(frame_by_index),
        )
        if _seg_mode == "continuous":
            # 连续导出: one clip per contiguous run (stitched with the streaming
            # merge), standalone for a checked segment with no checked neighbour —
            # matching exactly what ``run_segment_export`` writes to disk.
            #
            # NOTE: only *timeline-adjacent* checked segments are stitched.
            # Checking #1,#5,#9,#13 yields four standalone clips by design —
            # the run breakdown is logged below so this is never a silent guess.
            from .segment_cache import continuous_export_runs as _runs
            from .segment_cache import merge_run_audio as _merge_audio

            outputs: list[torch.Tensor] = []
            counts: list[int] = []
            run_audios: list[dict] = []
            _all_runs = _runs(frame_by_index.keys())
            log.info("分段导出 连续导出: %d run(s) → %s", len(_all_runs), _all_runs)
            for run in _all_runs:
                run = [i for i in run if i in frame_by_index]
                if not run:
                    continue
                if len(run) == 1:
                    outputs.append(frame_by_index[run[0]])
                    counts.append(int(frame_by_index[run[0]].shape[0]))
                    run_audios.append(audio_by_index.get(run[0]) or {})
                    continue
                run_counts = [int(frame_by_index[i].shape[0]) for i in run]
                try:
                    merged = concat_chunks_lazy(
                        node_id,
                        plan,
                        [seg_by_index[i] for i in run],
                        overrides={i: frame_by_index[i] for i in run},
                        workflow_name=workflow_name,
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    log.warning(
                        "分段导出 连续导出: images merge of #%d–#%d failed (%s); "
                        "falling back to standalone clips.",
                        run[0] + 1, run[-1] + 1, exc,
                    )
                    for i in run:
                        outputs.append(frame_by_index[i])
                        counts.append(int(frame_by_index[i].shape[0]))
                        run_audios.append(audio_by_index.get(i) or {})
                    continue
                log.info(
                    "分段导出 连续导出: merged #%d–#%d (%df + %df + … → %df)",
                    run[0] + 1, run[-1] + 1, run_counts[0],
                    run_counts[1] if len(run_counts) > 1 else 0,
                    int(merged.shape[0]),
                )
                outputs.append(merged)
                counts.append(int(merged.shape[0]))
                run_audios.append(
                    _merge_audio(plan, [audio_by_index.get(i) or {} for i in run], run_counts)
                    or {}
                )
            log.info(
                "分段导出 连续导出: %d clip(s) on images output (frame counts %s)",
                len(outputs), counts,
            )
            segment_outputs = outputs
            export_frame_counts = counts
            segment_audios = run_audios
        else:
            ordered = sorted(frame_by_index)
            segment_outputs = [frame_by_index[i] for i in ordered]
            export_frame_counts = [int(t.shape[0]) for t in segment_outputs]
            # Keep audio 1:1 with the frames actually emitted (a segment whose
            # frames could not be loaded must not shift the alignment).
            segment_audios = [audio_by_index.get(i) or {} for i in ordered]
    elif run_list:
        # 「选择运行」in batch mode: export only what actually ran — one clip per
        # contiguous run, stitched exactly like「分段导出」continuous mode. No more
        # splicing unselected slots back from cache / source into a single merged
        # clip that the selection was supposed to avoid.
        from .segment_cache import build_run_selection_clips

        mp4_run_dir = new_segment_mp4_run_dir(plan, for_selection=True)
        seg_outputs, run_audios, seg_counts, run_mp4s = build_run_selection_clips(
            node_id, plan, [seg.index for seg in run_list], segment_outputs, segment_audios,
            all_segments=all_segments, mp4_run_dir=mp4_run_dir,
            workflow_name=workflow_name,
        )
        if not seg_outputs:
            raise ValueError("Batch mode: export list is empty.")
        if run_mp4s:
            reports.append(
                "选择运行导出: " + ", ".join(f"#{p.split('seg_')[-1]}" for p in run_mp4s)
            )
        else:
            reports.append(
                "选择运行导出: "
                + ", ".join(f"#{run[0] + 1}-{run[-1] + 1}" for run in continuous_export_runs([seg.index for seg in run_list]))
            )
        segment_outputs = seg_outputs
        export_frame_counts = seg_counts
        segment_audios = run_audios
        merge_overrides = None
        # The runs are already stitched by build_run_selection_clips above. Bind
        # ``combined`` to the first clip and clear ``export_segments_list`` so the
        # generic merge below is skipped — otherwise the whole timeline was stitched
        # a SECOND time (the duplicated "additive opening luma" seams) and the run
        # paid for two full merges.
        combined = seg_outputs[0]
        export_segments_list = None
        merge_done = True
    elif plan.export_mode == "all" or (
        not run_list
        and seg_export is not None
        and bool(seg_export.enabled)
        and bool(seg_export.indices)
    ):
        # Batch mode only samples「选择运行」—「全部导出」still needs the whole
        # timeline, so unselected slots are filled from cache / source.
        # (An export-only run with every selected segment cached has an empty
        # ``run_list``; we still assemble the full merged clip from cache so the
        # node returns a valid output.)
        # Index them by seg.index: run_list positions are not contiguous, so the
        # old `all_segments[run_list[0].index + i]` mapped to the wrong segment
        # whenever the selection had gaps.
        export_segments_list, segment_audios, export_frame_counts, merge_overrides = (
            _assemble_export_list(
                node_id, plan, all_segments,
                decoded_frames=decoded_frames,
                completed_audios=completed_audios,
                reports=reports,
                workflow_name=workflow_name,
            )
        )
    else:
        export_segments_list = list(decoded_segments)
        export_frame_counts = [int(t.shape[0]) for t in segment_outputs]
        merge_overrides = None

    # Under「全部导出」the node emits the merged clip only — every segment frame
    # is re-read from disk by the merge, so the in-memory chunks are now a second
    # full copy of the same video. Release them *before* allocating the merge
    # result: keeping both is what pushed peak RAM to 2x the final video (the
    # long-timeline OOM).
    if plan.export_mode == "all":
        del segment_outputs[:]
        gc.collect()

    log.info("分段导出: export_mode=%r seg_export_active=%s export_segments_list_len=%d segment_outputs_len=%d",
             plan.export_mode, seg_export_active,
             len(export_segments_list) if export_segments_list else 0,
             len(segment_outputs) if segment_outputs else 0)
    # Streaming merge: one allocation for the result, one copy-in per segment.
    # Every branch must bind ``combined`` — the old ``pass`` on the「全部导出」
    # path left it unbound and killed the node with UnboundLocalError.
    if merge_done:
        # Already stitched (e.g. the「选择运行」runs above). Merging a second time
        # duplicated every seam pass and doubled the merge cost.
        pass
    elif seg_export_active and plan.export_mode == "segments":
        # 分段导出: the split layout emits every clip straight from
        # ``segment_outputs`` and never reads ``combined`` — a second merge would
        # just double peak RAM. Bind it to a real clip (no placeholder) so the
        # slot always carries actual frames.
        if segment_outputs:
            combined = segment_outputs[0]
        else:
            hh = int(getattr(plan, "height", 480) or 480)
            ww = int(getattr(plan, "width", 864) or 864)
            combined = torch.zeros((1, hh, ww, 3), dtype=torch.float32)
        merge_overrides = None
    elif not export_segments_list:
        # 「分段导出」of latent-only segments (no .pt) has nothing to merge here —
        # the per-segment files are still produced by run_segment_export below.
        # Emit a 1-frame placeholder so the node never fails on an empty list.
        if not seg_export_active:
            raise ValueError("Batch mode: export list is empty.")
        hh = int(getattr(plan, "height", 480) or 480)
        ww = int(getattr(plan, "width", 864) or 864)
        combined = torch.zeros((1, hh, ww, 3), dtype=torch.float32)
        merge_overrides = None
    else:
        combined = concat_chunks_lazy(
            node_id, plan, export_segments_list, overrides=merge_overrides,
            workflow_name=workflow_name,
        )
        merge_overrides = None

    # 「分段导出」: the user checked segments + a mode. The VAE is loaded here
    # (and the clip cache was just written during decode), so latent-only
    # segments can be decoded on demand. Best-effort: a failed export only logs.
    seg_export = getattr(plan, "segment_export", None)
    if seg_export is not None:
        log.info(
            "分段导出 config: enabled=%s mode=%s indices=%s",
            seg_export.enabled,
            seg_export.mode,
            seg_export.indices,
        )
    # Node-output-only export: the merged video is already on ``combined`` (see the
    # seg_export_active branch above). We intentionally do NOT write files to disk —
    # the user wants the result directly on the node's OUTPUT slot, same as a normal
    # run. Disk export via run_segment_export is disabled on purpose.
    if seg_export is not None and seg_export.enabled:
        log.info(
            "分段导出 -> 节点输出模式: mode=%s indices=%s (不写磁盘)",
            seg_export.normalized_mode(),
            seg_export.indices,
        )

    # Scratch intermediates (seg_*_scratch_*.pt) live in the same flat node cache dir
    # as the durable segments and are cleaned via _clear_batch_cache / iter_scratch_files,
    # driven by the node's 「清空缓存」button rather than a per-run flag, so they survive
    # failed runs for debugging.

    # Now — and only now — advance the cache generation: this run has just written
    # fresh renders, so the file groups it superseded are expendable. The pre-run
    # sync and every read-only HTTP probe leave them in place, otherwise re-wording
    # a prompt would delete the last render before a replacement exists.
    try:
        sync_segment_slots(node_id, plan, workflow_name=workflow_name, gc=True)
    except Exception as exc:  # pragma: no cover - GC is best-effort
        log.warning("Segment cache cleanup after batch run skipped (%s).", exc)

    # Drop per-run scratch intermediates (seg_*_scratch_*.pt, e.g. the position-
    # numbered seg_0000_scratch_ref.pt) now that the run finished cleanly. These
    # are regenerate-in-place working state — NOT the content-hash durable files
    # (seg_<hash>_*.pt) that「分段导出」/ motion-context read back — so keeping
    # them around only duplicates data on disk and clutters the cache dir. Failures
    # above skip this block (via the early raises) and leave scratch intact for
    # debugging, which is the intended behaviour.
    try:
        _clear_batch_cache(cache_dir, reports)
    except Exception as exc:  # pragma: no cover - cleanup is best-effort
        log.warning("Batch scratch cleanup skipped (%s).", exc)

    return (
        combined,
        segment_outputs,
        segment_audios,
        "\n".join(reports),
        export_frame_counts,
    )
