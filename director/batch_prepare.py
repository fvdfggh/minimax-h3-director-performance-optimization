"""Phase 1 (prepare): model-grouped encoding, before anything is sampled.

Moved out of :mod:`batch_executor` verbatim. The phase is split by *model*
because the official conditioning nodes touch CLIP, the video VAE and the audio
VAE inside a single call, which would keep all three resident at once:

    step 0  prepare    — resize / canvas / frame-grid alignment (no models)
    step 1  text       — CLIP tokenize+encode every prompt, then unload CLIP
    step 2  video vae  — encode every image/video pixel, then unload the VAE
    step 3  audio vae  — encode every soundtrack, then unload the audio VAE
    step 4  assemble   — attach the finished latent blocks to the conditioning

The split is only possible because the CLIP side consumes *pixels*: Qwen sees the
resized refs (``ref_items`` / ``images``), while the latents that ride in
``minimax_refs`` are produced afterwards by the VAEs from those same pixels.

``prepare_segment_materials`` also owns the R2V reference renumbering, so a
prompt's ``<Picture 3>`` tokens keep pointing at the material the user picked
even after some of them were filtered out.
"""

from __future__ import annotations

import logging
import re
from typing import Any

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.batch.prepare")


def _minimax_h3_official():
    """Official MiniMax H3 helpers, re-used so the batch path can't drift."""
    from comfy_extras.nodes_minimax_h3 import (
        MiniMaxH3ReferenceToVideo, _empty_av_latent, _resize, adapt_canvas,
    )
    return MiniMaxH3ReferenceToVideo, _empty_av_latent, _resize, adapt_canvas


def _rebuild_empty_latent(width, height, length):
    """Rebuild the all-zero AV canvas instead of persisting it.

    ``_empty_av_latent`` is a pure function of ``(width, height, length)`` and
    always returns ``torch.zeros``, so neither the durable conditioning cache
    nor the per-run scratch file stores it — both rebuild it here. That saves
    roughly 6 MB per segment per run of otherwise pointless I/O.
    """
    _cls, empty_av_latent, _resize, _adapt = _minimax_h3_official()
    latent, _frame_count = empty_av_latent(int(width), int(height), int(length))
    return latent


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


_REF_IMG_PREFIX = "ref_image_"


_REF_VID_PREFIX = "ref_video_"


_REF_AUD_PREFIX = "ref_audio_"


_REF_VAUD_PREFIX = "ref_video_audio_"


def _renumber_r2v_references(prompt, ref_images, ref_videos, ref_audios, ref_video_audios):
    """Renumber the actually-passed R2V materials to gap-free 1..N and rewrite the
    prompt tokens to match.

    A material's identity is its absolute index: ``<Picture 5>`` means absolute
    index 4. We sort the materials that are actually present, assign them
    contiguous ranks 0,1,2..., and rewrite every prompt token so ``<Picture K>``
    now points at the new rank of the material whose absolute index was ``K-1``.

    This guarantees MiniMax receives ``ref_image_0, ref_image_1, ...`` plus a
    prompt whose ``<Picture N>`` numbering is gap-free, regardless of which
    absolute ids the user typed in the prompt. Materials referenced in the prompt
    but absent from the payload are left untouched (so the model still surfaces
    them as missing).
    """
    if not prompt:
        return prompt, ref_images, ref_videos, ref_audios, ref_video_audios

    def rank_map(d, prefix):
        idxs = sorted(int(k[len(prefix):]) for k in (d or {}) if k.startswith(prefix))
        return {old: new for new, old in enumerate(idxs)}

    def renumber(d, prefix, rank):
        if not d:
            return d
        nd = {}
        for k, v in d.items():
            if k.startswith(prefix):
                old = int(k[len(prefix):])
                nd[f"{prefix}{rank.get(old, old)}"] = v
            else:
                nd[k] = v
        return nd

    def rewrite(text, token, rank):
        pat = re.compile(rf"<{token}\s+(\d+)\s*>", re.IGNORECASE)
        def repl(m):
            old = int(m.group(1))
            new = rank.get(old - 1, old - 1) + 1
            return f"<{token} {new}>"
        return pat.sub(repl, text)

    # images
    img_rank = rank_map(ref_images, _REF_IMG_PREFIX)
    if img_rank:
        ref_images = renumber(ref_images, _REF_IMG_PREFIX, img_rank)
        prompt = rewrite(prompt, "Picture", img_rank)
    # videos (+ paired audios, keyed by video index)
    vid_rank = rank_map(ref_videos, _REF_VID_PREFIX)
    if vid_rank:
        ref_videos = renumber(ref_videos, _REF_VID_PREFIX, vid_rank)
        if ref_video_audios:
            nd = {}
            for k, v in ref_video_audios.items():
                if k.startswith(_REF_VAUD_PREFIX):
                    old = int(k[len(_REF_VAUD_PREFIX):])
                    if old in vid_rank:
                        nd[f"{_REF_VAUD_PREFIX}{vid_rank[old]}"] = v
            ref_video_audios = nd
        prompt = rewrite(prompt, "Video", vid_rank)
    # audios
    aud_rank = rank_map(ref_audios, _REF_AUD_PREFIX)
    if aud_rank:
        ref_audios = renumber(ref_audios, _REF_AUD_PREFIX, aud_rank)
        prompt = rewrite(prompt, "Audio", aud_rank)

    return prompt, ref_images, ref_videos, ref_audios, ref_video_audios


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

    # Renumber the actually-passed materials to gap-free 1..N and rewrite the
    # prompt tokens so MiniMax's <Picture N> numbering lines up with the payload,
    # regardless of which absolute ids the user typed in the prompt.
    if task_key in {"r2v", "v2v", "rv2v"} and (
        ref_images or ref_videos or ref_audios or ref_video_audios
    ):
        prompt, ref_images, ref_videos, ref_audios, ref_video_audios = (
            _renumber_r2v_references(
                prompt, ref_images, ref_videos, ref_audios, ref_video_audios
            )
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
