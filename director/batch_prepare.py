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
reference pixels (``ref_items`` / ``images``) while the latents that ride in
``minimax_refs`` are produced afterwards by the VAEs. Reference *images* reach
Qwen at **native resolution** — Qwen's own preprocessing resizes and patches
them, so fitting them to the canvas here would only discard detail and pin the
ViT cache key to the canvas size — whereas the VAE still encodes the
canvas-fitted pixels.

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


def _job(media, container, key, raw=None, address=None, ref=None):
    """Describe one VAE encode.

    ``cache_key`` is ``(id(raw), shape, dtype)``. In global edit mode every
    segment shares the *same* reference tensor (``list(global_refs)`` is a shallow
    copy), so identical media yields an identical key and the VAE runs once
    instead of once per segment. ``raw`` is pinned in the cache so ``id()`` cannot
    be recycled onto a different tensor while the key is still live.

    Audio arrives as a dict, so the tensor under ``"waveform"`` is what identifies
    it — two dicts wrapping the same waveform still dedupe to one encode.

    ``address`` is where the encode lives in :mod:`ref_latent_cache`; ``ref`` is
    how to re-derive that address later at a *different* canvas (source hash +
    variant). ``ref`` is stamped onto the container so a cached text encoding can
    carry it instead of a second copy of the latent.
    """
    tensor = media["waveform"] if isinstance(media, dict) else media
    rt = raw if raw is not None else media
    if isinstance(rt, dict):
        rt = rt.get("waveform")
    if ref is not None:
        container["_reflat" if key == "latent" else "_reflat_audio"] = ref
    return {
        "pixel": media, "container": container, "key_name": key,
        "cache_key": (id(rt), tuple(tensor.shape), str(tensor.dtype)),
        "raw": rt,
        "address": address,
    }


def _audio_track(audio):
    """The tensor the audio VAE actually encodes (``None``-safe)."""
    if isinstance(audio, dict):
        return audio.get("waveform")
    return audio


def _reflat_ref(
    raw,
    canvas,
    variant: str,
    *,
    kind: str = "video",
    ref_image_size: str = "",
    frames: int | None = None,
) -> dict:
    """``address`` / ``ref`` kwargs for :func:`_job`, or ``{}`` when caching is off.

    The **address** is where this run's VAE encode is stored; the **reference** is
    what a cached text encoding keeps in its place. The reference names the source
    media and the variant but *not* the canvas — the reader supplies its own — so
    one text encoding can resolve to the 768p latent or the 1080p one.

    ``frames`` has to ride along because a reference clip is trimmed to the
    segment's frame count, which no reader can re-derive from the canvas.
    """
    from . import ref_latent_cache

    if raw is None or not ref_latent_cache.enabled():
        return {}
    seed = ref_latent_cache.seed(raw)
    ref = {"kind": kind, "seed": seed, "variant": variant}
    if frames is not None:
        ref["frames"] = int(frames)
    return {
        "address": ref_latent_cache.address_of_seed(
            kind, seed, canvas,
            variant=variant, ref_image_size=ref_image_size, frames=frames,
        ),
        "ref": ref,
    }


def _extract_referenced_token_indices(prompt: str, token: str) -> set[int]:
    """Extract the 0-based indices referenced in ``prompt`` as ``<Token N>``.

    ``token`` is ``"Picture"``, ``"Video"`` or ``"Audio"``. Returns a set of
    0-based indices, so ``<Picture 1>`` -> ``{0}``.

    Only the official MiniMax spelling counts — ``<Picture 1>``, case-insensitive
    and tolerant of inner spaces. Aliases (``图片1``) are deliberately *not*
    recognised: they are not the syntax the model reads, and matching them would
    silently reference a material the prompt never actually named.
    """
    if not prompt:
        return set()

    indices = set()
    for match in re.finditer(rf"<{token}\s+(\d+)\s*>", prompt, re.IGNORECASE):
        idx = int(match.group(1)) - 1  # Convert to 0-based
        if idx >= 0:
            indices.add(idx)
    return indices


def _filter_refs_by_prompt(
    refs: dict[str, Any] | None,
    prompt: str,
    prefix: str,
    token: str,
) -> dict[str, Any] | None:
    """Keep exactly the reference materials the prompt names — nothing else.

    ``refs`` format: ``{"<prefix><N>": tensor, ...}`` (e.g. ``ref_video_2``).

    A material counts as used only when the prompt names it with the official
    ``<Token N>`` tag (case-insensitive). Anything the prompt does not name is
    dropped — including the whole kind when it names none of it, because an
    un-referenced material must not influence the run. The returned dict may
    therefore be empty.

    Returns ``None`` only when there is nothing to filter (``refs`` empty), so
    the caller can keep its own default.
    """
    if not refs:
        return None

    referenced_indices = _extract_referenced_token_indices(prompt, token)

    filtered = {}
    for key, value in refs.items():
        if value is None or not key.startswith(prefix):
            continue
        # Extract index from "ref_video_2" -> 2
        try:
            idx = int(key[len(prefix):])
        except (ValueError, IndexError):
            # Malformed key — skip
            continue
        if idx in referenced_indices:
            filtered[key] = value

    return filtered


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
    
    # Drop every reference material the prompt does not name. This is both the
    # optimisation (video / audio VAE + ViT passes are expensive) and the
    # semantic contract: only material the prompt references may reach the model.
    # A kind the prompt never references is dropped entirely — see
    # _filter_refs_by_prompt. ``drops`` carries what was ignored so the caller can
    # surface it in the run report.
    drops: list[dict[str, Any]] = []

    def _keep(kind_refs, prefix, token):
        if not kind_refs:
            return kind_refs
        filtered = _filter_refs_by_prompt(kind_refs, prompt, prefix, token)
        if filtered is None:
            return kind_refs
        dropped = len(kind_refs) - len(filtered)
        if dropped:
            drops.append({"token": token, "dropped": dropped, "kept": len(filtered)})
            log.info(
                "R2V ref_%s filtered by prompt: %d -> %d items",
                token.lower(),
                len(kind_refs),
                len(filtered),
            )
        return filtered

    if task_key in {"r2v", "v2v", "rv2v"}:
        ref_images = _keep(ref_images, _REF_IMG_PREFIX, "Picture")
        ref_videos = _keep(ref_videos, _REF_VID_PREFIX, "Video")
        ref_audios = _keep(ref_audios, _REF_AUD_PREFIX, "Audio")
        if ref_video_audios:
            # Paired soundtracks are keyed by their video's absolute index, so they
            # follow the <Video K> tags exactly: none referenced -> none kept.
            vid_ref = _extract_referenced_token_indices(prompt, "Video")
            kept = {}
            for key, value in ref_video_audios.items():
                if not key.startswith(_REF_VAUD_PREFIX):
                    continue
                try:
                    vid_idx = int(key[len(_REF_VAUD_PREFIX):])
                except (ValueError, IndexError):
                    continue
                if vid_idx in vid_ref:
                    kept[key] = value
            dropped = len(ref_video_audios) - len(kept)
            ref_video_audios = kept
            if dropped:
                drops.append({"token": "VideoAudio", "dropped": dropped, "kept": len(kept)})

    if drops and not (ref_images or ref_videos or ref_audios or ref_video_audios):
        log.warning(
            "R2V: prompt references no <Picture>/<Video>/<Audio> tag; dropped all "
            "%d un-referenced reference material(s) for this segment.",
            sum(d["dropped"] for d in drops),
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
        "ref_drops": drops, # un-referenced material ignored, for the run report
    }

    if not use_reference:
        # --- MiniMaxH3ImageToVideo geometry -------------------------------------
        if first_frame is not None:
            img = resize(first_frame[:1], width, height, "disabled")
            out["images"].append(img)
            kf = {"resolved_frame_index": 0, "image": img}
            out["keyframes"].append(kf)
            out["image_jobs"].append(_job(
                img, kf, "latent", first_frame,
                **_reflat_ref(first_frame, (width, height), "first_frame"),
            ))
        if last_frame is not None:
            # Centre-cropped, where the first frame is stretched: same source
            # image as both keyframes must not land on one cache entry.
            img = resize(last_frame[:1], width, height, "center")
            out["images"].append(img)
            kf = {"resolved_frame_index": frame_count - 1, "image": img}
            out["keyframes"].append(kf)
            out["image_jobs"].append(_job(
                img, kf, "latent", last_frame,
                **_reflat_ref(last_frame, (width, height), "last_frame"),
            ))
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
        # Qwen's own preprocessing (``process_qwen2vl_images``) resizes and
        # patches whatever it is handed, so the ViT gets the *native* reference
        # pixels. The canvas fit above exists for the VAE (its latent must sit on
        # the canvas grid); reusing it for Qwen would just throw detail away and
        # tie the ViT cache key to the canvas size.
        out["ref_items"].append({"type": "image", "data": img[:1, ..., :3]})
        out["ref_blocks"].append(blk)
        out["image_jobs"].append(_job(
            resized, blk, "latent", img,
            **_reflat_ref(
                img, (width, height), "ref_image", ref_image_size=ref_image_size,
            ),
        ))

    for name, video_frames in ref_videos.items():
        if video_frames is None:
            continue
        soundtrack = ref_video_audios.get("ref_video_audio_" + name.rsplit("_", 1)[-1])
        # Trim to the model's 17k+5 grid first: the frame count follows from the
        # clip and the segment length, never from the canvas.
        n = video_frames.shape[0]
        if n > frame_count:
            n = frame_count
        if n < 5:
            raise ValueError("MiniMax H3 reference videos need at least 5 frames (~0.2s at 24 fps)")
        while n % 17 != 5:
            n -= 1
        vh, vw = video_frames.shape[1], video_frames.shape[2]
        cw, ch = adapt(vw, vh)
        if vw * vh < cw * ch:
            cw = max(CANVAS_MULTIPLE, round(vw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            ch = max(CANVAS_MULTIPLE, round(vh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        # The VAE gets the canvas fit — that latent rides in the DiT pack at
        # every step, so it is the half that has to sit on the canvas grid.
        frames = resize(video_frames[:n], cw, ch, "disabled")

        blk = {
            "kind": None,               # resolved once the soundtrack is encoded
            "_has_soundtrack": soundtrack is not None,
            "latent_t": None, "latent_h": ch // 16, "latent_w": cw // 16,
            "ref_audio_t": 0, "latent": None, "audio_latent": None,
        }
        out["image_jobs"].append(_job(
            frames, blk, "latent", video_frames,
            **_reflat_ref(video_frames, (width, height), "ref_video", frames=n),
        ))

        if soundtrack is not None:
            out["ref_items"].append({"type": "audio"})
            out["audio_jobs"].append(_job(
                soundtrack, blk, "audio_latent", soundtrack,
                **_reflat_ref(_audio_track(soundtrack), (0, 0), "ref_audio", kind="audio"),
            ))

        # Qwen sees the video at 2 fps with timestamps — at *native* resolution,
        # exactly like a reference image: ``process_video_block`` does its own
        # resize and patching, so fitting the clip to the canvas here would only
        # discard detail and tie the vision cache to the canvas.
        sample_idx = list(range(0, n, FPS // 2))
        out["ref_items"].append({
            "type": "video", "data": video_frames[sample_idx, ..., :3],
            "timestamps": [i / 2.0 for i in range(len(sample_idx))],
        })
        out["ref_blocks"].append(blk)

    for audio in ref_audios.values():
        if audio is None:
            continue
        blk = {"kind": "audio", "ref_audio_t": 0, "audio_latent": None}
        out["ref_items"].append({"type": "audio"})
        out["ref_blocks"].append(blk)
        out["audio_jobs"].append(_job(
            audio, blk, "audio_latent", audio,
            **_reflat_ref(_audio_track(audio), (0, 0), "ref_audio", kind="audio"),
        ))

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
        if item.get("cond") is not None:
            # Served from disk: the text half only, because its reference latents
            # were never encoded at this canvas. Nothing to tokenize or prefill.
            item["text_reused"] = True
            reused += 1
            continue
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

    ``cache`` spans one run only; :mod:`ref_latent_cache` is the layer beneath it
    and persists the same result across runs, addressed by the pixels. The
    canvas-fitted pixels are a pure function of (reference media, canvas), so
    switching the canvas back — or editing a prompt, which moves the
    conditioning cache onto another key and gets the old file pruned — reuses
    the encode instead of paying for the VAE again.
    """
    from . import ref_latent_cache

    cache = {} if cache is None else cache
    done = 0
    for item in prepared:
        for job in item["image_jobs"]:
            ck = job["cache_key"]
            if ck in cache:
                job["container"][job["key_name"]] = cache[ck]
                continue
            # ``job["address"]`` was derived from the source media + canvas, so it
            # is known before any tensor work — a hit costs no resize at all.
            addr = job.get("address")
            entry = ref_latent_cache.load_by_key(addr) if addr else None
            if entry is not None:
                z = entry["latent"]
            else:
                z = vae.encode(job["pixel"])
                if addr:
                    ref_latent_cache.store_by_key(addr, z)
                done += 1
            cache[ck] = z
            cache[(ck, "raw")] = job["raw"]   # pin: keep id() stable
            job["container"][job["key_name"]] = z
        # Downstream only needs the latent, so drop the pixel after encoding.
        for kf in item["keyframes"]:
            kf.pop("image", None)
        item["image_jobs"].clear()
    return done


def encode_audio_vae_batch(audio_vae, prepared: list[dict], cache: dict | None = None) -> int:
    """Step 3 — audio VAE: encode every prepared soundtrack.

    Shares results the same way as :func:`encode_video_vae_batch`, including the
    cross-run layer: a soundtrack is the single most expensive thing to encode
    here and it is entirely independent of prompt and canvas, so it is cached by
    its waveform under :mod:`ref_latent_cache`.

    ``ref_audio_t`` has to ride along in the payload — it is the latent length
    the audio VAE reports, and the DiT payload reads it back.
    """
    from . import ref_latent_cache

    cache = {} if cache is None else cache
    done = 0
    for item in prepared:
        for job in item["audio_jobs"]:
            ck = job["cache_key"]
            if ck in cache:
                z, t = cache[ck]
            else:
                # Audio arrives as a dict; the waveform under it is what the VAE
                # actually encodes, so it is what identifies the entry.
                media = job["pixel"]
                addr = job.get("address")
                entry = ref_latent_cache.load_by_key(addr) if addr else None
                if entry is not None and entry.get("ref_audio_t") is not None:
                    z = entry["latent"]
                    t = int(entry["ref_audio_t"])
                else:
                    z, t = _ref_audio_encode(audio_vae, media)
                    t = int(t)
                    if addr:
                        ref_latent_cache.store_by_key(addr, z, ref_audio_t=int(t))
                    done += 1
                cache[ck] = (z, t)
                cache[(ck, "raw")] = job["raw"]
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
