"""Reference materials: loading them, and turning them into node kwargs.

``_load_refs`` / ``_load_ref_audios`` / ``_load_ref_videos`` walk a timeline's
per-segment reference blocks (and the workflow's global refs) into ``SegmentRef*``
values; the ``*_to_dict`` and ``refs_to_kwargs*`` helpers then flatten them into
the kwargs the official MiniMax nodes accept.

Two rules are encoded here and are easy to break by accident:

* a reference block with no usable file is **dropped**, not passed through — the
  node would otherwise fail deep inside encoding with a path the user never sees;
* i2v / fl2v / t2v / v2v take keyframes or a source clip, so
  ``CONTEXT_REFERENCE_EXCLUDED_KEYS`` keeps reference images out of their context;
  feeding them a still made the model treat it as a moving shot.
"""

from __future__ import annotations

import base64
import io
import logging
import os

import numpy as np
import torch
from PIL import Image

import folder_paths

from ..lib.audio_io import load_reference_audio
from ..lib.ref_audios import MAX_REFERENCE_AUDIOS, ref_audios_dict
from ..lib.ref_images import MAX_REFERENCE_IMAGES, REF_IMAGE_KEY_PREFIX
from ..lib.ref_videos import MAX_REFERENCE_VIDEOS, ref_videos_dict
from ..lib.video_io import (
    load_reference_video_clip,
    load_timeline_segment,
    logical_frame_count,
)
from .plan_types import (
    DirectorPlan,
    SegmentPlan,
    SegmentRef,
    SegmentRefAudio,
    SegmentRefVideo,
)

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.plan.refs")


def _ref_video_has_file(ref_block: dict | None) -> bool:
    if not ref_block:
        return False
    return bool((ref_block.get("videoFile") or ref_block.get("fileName") or "").strip())


def _continuous_reference_enabled(timeline: dict, edit_mode: str, task_key: str) -> bool:
    """Global ads2v only: align reference video timeline offset with each segment start."""
    if edit_mode != "global" or task_key != "ads2v":
        return False
    global_block = timeline.get("global") or {}
    return bool(
        global_block.get("continuousReference")
        or global_block.get("continuous_reference")
        or timeline.get("continuousReference")
        or timeline.get("continuous_reference")
    )


def _resolve_global_reference_video(timeline: dict) -> dict:
    global_block = timeline.get("global") or {}
    ref = global_block.get("referenceVideo") or global_block.get("reference_video") or {}
    if _ref_video_has_file(ref):
        return dict(ref)
    legacy = timeline.get("referenceVideo") or timeline.get("reference_video") or {}
    return dict(legacy) if isinstance(legacy, dict) else {}


def _decode_image_b64(b64_str: str) -> torch.Tensor:
    if not b64_str:
        raise ValueError("Empty image data.")
    if b64_str.startswith("/view?"):
        raise ValueError("Remote view URLs are not supported; upload images in the Director node.")
    payload = b64_str.split(",", 1)[1] if "," in b64_str else b64_str
    img_bytes = base64.b64decode(payload)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def load_reference_tensor(ref: dict) -> torch.Tensor | None:
    if ref.get("imageFile"):
        rel = str(ref["imageFile"]).replace("\\", "/")
        file_path = os.path.join(folder_paths.get_input_directory(), rel.replace("/", os.sep))
        if os.path.exists(file_path):
            img = Image.open(file_path).convert("RGB")
            arr = np.array(img, dtype=np.float32) / 255.0
            return torch.from_numpy(arr).unsqueeze(0)

    b64_str = ref.get("imageB64", "")
    if not b64_str:
        return None
    try:
        return _decode_image_b64(b64_str)
    except Exception as exc:
        log.warning("Failed to decode reference image: %s", exc)
        return None


def load_source_video_from_timeline(timeline: dict) -> torch.Tensor:
    """Load all logical frames (legacy). Prefer load_timeline_segment for long videos."""
    total = logical_frame_count(timeline)
    if total <= 0:
        video = timeline.get("video") or {}
        if not (video.get("frames") or []):
            raise ValueError("No frames in MiniMax H3 Director Opt timeline.")
    return load_timeline_segment(timeline, 0, max(1, total))


def _load_refs(ref_list: list[dict]) -> list[SegmentRef]:
    refs: list[SegmentRef] = []
    for item in ref_list or []:
        index = int(item.get("index", item.get("slot", len(refs))))
        if index < 0 or index >= MAX_REFERENCE_IMAGES:
            continue
        tensor = load_reference_tensor(item)
        if tensor is not None:
            image_file = str(
                item.get("imageFile") or item.get("image_file") or item.get("fileName") or ""
            ).replace("\\", "/").strip()
            refs.append(SegmentRef(index=index, tensor=tensor, image_file=image_file))
    return sorted(refs, key=lambda r: r.index)


def load_reference_audio_item(item: dict) -> dict | None:
    """Load one timeline refAudios entry into a ComfyUI AUDIO dict."""
    rel = str(
        item.get("audioFile")
        or item.get("audio_file")
        or item.get("fileName")
        or item.get("file_name")
        or ""
    ).replace("\\", "/").strip()
    if not rel:
        return None
    sub = str(item.get("subfolder") or "").replace("\\", "/").strip().strip("/")
    if sub and not rel.startswith(sub + "/"):
        rel = f"{sub}/{rel}"
    file_path = os.path.join(folder_paths.get_input_directory(), rel.replace("/", os.sep))
    if not os.path.isfile(file_path):
        log.warning("Reference audio missing: %s", file_path)
        return None
    audio = load_reference_audio(file_path)
    if audio is None:
        log.warning("Failed to decode reference audio: %s", file_path)
    return audio


def _load_ref_audios(audio_list: list[dict]) -> list[SegmentRefAudio]:
    out: list[SegmentRefAudio] = []
    for item in audio_list or []:
        if not isinstance(item, dict):
            continue
        index = int(item.get("index", item.get("slot", len(out))))
        if index < 0 or index >= MAX_REFERENCE_AUDIOS:
            continue
        audio = load_reference_audio_item(item)
        if audio is None:
            continue
        rel = str(item.get("audioFile") or item.get("audio_file") or item.get("fileName") or "").strip()
        out.append(SegmentRefAudio(index=index, audio=audio, audio_file=rel))
    return sorted(out, key=lambda a: a.index)


def segment_ref_audios_for_context(task_key: str, audios: list[SegmentRefAudio]) -> list[SegmentRefAudio]:
    """Standalone ref audios apply to r2v / rv2v (official ReferenceToVideo)."""
    if task_key not in {"r2v", "rv2v"}:
        return []
    return audios


def ref_audios_to_dict(audios: list[SegmentRefAudio]) -> dict | None:
    return ref_audios_dict([(a.index, a.audio) for a in audios])


def ref_video_audios_to_dict(items) -> dict | None:
    """Index-paired soundtracks for reference videos.

    Keys MUST be ``ref_video_audio_<N>``: the official node pairs a soundtrack
    with ``ref_video_<N>`` via ``ref_video_audios.get("ref_video_audio_" + N)``,
    so any other naming silently drops the soundtrack and the reference video
    is treated as silent (``kind="video"`` instead of ``"video_audio"``).
    """
    out: dict = {}
    for item in items or []:
        idx = int(getattr(item, "index", -1))
        audio = getattr(item, "audio", None)
        if idx < 0 or not isinstance(audio, dict) or audio.get("waveform") is None:
            continue
        out[f"ref_video_audio_{idx}"] = audio
    return out or None


def _ref_video_entry_has_file(item: dict | None) -> bool:
    if not isinstance(item, dict):
        return False
    return bool((item.get("videoFile") or item.get("fileName") or "").strip())


def _load_ref_videos(
    video_list: list[dict],
    timeline: dict,
    num_frames: int,
) -> list[SegmentRefVideo]:
    """Load up to 3 standalone reference videos for r2v / ReferenceToVideo."""
    out: list[SegmentRefVideo] = []
    for item in video_list or []:
        if not isinstance(item, dict) or not _ref_video_entry_has_file(item):
            continue
        index = int(item.get("index", item.get("slot", len(out))))
        if index < 0 or index >= MAX_REFERENCE_VIDEOS:
            continue
        try:
            tensor = load_reference_video_clip(item, timeline, num_frames, start_frame=0)
        except Exception as exc:
            log.warning("Failed to load reference video slot %s: %s", index, exc)
            continue
        if tensor is None or tensor.numel() <= 0:
            continue
        rel = str(item.get("videoFile") or item.get("fileName") or "").strip()
        out.append(SegmentRefVideo(index=index, tensor=tensor, video_file=rel, meta=dict(item)))
    return sorted(out, key=lambda v: v.index)


def ref_videos_to_dict(videos: list[SegmentRefVideo]) -> dict | None:
    return ref_videos_dict([(v.index, v.tensor) for v in videos])


# i2v/fl2v use keyframes; v2v uses source clip as <Video 1>; r2v uses ref_images.
CONTEXT_REFERENCE_EXCLUDED_KEYS = frozenset({"i2v", "fl2v", "t2v", "v2v"})

def segment_refs_for_context(task_key: str, refs: list[SegmentRef]) -> list[SegmentRef]:
    if task_key in CONTEXT_REFERENCE_EXCLUDED_KEYS:
        return []
    return refs


def refs_to_kwargs(refs: list[SegmentRef]) -> dict[str, torch.Tensor]:
    return {f"{REF_IMAGE_KEY_PREFIX}{ref.index}": ref.tensor for ref in refs}


def reference_video_for_segment(plan: DirectorPlan, seg: SegmentPlan, num_frames: int) -> torch.Tensor | None:
    """Optional separate reference video for r2v (not used by v2v — source clip is the ref)."""
    if seg.task_key != "r2v":
        return None
    if not _ref_video_has_file(seg.reference_video_meta):
        return None
    return load_reference_video_clip(
        seg.reference_video_meta,
        plan.raw,
        num_frames,
        start_frame=seg.reference_video_start_frame,
    )


def refs_to_kwargs_for_context(task_key: str, refs: list[SegmentRef]) -> dict[str, torch.Tensor]:
    return refs_to_kwargs(segment_refs_for_context(task_key, refs))
