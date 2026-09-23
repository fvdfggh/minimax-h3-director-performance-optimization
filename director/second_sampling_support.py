"""Second-pass tuning constants and the small helpers the executor works in.

「二次采样」re-runs a segment at a higher resolution: the first pass's AV latent is
upscaled, re-seamed against the *upscaled* previous segment, sampled on a different
noise schedule, then decoded and cached under its own group. This module holds the
policy that is easier to reason about on its own than buried in the executor:

* the sigma schedule and sampler — taken from the removed upstream refine module
  (``ManualSigmas`` with 4 values = euler in 3 steps);
* the fixed second-pass seed, deliberately **not** user-exposed: the second pass
  must own a global noise field distinct from the first pass's UI seed (0 by
  default), or the two passes would sample overlapping noise;
* the system-RAM budget for holding upscaled latents before a batch is flushed —
  purely a RAM guard; VRAM is handled by parking the upscaler before sampling, not
  by shrinking this;
* the AV split / join and frame-budget helpers the executor is written in terms of.

Nothing here imports the executor: :mod:`second_sampling` depends on this module, not
the other way round.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from .cache_export import _expected_export_frames
from .frame_align import minimax_align_frame_count
from .h3_motion_context import snap_context_frames


#: (seg_index, done, total) progress callback.
ProgressCb = Callable[[int, int, int], None]

# 二采默认噪声调度：取自已删除的 MiniMaxH3DirectorOptRefine 模块
# （原 director/refine_pack.py: HAILUO_REFINE_SIGMAS + DEFAULT_REFINE_SIGMA_SAMPLER）。
# 海螺参考生视频二采：ManualSigmas 4 个数 = euler 3 步。
DEFAULT_SECOND_SIGMAS = (0.85, 0.7250, 0.4219, 0.0)
DEFAULT_SECOND_SIGMA_SAMPLER = "euler"
#: 二采专属固定种子：不暴露给用户调整，始终用同一个全局噪声场。
#: 取一个与一首采样种子（UI 的 seed 默认 0）不同的固定值，确保二采有自己独立的
#: 全局噪声场，不会与一首采样完全重合。
SECOND_SEED_FIXED = 20240

#: Batch budget for the upscaled latents held in system RAM before a batch is
#: flushed through sampling. Purely a RAM guard — VRAM is handled by parking the
#: upscaler before sampling, not by shrinking this.
DEFAULT_MEMORY_GUARD_BYTES = 2 * 1024**3


# ---------------------------------------------------------------------------
# AV latent split / join (mirrors the original refine helpers)
# ---------------------------------------------------------------------------

def _split_av(samples: dict):
    """Split an AV latent dict into ``(video_latent, audio_latent)``."""
    from comfy_extras.nodes_lt import LTXVSeparateAVLatent

    sep = LTXVSeparateAVLatent.execute(samples)
    if hasattr(sep, "args"):
        sep = sep.args
    return sep[0], sep[1]


def _join_av(video_latent, audio_latent, template: dict) -> dict:
    """Re-merge a video latent and an audio latent back into an AV latent dict."""
    out = dict(template)
    out.pop("noise_mask", None)
    try:
        from comfy_extras.nodes_lt import LTXVConcatAVLatent

        joined = LTXVConcatAVLatent.execute(video_latent, audio_latent)
        if hasattr(joined, "args"):
            joined = joined.args
        packed = joined[0]
        if isinstance(packed, dict) and "samples" in packed:
            return packed
        out["samples"] = packed
        return out
    except Exception:
        pass
    try:
        import comfy.nested_tensor

        v = video_latent.get("samples") if isinstance(video_latent, dict) else video_latent
        a = audio_latent.get("samples") if isinstance(audio_latent, dict) else audio_latent
        out["samples"] = comfy.nested_tensor.NestedTensor((v, a)) if a is not None else v
        return out
    except Exception:
        v = video_latent.get("samples") if isinstance(video_latent, dict) else video_latent
        a = audio_latent.get("samples") if isinstance(audio_latent, dict) else audio_latent
        out["samples"] = (v, a) if a is not None else v
        return out


def _accept_upscaled(up) -> dict | None:
    """Normalise whatever the upscale model returned into a latent dict."""
    if isinstance(up, dict) and "samples" in up:
        return up
    if isinstance(up, torch.Tensor):
        return {"samples": up}
    if isinstance(up, (list, tuple)) and up and isinstance(up[0], dict) and "samples" in up[0]:
        return up[0]
    if isinstance(up, dict):
        return up
    return None


def _tensor_bytes(value: Any) -> int:
    """Approximate resident bytes of an AV latent (video + audio)."""
    total = 0
    samples = value.get("samples") if isinstance(value, dict) else value
    if isinstance(samples, torch.Tensor):
        total += int(samples.numel()) * int(samples.element_size())
    elif isinstance(samples, (tuple, list)):
        for item in samples:
            if isinstance(item, torch.Tensor):
                total += int(item.numel()) * int(item.element_size())
    return total


def _wanted_context_frames(params: dict, plan) -> int:
    """How many context frames the *cached* conditioning was actually built for.

    ``context_n`` is what the FIRST pass really pinned — 0 when it pinned
    nothing, e.g. because the previous segment had not been sampled in that run.
    The cached text encoding only carries room for that many head frames, so the
    second pass has to reuse the exact same number: re-deriving it from the
    widget would pin a prefix the conditioning has no capacity for (either a
    refused seam or a clip whose head is unpinned noise).
    """
    raw = params.get("context_n")
    if raw is None:
        # Map written before ``context_n`` was recorded: fall back to the plan.
        return int(snap_context_frames(plan.continuity_overlap_frames))
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def _export_frame_budget(plan, seg, decoded_n: int, context_n: int) -> int:
    """Export length of a second-pass segment, on the FIRST pass's terms.

    ``seg.frame_count`` alone is not the export length: the first pass exports
    the segment's own aligned length (see :func:`_expected_export_frames`), and
    the replayed head is trimmed separately rather than snapping the export
    short. Reusing that helper is what keeps a second-pass clip the same length
    as its first-pass counterpart instead of a few frames longer/shorter.
    """
    _planned_trim, planned = _expected_export_frames(plan, seg, fallback_n=decoded_n)
    if int(context_n or 0) <= 0:
        # No replayed head was applied, so the body is the plain aligned length.
        frame_count = int(getattr(seg, "frame_count", 0) or 0)
        if frame_count > 0:
            try:
                return int(minimax_align_frame_count(max(5, frame_count)))
            except Exception:  # pragma: no cover - defensive
                return int(planned or decoded_n)
    return int(planned or 0) or int(decoded_n)
