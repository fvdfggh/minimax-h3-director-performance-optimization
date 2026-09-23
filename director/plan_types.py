"""Pure data model + wire-format normalisers for the Director plan.

``DirectorPlan`` / ``SegmentPlan`` are what every stage of the pipeline hands
around, so they live in their own **leaf** module: ``plan`` *builds* them, while
``segment_continuity``, ``segment_cache``, ``segment_runtime`` and
``segment_mp4_export`` only need the shape. Splitting the model out is what
removes the old ``plan`` ↔ ``segment_continuity`` import cycle.

The constants and normalisers below describe the same wire format (the timeline
JSON's ``segmentExport`` / ``secondSample`` / ``refImageSize`` fields), so they
travel with the model. Nothing here imports ComfyUI or torch at runtime — the
tensor fields are annotations only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - annotations only, keeps this module dep-free
    import torch

MIN_SEGMENT_FRAMES = 4

DEFAULT_SEGMENT_EXPORT_MODE = "piecewise"
SEGMENT_EXPORT_MODE_PIECEWISE = "piecewise"
SEGMENT_EXPORT_MODE_CONTINUOUS = "continuous"

_CONTINUOUS_MODE_ALIASES = frozenset({"continuous", "concat", "concatenate", "merged"})

#: Which pass's cache the「分段导出」picker reads: first-pass ``seg_*`` or
#: second-pass ``seg2_*``.
SEGMENT_EXPORT_SOURCE_FIRST = "1st"
SEGMENT_EXPORT_SOURCE_SECOND = "2nd"
DEFAULT_SEGMENT_EXPORT_SOURCE = SEGMENT_EXPORT_SOURCE_FIRST

_SECOND_SOURCE_ALIASES = frozenset({"2nd", "second", "2", "seg2", "second_pass"})


def normalize_segment_export_source(source) -> str:
    """Map a UI/payload cache-source string onto ``1st`` | ``2nd``."""
    text = str(source or "").strip().lower()
    if text in _SECOND_SOURCE_ALIASES:
        return SEGMENT_EXPORT_SOURCE_SECOND
    return SEGMENT_EXPORT_SOURCE_FIRST


def normalize_segment_export_mode(mode) -> str:
    """Map a UI/payload mode string onto ``piecewise`` | ``continuous``."""
    text = str(mode or "").strip().lower()
    if text in _CONTINUOUS_MODE_ALIASES:
        return SEGMENT_EXPORT_MODE_CONTINUOUS
    return SEGMENT_EXPORT_MODE_PIECEWISE


def resolve_export_mode(output_block: dict) -> str:
    """Read ``output.exportMode`` onto ``segments`` | ``all``."""
    mode = str(output_block.get("exportMode") or output_block.get("export_mode") or "all").lower()
    if mode in ("segments", "segment", "per_segment", "by_segment"):
        return "segments"
    return "all"


def parse_run_selection(timeline: dict, segment_count: int) -> frozenset[int] | None:
    """Return selected segment indices, or None when all segments should run."""
    enabled = bool(timeline.get("runSelectEnabled") or timeline.get("run_select_enabled"))
    if not enabled:
        return None
    raw = timeline.get("runSelection")
    if raw is None:
        raw = timeline.get("run_selection")
    if raw is None:
        return None
    if not isinstance(raw, list):
        return None
    indices = {int(i) for i in raw if 0 <= int(i) < segment_count}
    if not indices:
        raise ValueError(
            "MiniMax H3 Director Opt: 「选择运行」已开启但未勾选任何片段/提示词组。请至少勾选一组再执行。"
        )
    if len(indices) >= segment_count:
        return None
    return frozenset(indices)


@dataclass(frozen=True)
class SegmentExportRequest:
    """A「分段导出」request carried by the timeline JSON.

    ``indices`` holds the user-checked segment indices (already normalised and
    sorted).

    * ``piecewise`` — every checked segment is written out as its own mp4.
    * ``continuous`` — checked segments that are adjacent on the timeline are
      stitched into one mp4 with the streaming merge (``concat_chunks_lazy``); a
      checked segment with no neighbour stays a standalone mp4.

    ``source`` picks which pass's cache group is read (``"1st"`` → ``seg_*``,
    ``"2nd"`` → ``seg2_*``). The picker shows only the selected source's state,
    so there is deliberately no cross-source fallback — a segment with no
    second-pass cache is simply not exportable while「二采」is selected.
    """

    enabled: bool
    mode: str = DEFAULT_SEGMENT_EXPORT_MODE
    indices: tuple[int, ...] = ()
    source: str = DEFAULT_SEGMENT_EXPORT_SOURCE

    def normalized_mode(self) -> str:
        return normalize_segment_export_mode(self.mode)

    def normalized_source(self) -> str:
        return normalize_segment_export_source(self.source)


@dataclass(frozen=True)
class SegmentSecondSampleRequest:
    """A「二次采样」(second-pass) request carried by the timeline JSON.

    ``indices`` are the user-checked segment indices to re-sample (the segments
    whose first-pass AV latent + text encoding are on disk). The picker writes
    ``enabled=True`` as a one-shot trigger (mirroring :class:`SegmentExportRequest`);
    ``indices`` is persistent so the picker reopens with the last selection.

    The second pass always stitches adjacent selected segments into one clip
    (continuous-merge), so no ``mode`` field is needed here.
    """

    enabled: bool
    indices: tuple[int, ...] = ()


MIN_CONTINUITY_OVERLAP = 5
MAX_CONTINUITY_OVERLAP = 56
REF_IMAGE_SIZE_MATCH = "match"
REF_IMAGE_SIZE_MAX = "max"


def normalize_ref_image_size(value) -> str:
    raw = str(value or "").strip().lower()
    return REF_IMAGE_SIZE_MAX if raw == REF_IMAGE_SIZE_MAX else REF_IMAGE_SIZE_MATCH


def _timeline_dict(plan_or_timeline) -> dict:
    if isinstance(plan_or_timeline, dict):
        return plan_or_timeline
    raw = getattr(plan_or_timeline, "raw", None)
    return raw if isinstance(raw, dict) else {}


def _legacy_output_ref_image_size(timeline: dict | None) -> str | None:
    out = (timeline or {}).get("output") or {}
    raw = out.get("refImageSize")
    if raw is None:
        raw = out.get("ref_image_size")
    if raw is None or str(raw).strip() == "":
        return None
    return normalize_ref_image_size(raw)


def resolve_ref_image_size(seg_or_data=None, plan_or_timeline=None) -> str:
    """Per-segment MiniMax ``ref_image_size``; legacy ``output.refImageSize`` as fallback."""
    raw = None
    if isinstance(seg_or_data, dict):
        if "refImageSize" in seg_or_data or "ref_image_size" in seg_or_data:
            raw = seg_or_data.get("refImageSize")
            if raw is None:
                raw = seg_or_data.get("ref_image_size")
    elif seg_or_data is not None:
        raw = getattr(seg_or_data, "ref_image_size", None)
    if raw is not None and str(raw).strip() != "":
        return normalize_ref_image_size(raw)
    legacy = _legacy_output_ref_image_size(_timeline_dict(plan_or_timeline))
    return legacy if legacy is not None else REF_IMAGE_SIZE_MATCH


@dataclass
class SegmentRef:
    index: int
    tensor: "torch.Tensor"
    image_file: str = ""


@dataclass
class SegmentRefAudio:
    """Standalone reference audio for MiniMax ``<Audio N>`` (index 0-based)."""

    index: int
    audio: dict  # ComfyUI AUDIO: {waveform, sample_rate}
    audio_file: str = ""


@dataclass
class SegmentRefVideo:
    """Standalone reference video for MiniMax ``<Video N>`` (index 0-based)."""

    index: int
    tensor: "torch.Tensor"
    video_file: str = ""
    meta: dict = field(default_factory=dict)


def concat_common_segment_prompt(common: str | None, segment: str | None) -> str:
    """Join shared (common) prompt with per-group prompt.

    Both non-empty → ``common + blank line + segment``.
    Only one side → that side alone (replaces legacy ``segment or common`` fallback).
    """
    common_s = (common or "").strip()
    segment_s = (segment or "").strip()
    if common_s and segment_s:
        return f"{common_s}\n\n{segment_s}"
    return common_s or segment_s


def merge_indexed_refs(common: list, segment: list) -> list:
    """Merge common + per-group refs by slot index; segment wins on conflict."""
    by_idx: dict[int, object] = {}
    for item in common or []:
        by_idx[int(getattr(item, "index", 0))] = item
    for item in segment or []:
        by_idx[int(getattr(item, "index", 0))] = item
    return sorted(by_idx.values(), key=lambda r: int(getattr(r, "index", 0)))


@dataclass
class SegmentPlan:
    index: int
    start_frame: int
    end_frame: int
    prompt: str
    task_type: str
    task_key: str
    use_global: bool
    refs: list[SegmentRef] = field(default_factory=list)
    ref_audios: list[SegmentRefAudio] = field(default_factory=list)
    ref_videos: list[SegmentRefVideo] = field(default_factory=list)
    ref_video_audios: list[SegmentRefAudio] = field(default_factory=list)
    reference_video_meta: dict = field(default_factory=dict)
    reference_video_start_frame: int = 0
    negative_prompt: str = ""
    source_clip: "torch.Tensor | None" = None
    # When external groups filter by「选择运行」, plan.index is the compact run
    # order (0..N-1) while ui_index keeps the Director timeline card index.
    ui_index: int | None = None
    # Per-segment「引用上段」; master「段间引导」must also be on. Default True.
    continuity_from_prev: bool = True
    # Per-segment「对齐下段」: pin the *next* segment's opening into this tail so
    # the join is forged from both sides. Cache-driven middle-out mode, so it is
    # opt-in (default False) and only runs when that neighbour has a cached AV
    # latent. Master「段间引导」must also be on.
    continuity_to_next: bool = False
    # Official MiniMaxH3ReferenceToVideo combo: match | max. Per r2v/rv2v group.
    ref_image_size: str = "match"

    @property
    def frame_count(self) -> int:
        return max(0, self.end_frame - self.start_frame)

    @property
    def timeline_index(self) -> int:
        """Index used for UI preview / highlight (timeline card)."""
        return int(self.index if self.ui_index is None else self.ui_index)


@dataclass
class DirectorPlan:
    frame_rate: float
    total_frames: int
    width: int
    height: int
    ref_max_size: int
    output_mode: str
    source_width: int
    source_height: int
    global_task_type: str
    global_task_key: str
    global_prompt: str
    global_refs: list[SegmentRef]
    segments: list[SegmentPlan]
    source_video: "torch.Tensor"
    edit_mode: str
    raw: dict
    source_total_frames: int = 0
    export_max_frames: int = 0
    export_mode: str = "all"  # "all" | "segments"
    run_indices: frozenset[int] | None = None  # None = run all segments
    segment_export: SegmentExportRequest | None = None
    second_sample: SegmentSecondSampleRequest | None = None
    continuity_enabled: bool = False
    continuity_overlap_frames: int = 0
    continuity_redraw: float = 0.10  # 段间锥形重绘幅度（seam_min_mask），0..0.95
    global_ref_audios: list[SegmentRefAudio] = field(default_factory=list)
    # Sampling knobs stamped at execute time (cache fingerprint).
    sample_seed: int = 0
    sample_cfg: float = 1.0
    sample_steps: int = 25
    sample_sampler: str = ""
    sample_scheduler: str = ""
    sample_shift_video: float = 12.0
    sample_shift_audio: float = 3.0

    @property
    def segment_count(self) -> int:
        return len(self.segments)
