"""Continuity tuning knobs for the active MiniMax H3 path.

Single source for every ``CONTINUITY_*`` value: :mod:`continuity_settings` reads
them while parsing a timeline, :mod:`continuity_seam` applies them while grading a
seam / repairing a hold-pop, and :mod:`segment_continuity` uses them while
concatenating chunks.

Most of the upstream Wan/SCAIL-era tuning set (seam echo / join limits, lock &
unlock feather masks, free-latent warm-start ramps, additive-luma and
opening-Y-map blends) was never read by this pipeline and has been deleted. What
is left is deliberately conservative: the RGB seam blends are pinned to 0 because
they caused 拖影 / 一顿一顿 / 重影 (see 00035), so the active path only nudges the
low-frequency luma+chroma field.
"""


# --- Continuity knobs still read by the active MiniMax H3 path ----------------
# The upstream Wan/SCAIL-era tuning set (seam echo / join limits, lock & unlock
# feather masks, free-latent warm-start ramps, additive-luma and opening-Y-map
# blends) was never read by this pipeline and has been removed.

# No multiplicative gain / long RGB blend (画面花 / 幻影).
CONTINUITY_SEAM_SOFTEN_FRAMES = 0

CONTINUITY_TAIL_LUMA_BLEND = 0
CONTINUITY_OPENING_EXPOSURE_SMOOTH = 0
CONTINUITY_OPENING_LUMA_BLEND = 0

# Concat: additive luma ONLY — body0/hold→pop RGB caused 拖影+一顿一顿 (00035).
CONTINUITY_SEAM_ADD_LUMA_FRAMES = 12

# Concat opening grade: low-freq appearance pull replaces the additive mean-luma
# nudge in the seam pipeline (no 重影; pulls luma+chroma low-freq field instead).
CONTINUITY_EXPORT_GRADE_FRAMES = 12
CONTINUITY_EXPORT_GRADE_WEIGHT = 0.70
CONTINUITY_EXPORT_GRADE_BLUR = 64
CONTINUITY_BODY0_SEAM_WEIGHT = 0.0
CONTINUITY_BODY1_SEAM_WEIGHT = 0.0
CONTINUITY_MICRO_SEAM_MAD = 99.0
CONTINUITY_MICRO_SEAM_WEIGHT = 0.0
CONTINUITY_MICRO_SEAM_WEIGHT_MAX = 0.0
CONTINUITY_MICRO_SEAM_SECOND_WEIGHT = 0.0
CONTINUITY_MICRO_SEAM_THIRD_WEIGHT = 0.0
CONTINUITY_MICRO_SEAM_MAD_SPAN = 8.0

CONTINUITY_TAIL_SOFTEN_FRAMES = 0
CONTINUITY_HOLD_MAX_FRAMES = 0
CONTINUITY_HOLD_MAD = 4.5
CONTINUITY_HOLD_POP_JUMP_MAD = 10.0
CONTINUITY_HOLD_POP_BRIDGE_WEIGHT = 0.0
CONTINUITY_HOLD_POP_BRIDGE_WEIGHT_MAX = 0.0
CONTINUITY_HOLD_POP_LAND_WEIGHT = 0.0
CONTINUITY_HOLD_POP_LOOKAHEAD = 2
CONTINUITY_HOLD_POP_SCAN = 6
CONTINUITY_HOLD_POP_LOWFREQ = False
CONTINUITY_HOLD_POP_BLUR = 32
CONTINUITY_SPIKE_PREV_MAD = 6.5
CONTINUITY_SPIKE_JUMP_MAD = 12.0
CONTINUITY_SPIKE_WEIGHT = 0.0
CONTINUITY_SPIKE_LAND_WEIGHT = 0.0
CONTINUITY_SPIKE_SCAN = 5
CONTINUITY_HOLD_POP_ON_TAIL = False


# 段间「锥形重绘」幅度（=上游 seam_min_mask）。0 = 接缝硬锁（前缀几乎全重绘、仅
# 接缝保留旧尾）；0.95 = 几乎不重绘。沿用上游成熟默认 0.10。
DEFAULT_CONTINUITY_REDRAW = 0.10
CONTINUITY_REDRAW_MIN = 0.0
CONTINUITY_REDRAW_MAX = 0.95
