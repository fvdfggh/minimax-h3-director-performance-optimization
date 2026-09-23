/** Layout and design values shared by the editor class and its mount helpers.
 *
 * The timeline is drawn in canvas pixels, so these numbers decide how tall the
 * track is, how big a drag handle is, how close two clip joints may sit, how many
 * thumbnails a segment may cache, and how far the DOM widget is allowed to grow
 * before a resize is treated as runaway. Both the drawing code (inside the editor
 * class) and the mount / resize helpers read them, which is why they live in their
 * own module instead of inside either.
 *
 * Values are unchanged from when they were declared in minimax_timeline.js.
 */


export const RULER_H = 24;
export const SEG_LABEL_H = 20;
export const TRACK_H = 160;
export const TRACK_Y = RULER_H + SEG_LABEL_H;
export const STAGE_PREVIEW_H = 220;
export const LIVE_SAMPLE_PREVIEW_H = 320;
export const MIN_SEG = 4;
export const HANDLE_PX = 14;
/** Canvas-drawn run-select checkbox (not a DOM control). */
export const RUN_CHECK_SIZE = 14;
export const RUN_CHECK_HIT_PAD_X = 8;
export const RUN_CHECK_HIT_PAD_Y = 4;
/** Canvas-drawn 段间引导 marker at a clip joint (only when master switch is on). */
export const CONT_JOINT_W = 22;
export const CONT_JOINT_H = 16;
export const CONT_JOINT_Y = TRACK_Y + 4;
export const CONT_JOINT_HIT_PAD = 5;
export const THUMB_MAX_W = 168;
export const THUMB_JPEG_Q = 0.55;
export const TIMELINE_SYNC_DEBOUNCE_MS = 500;
export const MAX_THUMBS_PER_SEGMENT = 20;
export const THUMB_PREFETCH_BATCH = 6;
export const DIRECTOR_MIN_WIDTH = 900;


/** Segment continuity is opt-in; default off unless explicitly enabled in output. */

// 仅隐藏内部序列化字段 timeline_data，以及和时间轴面板重复的只读/派生参数。
// cfg 与“声音”控制按用户要求可见（不在此列表）。bd_grp_* 分组头本就不在此列表，永远可见。
export const HIDDEN_WIDGETS = [
    "timeline_data", "total_frames", "width", "height", "ref_max_size",
    "task_type", "global_prompt", "frame_rate",
];

export const DIRECTOR_UI_RUNAWAY_ABS_H = 12000;
export const DIRECTOR_UI_RUNAWAY_EXTRA_H = 8000;


export const DIRECTOR_DOM_WIDGET_NAME = "minimax_director_ui";
