# ComfyUI MiniMax H3 Director Opt

A multi-segment **timeline Director** node pack built on ComfyUI's **official MiniMax H3 pipeline**
(`MiniMaxH3ImageToVideo` / `MiniMaxH3ReferenceToVideo`, ComfyUI PR #15224 / #15228). It adds an
in-node timeline editor, per-segment caching, segment continuity, a second sampling pass and
audio/video export on top of that pipeline.

- Upstream project: `AIMixer/ComfyUI_MiniMaxH3_Director` (author: AI搅拌手)
- **Opt fork author: fvdfggh**
- Repository: <https://github.com/fvdfggh/minimax-h3-director-performance-optimization>
- License: Apache-2.0 (see `LICENSE`)

> Every node type id, display name, HTTP route, frontend event and cache root in this fork carries
> the `Opt` marker, so it can be installed **side by side** with the original pack without collisions.

---

## Features

- **In-node timeline**: upload a source clip → split / equal-split / smart-split → write per-segment
  prompts and attach reference images / audio / videos → queue the run.
- **Official H3 pipeline**: conditioning goes through `MiniMaxH3ImageToVideo` /
  `MiniMaxH3ReferenceToVideo`; sampling is a single KSampler with `MiniMaxH3SigmaShift`; decoding
  splits the AV latent via `LTXVSeparateAVLatent`, so picture and sound come out together.
- **Six tasks**: `t2v`, `i2v`, `fl2v`, `r2v`, `v2v`, `rv2v`.
- **Per-segment cache** keyed by content fingerprint — editing one segment re-renders only that
  segment. Cache lives in `output/minimax_director_opt_cache/<workflow slug>/node_<id>/`.
- **Segment continuity (段间引导)**: the previous segment's tail is written into this segment's body
  prefix and re-drawn (continue mode), with a 5 / 22 / 39 / 56-frame context window.
- **Second sampling (二采)**: upscale + re-sample already cached segments into their own `seg2_*`
  cache — the first pass is never overwritten.
- **Segment export**: check segments and get the clips straight on the node's `images` / `audio`
  outputs (piecewise or continuous, first- or second-pass source).
- **Audio modes**: `generate` (model audio) / `source` (original soundtrack) / `mute`, plus
  extract-audio and retain-audio.
- **Audio validity check (ASR)**: with a MOSS transcribe model wired, the generated soundtrack is
  compared against the prompt's speaking lines, speaker by speaker.
- **Director pack (`.mmxpack.zip`)**: timeline + assets export/import, layout-compatible with the
  upstream pack.
- **External Group nodes**: wire `MMX_DIR_GROUP` from the graph; external groups take priority over
  the in-node UI cards.

---

## Install

```powershell
cd ComfyUI/custom_nodes
git clone https://github.com/fvdfggh/minimax-h3-director-performance-optimization ComfyUI_MiniMaxH3_Director_Opt
pip install -r ComfyUI_MiniMaxH3_Director_Opt/requirements.txt
```

Dependencies (`requirements.txt`):

| Package | Used for |
|---|---|
| `av>=13.0` | source-video decode for v2v / timelines (ships with the ComfyUI portable build; OpenCV is deliberately NOT used) |
| `imageio-ffmpeg>=0.4` | source-audio extraction, incremental segment mp4 encoding |
| `scenedetect>=0.6.4,<0.8` | smart split (shot detection) |

Restart ComfyUI; the frontend (`web/js`, extension `ComfyUI.MiniMaxH3DirectorOptPlugin`) loads
automatically.

---

## Required models

| Socket | Type | Notes |
|---|---|---|
| `model` (`model_b`, `model_c`) | MODEL | MiniMax H3 UNET (`UNETLoader`). `model` is required and is the fallback for an unwired backup slot |
| `video_vae` | VAE | MiniMax H3 video VAE (`minimax_h3_video_vae`) |
| `audio_vae` | VAE | MiniMax H3 audio VAE (`minimax_h3_audio_vae`); required for `r2v` / `v2v` / `rv2v` |
| `clip` | CLIP | `CLIPLoader` type=minimax (qwen3vl) |
| `upscale_model` | LATENT_UPSCALE_MODEL | second sampling only — `Minimax H3 Latent Upscaler Opt (3D) [Model]` |
| `asr_model` | T8_MOSS_TRANSCRIBE_MODEL | optional, from `Comfyui-MOSS-Transcribe-Diarize-T8` |

ComfyUI must also ship `comfy_extras.nodes_minimax_h3` (official MiniMax H3 nodes); otherwise the
node fails with “Upgrade to ComfyUI with PR #15224 merged”.

Second sampling and the two Autogrow Group nodes need `comfy_api.latest` (V3 node API); without it
the latent-upscale model node is not registered and the R2V group falls back to static slots.

---

## Quick start

1. Add **MiniMaxH3Director Opt** (category `MiniMaxH3 Opt`).
2. Wire `model`, `video_vae`, `audio_vae`, `clip`.
3. In the node, pick a task type, write the global (or per-segment) prompt and upload assets.
4. Connect `images` / `audio` to `VHS_VideoCombine` (or `SaveImage` / `SaveAudio`).
5. Queue with ComfyUI's **Run**; progress and live sampling previews show on the node.

Defaults: 864×480 (0.4 MP 16:9), 124 frames @ 24 fps, cfg 1.0, 25 steps, `res_multistep` + `simple`,
`shift_video=12` / `shift_audio=3` (matching the official template).

---

## Node list

| Type id | Display name | Category | Outputs |
|---|---|---|---|
| `MiniMaxH3DirectorOpt` (legacy id `ComfyMiniMaxH3DirectorOpt` still loads) | MiniMaxH3Director Opt | `MiniMaxH3 Opt` | `images`, `audio`, `fps`, `frame_count`, `source_images`, `report` |
| `MiniMaxH3DirectorOptConditioning` | MiniMax H3 Director Opt Conditioning | `MiniMaxH3 Opt` | `positive`, `latent` |
| `MiniMaxH3DirectorOptPlannerConditioning` | MiniMax H3 Director Opt Planner Conditioning | `MiniMaxH3 Opt` | `positive`, `latent`, `task_mode` |
| `MiniMaxH3DirectorOptGroupImageToVideo` | MiniMax H3 Director Opt Group (Image to Video) | `MiniMaxH3 Opt/Director Groups` | `group` |
| `MiniMaxH3DirectorOptGroupReferenceToVideo` | MiniMax H3 Director Opt Group (Reference to Video) | `MiniMaxH3 Opt/Director Groups` | `group` |
| `MiniMaxH3DirectorOptGroupsCombine` | MiniMax H3 Director Opt Groups Combine | `MiniMaxH3 Opt/Director Groups` | `groups` |
| `MiniMaxH3FastVideoVAEOpt` | MiniMax H3 Fast Video VAE Opt | `MiniMaxH3 Opt` | `vae` |
| `MiniMaxH3LatentUpscaleModelOpt` | Minimax H3 Latent Upscaler Opt (3D) [Model] | `MiniMaxH3 Opt` | `upscale_model` |

`images`, `audio` and `source_images` are **lists** (`OUTPUT_IS_LIST`): in segments export mode each
clip is one entry.

### Group nodes (external wiring)

- **Group (Image to Video)** — `prompt` + `duration_sec` + optional `first_frame` / `last_frame`.
  No frames = t2v; first only = i2v; any last frame = fl2v.
- **Group (Reference to Video)** — `prompt` + `duration_sec` + Autogrow slots: reference images ≤ 9,
  reference videos ≤ 3, reference video audios ≤ 3, standalone reference audios ≤ 3 (same UX as the
  official R2V node).
- **Groups Combine** — fan several groups into one list; **do not mix** i2v and r2v groups in one list.

Wiring `i2v_groups` / `r2v_groups` overrides the in-node UI cards. Output size is always controlled
by the Director node, never by the group packers.

---

## Director node parameters

### Required

| Widget | Default | Notes |
|---|---|---|
| `model` / `video_vae` / `audio_vae` / `clip` | — | see “Required models” |
| `task_type` | `t2v — 文生视频(Text to Video)` | one of six |
| `global_prompt` | `A cinematic scene with natural motion and synchronized ambience` | sent straight to the H3 nodes |
| `cfg` | 1.0 | KSampler cfg |
| `seed` | 0 | supports control_after_generate |
| `frame_rate` | 24.0 | timeline / output fps |
| `width` / `height` | 864 / 480 | step 32 |
| `ref_max_size` | 864 | long edge for the `long_edge` scaling mode |
| `total_frames` | 124 | timeline length (fl2v = sum of shots) |
| `timeline_data` | — | internal; written by the frontend timeline (hidden in the UI) |

### Optional

| Widget | Default | Notes |
|---|---|---|
| `run_model` | main model | which MODEL socket samples: `model` / `model_b` / `model_c`; an unwired pick falls back to `model` |
| `model_b` / `model_c` | — | backup UNET 1 / 2 |
| `i2v_groups` / `r2v_groups` | — | external group packs (priority over UI cards) |
| `steps` | 25 | sampling steps |
| `sampler` | `res_multistep` | KSampler sampler |
| `scheduler` | `simple` | scheduler |
| `shift_video` / `shift_audio` | 12.0 / 3.0 | `MiniMaxH3SigmaShift` |
| `sigmas` + `use_sigmas` | off | custom noise schedule (`BasicScheduler` / `ManualSigmas`). When on: steps = `len(sigmas)-1`, denoise is fixed at 1.0 and `steps` / `scheduler` are ignored; an unwired or unparsable input falls back to defaults |
| ~~`conn_noise`~~ | — | **removed, always on**: taper-redraw is unconditional now (see Continuity). An old workflow's slot for it is dropped on load |
| `upscale_model` | — | **mandatory** for second sampling; the run errors out without it |
| `second_sigmas` | unwired | second-pass schedule; defaults to the Hailuo second-pass schedule `(0.85, 0.7250, 0.4219, 0.0)` (euler, 3 steps) |
| `second_run_model` | main model | which MODEL socket the second pass uses |
| `second_seed` | 20240 | fixed second-pass seed, independent of the first-pass seed |
| `asr_model` | optional | audio validity check — wire it and run once; the r2v toolbar then offers an「Audio validity check」button |
| `workflow_name` | hidden | filled by the frontend; namespaces the on-disk cache |

The group headers (`采样设置` / `高级采样` / `二级采样`) are a custom frontend `BDGROUP` widget —
cosmetic folding only.

### Fixed behaviour (no longer widgets)

Constants at the top of `nodes/director_common.py` (restart ComfyUI after editing):

| Constant | Value | Meaning |
|---|---|---|
| `USE_CONDITIONING_CACHE` | `True` | reuse on-disk CLIP/VAE conditioning across runs (key = prompt + canvas + model fingerprint) |
| `CLEAR_VRAM_BETWEEN_SEGMENTS` | `True` | unload models and empty the CUDA cache after each segment |
| `EXPORT_SOURCE_IMAGES` | `False` | decode the source timeline onto the `source_images` output |

---

## Task types

| Key | Meaning |
|---|---|
| `t2v` | text to AV — no keyframes, no references |
| `i2v` | image to AV — first keyframe conditioning |
| `fl2v` | first + last keyframe AV |
| `r2v` | reference to AV — subject images / videos / audios referenced via `<Picture N>` / `<Video K>` / `<Audio J>` |
| `v2v` | video edit — each source timeline slice is fed as `<Video 1>` to `ReferenceToVideo` |
| `rv2v` | video edit with references — source `<Video 1>` + `<Picture N>` + `<Audio J>`; without references it behaves like `v2v` |

Frontend modes: `fl2v` → shot (first/last frame) panel; `t2v` / `i2v` / `r2v` → prompt-group batch
panel (no source-video upload); `v2v` / `rv2v` → source-video timeline panel.

---

## Timeline editor (frontend)

Entry point `web/js/minimax_timeline.js`; DOM widget `minimax_director_ui`; editor class
`MiniMaxH3DirectorOptEditor` (assembled from 25 mixins). Default node size `[1000, 680]`, minimum
width 900.

### Toolbar

| Button | Action |
|---|---|
| Upload video / Pick existing / Append video | load (or append) the source clip |
| + Split | cut at the playhead |
| Equal split (2–64 segments) | split evenly, keeping clip boundaries as forced cuts |
| Smart split | backend `detect_shots` (scenedetect); needs ≥ 8 total frames |
| Run selection / Select all | re-run only the checked segments |
| Segment export | checked segments → node outputs (piecewise / continuous, 1st / 2nd pass cache) |
| Second sample | checked segments → upscale + re-sample |
| Extract audio | pull a clip out of the source video as a reusable entry |
| Delete segment | really removes the source frames and re-packs later segments |
| Global / Segment mode | switch prompt editing scope |
| Import / Export director pack | `.mmxpack.zip` |
| Clear cache / Clear all node cache | see Cache |
| EN | Chinese ⇄ English UI (stored in `mmx_director_ui_locale`) |

### Output bar

Resolution presets (default 16:9 widescreen), megapixels (0.1–16), long edge, width / height
(step 32), scaling mode (`long_edge` / `fixed`), fps (1–240), sound (`generate` / `source` / `mute`),
export mode (`all` / `segments`), segment continuity + context frames (5 / 22 / 39 / 56, default 22),
live preview toggle.

### Playback

`▶` play, `⟳` loop, `‹` / `›` step one frame, frame input (1-based), timecode, seek bar, plus a live
sampling preview fed by the `minimax_director_opt_preview` event.

### Keyboard shortcuts (active while the pointer is over the editor)

| Key | Action |
|---|---|
| `Delete` / `Backspace` | delete the selected segment |
| `Space` | play / pause |
| `←` / `→` | ±1 frame; `Shift` + arrow = ±10 frames |
| `Esc` | close a modal / revert the frame input |
| Right click on canvas | add a split point (disabled in fl2v mode) |

Shortcuts are suppressed while an `INPUT` / `TEXTAREA` / `SELECT` / editable region has focus.
**Split points cannot be deleted with Delete** — use the “delete split point” button.

### Drag & drop

Drag a segment body to reorder; drag its edges to retime (video mode keeps neighbours aligned, fl2v
ripples later shots); drop files directly (video → source clip, image → reference slot).

---

## Segment continuity

Continuity is **opt-in per timeline** (`timeline.output.continuityEnabled`, off by default):

- needs ≥ 2 segments; supported for `t2v` / `i2v` / `fl2v` / `r2v` / `v2v` / `rv2v`; segment 1 never
  references a predecessor;
- context frames can only be **5 / 22 / 39 / 56** (default 22 — aligned to the VAE's 17-frame cycle);
- per segment you can switch off “reference previous” (default on); “align to next” is off by default
  and only available when that neighbour already holds an AV latent;
- taper-redraw is **always on** (the old `conn_noise` switch is gone): the previous tail is written
  into this segment's body prefix and re-drawn (continue mode); the redraw amount comes from the
  timeline's 重绘幅度 (default 0.10 — 0 = hard seam lock, 0.95 = almost no redraw);
- **do not enable it together with a standalone H3 Motion Context node.**

After swapping the source video the old cache no longer matches; if a segment needs its predecessor
and that one was never rendered (or is not part of the run selection), the run fails loudly instead
of producing a broken clip.

---

## Second sampling (二采)

For **already cached** segments: read the first-pass AV latent → upscale with `upscale_model`
(`Minimax H3 Latent Upscaler Opt (3D) [Model]`) → re-seam using the context length the first pass
actually pinned → sample on the second pass's own schedule (default: euler, 3 steps; when wired, the
connected SIGMAS plus the sampler from 高级采样) → write the result into a separate `seg2_*` file
group with its own `segment_slots_2nd.json`, so the first pass is never overwritten → export in
batches of adjacent segments so decoded frames do not all sit in memory at once.

`upscale_model` is mandatory: without it the run returns an error report instead of silently
exporting the original resolution.

Trigger: the node's “二次采样” picker writes a one-shot flag into `timeline.output.secondSample` and
queues the prompt. That run does the second pass only — it never mixes with a first-pass run.

---

## Segment export

The node's “分段导出” picker lists segments (those without cache are greyed out) and writes a
one-shot flag into `timeline.output.segmentExport`.

- **Mode**: `piecewise` (one video per checked segment) / `continuous` (checked segments that sit
  next to each other are stitched into one video).
- **Cache source**: `1st` (`seg_*`) or `2nd` (`seg2_*`).
- The result goes **straight to the node's `images` / `audio` outputs** — nothing is written to disk;
  each clip is one `images` entry.
- During a second sample, adjacent segments marked “reference previous” are merged into one
  indivisible run range.

---

## Audio

`timeline.output.audioMode`:

| Mode | Behaviour |
|---|---|
| `generate` (default) | the model's own soundtrack |
| `source` | the source clip's audio (only meaningful for `v2v` / `rv2v`; other tasks fall back to `generate`) |
| `mute` | silent output (44.1 kHz empty track) |

When the merged layout is used, the per-group audio is concatenated along the timeline, so the tail
is never silent.

**Extract audio** stores PCM under `<node cache dir>/audio_extract/`, named by **entry id** (never a
content hash, never a timeline position), so re-running or editing prompts cannot invalidate it and
“clear cache” cannot delete it. Reordering moves entries with their card; deleting a card deletes its
entries. **One entry per segment + source** (keyed by `(seg_id, variant)`, so extracting both passes
leaves one row for each): re-extracting overwrites the previous take (same id, files replaced in place)
so nothing piles up, and **rows already on disk converge too** — opening the Audio tab or deleting a
card drops duplicates (a「Keep audio」-pinned take wins, otherwise the newest) together with their
files. While the batch toolbar offers the button, the duplicate one in the main toolbar hides itself.
Use Clear-all to drop every extracted entry at once.

A segment that cannot be extracted says **why**, per segment: nothing rendered for that pass
(`no-cache`), only an AV latent cached and an HTTP request has no audio VAE (`latent-only`), the cached
clip carries no audio track (silent render — `no-audio-track`), the cache could not be read
(`unreadable`), or the segment is not in the timeline the backend parsed (`not-in-plan`).

**Retain audio** is a switch on the card head (next to「From prev」). Phase 2 VAE-encodes that segment's
extracted audio into the AV latent's audio stream and zeroes that stream's `noise_mask`, so the UNet
conditions on it but can never re-draw it; Phase 3 skips the audio VAE decode and muxes the *same* PCM
back, so you hear exactly that clip — no round-trip through the audio VAE. It keeps the newest
extraction; with none extracted yet, the switch tells you to extract first.

---

## Audio validity check (ASR)

Wire an `asr_model` (`T8_MOSS_ModelLoader` from `Comfyui-MOSS-Transcribe-Diarize-T8`, type
`T8_MOSS_TRANSCRIBE_MODEL`) to reuse your loader settings — no run required, since with nothing wired
the check builds that loader's own handle with its defaults. The **r2v toolbar** then offers an
「Audio validity check」button next to「Extract audio」:

1. click it, pick the cache source (1st / 2nd pass) and tick the segments to verify;
2. the backend loads those segments' *cached* audio and compares it against the lines in each
   segment's **current** prompt, speaker by speaker (error rate + speaker alignment);
3. the verdict opens in a dialog — nothing is written to `report` and nothing is regenerated.

A segment with no cached audio is listed as skipped rather than counted as a pass, so the workflow
is: generate first, then re-word the prompt and re-check whenever you like.

A line must be written exactly like this:

```
<Subject 1> (S1) says: <d>[Chinese] 师尊，你一直说你是毒修。</d>
```

Anything else is treated as plain text (no chip, no expectation). Concatenated audio longer than
30 s automatically switches to the chunked long-audio route; the report always names the route it
took.

See [`docs/asr_check.md`](docs/asr_check.md).

---

## Cache

Unified root:

```
output/minimax_director_opt_cache/<workflow slug>/node_<node id>/
```

File-name prefixes tell the artefacts apart (`<hash>` is a segment's **content** fingerprint, not its
position):

| File | Contents |
|---|---|
| `cond_text_<hash>.pt` | text-encoding cache |
| `seg_<hash>_latent.pt` | first-pass sampled latent |
| `seg_<hash>_clip.mp4` | first-pass rendered clip |
| `seg_<hash>_frames_ht.mp4` (+`.json`) | head/tail seam window (crf 12) |
| `seg_<hash>_audio.pt` | audio latent |
| `seg_<hash>_meta.json` / `_handoff.json` | fingerprint and handoff info |
| `segment_slots.json` | position → file-group map (first pass) |
| `seg2_<hash>_*` / `segment_slots_2nd.json` | the second pass's own set |
| `seg_XXXX_scratch_*.pt` | per-run scratch files |
| `audio_extract/` | extracted audio (own lifecycle; only Clear-all touches it) |
| `_vit/` | global ViT (vision tower) output cache, shared across workflows, capped at 8 GB |

The fingerprint covers prompt, task, canvas, fps, reference files, source-video identity
(relative path + size + mtime), continuity flags and the pipeline version — changing any of them
re-renders that segment.

Two node buttons:

- **Clear cache** — drops the text-encoding cache, batch scratch files and legacy `*_frames_ht.pt`
  windows; rendered segments survive, and the Extract-audio store is left alone.
- **Clear all node cache** — additionally deletes every `seg_*` file **and the whole
  `audio_extract/` store** (manifest included), forcing a full re-render as well as a re-extract; the
  timeline's Keep-audio pins are cleared too, since those entries no longer exist.

Both confirm first (showing workflow id, node id and what will be deleted) and then POST
`/minimax/director_opt/clear_cache` (`clear_all: true/false`).

> Caches are namespaced by workflow name. If the hidden `workflow_name` widget is empty, the cache
> lands in a bare `node_<id>/` directory, which can grey out “align to next” and make exports empty.
> The frontend fills this field — do not edit it by hand.

---

## Director pack

`.mmxpack.zip` bundles the timeline JSON plus reference images / videos / audios and the source
video. The on-disk layout matches the upstream `ComfyUI_MiniMaxH3_Director` pack, so packs can be
imported both ways.

- Export: `POST /minimax/director_opt/export_pack` → `GET .../download_pack?filename=...`
  (name like `MiniMaxH3DirectorOpt-<task>-<timestamp>.mmxpack.zip`; over 500 MB asks for a second
  confirmation).
- Import: `POST /minimax/director_opt/import_pack` (large files upload in 8 MiB chunks).

---

## HTTP routes & WebSocket events

Route prefix: `/minimax/director_opt` (`lib/constants.py`).

| Method | Path | Purpose |
|---|---|---|
| POST | `/upload_chunk` | chunked upload (8 MiB) |
| POST | `/probe_video` | probe source fps / frame count / size / duration |
| GET | `/list_input_media` | list `input` media (`includeCache=1` also lists generated clips) |
| POST | `/detect_shots` | smart split |
| POST | `/prepare_reference_audio_chunk`, `/extract_reference_audio` | reference audio upload / extraction |
| POST | `/audio_extract_status`, `/audio_extract`, `/audio_extract_list`, `/audio_extract_remove` | extracted-audio CRUD |
| GET | `/audio_extract_file` | download an extracted entry |
| POST | `/segment_export_status` | segment export availability |
| POST | `/segment_export` | segment export |
| GET | `/segment_clip` | stream one segment's clip |
| POST | `/second_sample_status` | second-sample availability |
| POST | `/align_to_next_status` | “align to next” availability |
| POST | `/remove_segment_slot` | drop one segment's cache files |
| POST | `/asr_check_status` | audio validity check availability: which segments have audio for the chosen source, whether a model is ready |
| POST | `/asr_check` | grade the picked segments' cached audio against their current prompts (nothing is regenerated) |
| POST | `/clear_cache` | clear cache |
| POST | `/export_pack`, `/import_pack`; GET `/download_pack` | director pack |

WebSocket events:

| Event | Payload highlights |
|---|---|
| `minimax_director_opt_progress` | `node_id`, `segment`, `segment_total`, `phase` (`prepare` / `context_encode` / `sample` / `decode`), `phase_value`, `overall_value`, … |
| `minimax_director_opt_preview` | `node_id`, `segment_index`, `image_b64`, `width`, `height`, optional `step` / `total_steps` |

---

## Limits & troubleshooting

| Symptom / limit | Reason |
|---|---|
| A single diffusion segment maxes out at 512 frames | frame counts must land on the `17k+5` grid, minimum 5 (124 ≈ 5 s @ 24 fps) |
| Shortest video-timeline segment is 4 frames; batch / fl2v minimum is 5 | frontend constants |
| ≤ 9 reference images, ≤ 3 reference videos, ≤ 3 reference audios | MiniMax H3 `ReferenceToVideo` limits |
| Equal split 2–64; canvas aligned to 32; fps 1–240 | frontend constants |
| Uploads ≤ 95 MiB use `/upload/image`, larger ones use 8 MiB chunks | `core/upload.js` |
| Continuity context can only be 5 / 22 / 39 / 56 and needs ≥ 2 segments | `h3_motion_context.py` |
| `use_sigmas` on but nothing wired | logged, then falls back to the default schedule |
| The frame rate mysteriously becomes 240 | 240 is the widget's ceiling, so something fed it a value above it (most often a source-video probe: some containers report their **time base** `1000/1` as the frame rate, and VFR files report `avg_frame_rate = 0/0`). A probed rate is now cross-checked against `frame count / duration` and simply not applied — with a warning — when it disagrees; clamping a frame rate also logs the raw value, so the console names the culprit |
| Second sample without `upscale_model` | returns an error report, never a silent same-resolution export |
| Widget values shifted after loading an old workflow | the frontend strips the removed `segment_images` output and repairs stale widget values; absurd width/height/total_frames are clamped back to safe defaults with a warning — save the workflow again to fix it permanently |
| A few widgets **reset to their defaults on every refresh** (e.g. `use_sigmas` / `second_seed`) | that workflow was saved *while its values were shifted*: the wrong values were frozen **by name** in `widgets_values_named`, so every load restores them faithfully. Loading now detects and heals it (a group header holding anything but its own label is the proof → that widget and everything after it go back to defaults, and the console names them); save the workflow once and it stops coming back |
| `/minimax/director_opt/*` returns 404 | PromptServer was not ready at import time; restart ComfyUI |
| `import cv2` fails | OpenCV is intentionally not a dependency; decoding uses PyAV |
| Washed-out / inverted colours | do not post-process `MiniMax H3 Fast Video VAE Opt` output with a non-H3 VAE path; the wrapper deliberately skips ComfyUI's `*2-1` transform |

---

## License & credits

Apache License 2.0 — see `LICENSE`.

- Upstream: `AIMixer/ComfyUI_MiniMaxH3_Director` (author: AI搅拌手), itself based on ComfyUI's
  official MiniMax H3 support (PR #15224 / #15228).
- Opt fork: **fvdfggh** — <https://github.com/fvdfggh/minimax-h3-director-performance-optimization>
