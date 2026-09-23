"""Source video loading from ComfyUI's input folder (VHS-style file references).

The implementation is split into three layers; this module re-exports what the rest
of the plugin imports (see ``__all__``):

* :mod:`av_probe`        — resolve paths, ffprobe / PyAV metadata, frame counts,
  raw frame reads (PyAV only; this build has no OpenCV)
* :mod:`video_decode`    — decode a file or clip into model-ready tensors
* :mod:`timeline_frames` — logical→source frame mapping and the segment / timeline
  readers built on it

New code should import from the layer it needs.
"""

from .av_probe import (
    av_video_meta,
    peek_video_size,
    probe_video_clip,
    resolve_video_path,
)
from .timeline_frames import (
    load_timeline_segment,
    logical_frame_count,
    logical_frame_map,
    resolve_logical_frame_entry,
    video_clips_from_timeline,
)
from .video_decode import decode_video_frames, load_reference_video_clip

__all__ = [
    # av_probe
    "resolve_video_path",
    "av_video_meta",
    "peek_video_size",
    "probe_video_clip",
    # video_decode
    "decode_video_frames",
    "load_reference_video_clip",
    # timeline_frames
    "logical_frame_map",
    "logical_frame_count",
    "resolve_logical_frame_entry",
    "video_clips_from_timeline",
    "load_timeline_segment",
]


