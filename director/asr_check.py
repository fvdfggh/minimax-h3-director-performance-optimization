"""「音频有效性校验」— grade the generated soundtrack against the prompt with ASR.

What it does
------------

Triggered by the「音频有效性校验」button the node shows in r2v mode. An ASR model is
needed — ``T8_MOSS_TRANSCRIBE_MODEL``, from the sibling
``Comfyui-MOSS-Transcribe-Diarize-T8`` pack. Wiring it to the node and running once
hands us *your* loader settings (see :mod:`director.asr_runtime`); with nothing
wired we build the pack's handle ourselves with its defaults
(:func:`default_model_handle`), so the button works without any run.
The picked segments' *cached* audio is then:

1. concatenated into one timeline (timeline order — that is the order the
   audience hears the speakers in);
2. transcribed with speaker diarization;
3. compared against the **speaking lines** those segments' *current* prompts ask
   for — per speaker, plus a speaker-identity check.

Nothing is generated here, and a segment with no cached audio is reported as
missing rather than treated as a pass.

Speaking-line grammar
---------------------

The prompt must spell a line out exactly like this::

    <Subject 1> (S1) says: <d>[Chinese] 师尊，你一直说你是毒修。</d>

* ``<Subject N>`` — which reference subject is speaking;
* ``(SN)``        — the speaker id the line belongs to;
* ``<d>[lang]…</d>`` — the spoken text and its language tag.

Anything that does not match byte-for-byte is **not** a line: the front end
leaves it as plain text (no chip) and this module does not build an expectation
out of it. A typo can therefore never silently become a requirement.

Speaker numbering under long audio
----------------------------------

The recogniser numbers speakers by *its own* first-appearance order, which has
nothing to do with the order the prompt introduces them. So the mapping is done
by rank: the prompt's speakers are ordered by their first line, the recognised
speakers by their first utterance, and rank *i* is matched to rank *i*. That is
what「根据 prompt 里面的说话人首次说话的顺序，确定哪个是说话人几」asks for.

``S00`` is the pack's「未分配到说话人」marker, not person zero: it is kept out of
the ranking and reported as「未知说话人」, because letting it win rank 0 shifts
every real speaker one slot down.

Long timelines
--------------

A concatenated run easily outgrows what the single-shot ``run_transcription``
covers. Past :data:`LONG_AUDIO_SECONDS` the timeline is handed to the pack's
chunked ``transcribe_long_audio`` instead, and a result that still comes back
truncated (``metadata["possibly_truncated"]``) is called out in the report —
missing tail speech would otherwise be scored as「这句台词没说」. The report
always names the route it took.

Scoring
-------

Comparison is alignment-free per speaker: the expected lines of a speaker are
concatenated in prompt order and matched against that speaker's recognised
utterances in time order. Errors are Levenshtein edit distance over tokens,
where a CJK character is one token and a latin/digit run is another — Chinese
therefore scores CER (character error rate) and English scores WER, without
pulling in a segmentation dependency. A speaker whose expected text is empty is
skipped rather than handed a free pass.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.asr_check")

#: Custom ComfyUI type emitted by the sibling pack's model loader.
ASR_MODEL_TYPE = "T8_MOSS_TRANSCRIBE_MODEL"
#: Sibling custom-node directory (same ``custom_nodes`` parent as this pack).
ASR_PACKAGE = "Comfyui-MOSS-Transcribe-Diarize-T8"

#: Speaker id the recogniser emits for「未分配到说话人」. It is *not* a person:
#: letting it into the rank mapping (as ``S0``) shifts every real speaker by one
#: and silently mis-grades the whole run, so it is always reported separately.
UNKNOWN_SPEAKER = "S00"

#: Past this length the single-shot ``run_transcription`` starts dropping tail
#: speech (``metadata["possibly_truncated"]``), which reads as「台词没说」. The
#: pack has a chunked path for that — see :func:`transcribe`.
LONG_AUDIO_SECONDS = 30.0

#: Strict speaking-line grammar. ``re.DOTALL`` so a line may wrap.
#: Mirror of ``SPEECH_RE`` in ``web/js/minimax_prompt_mentions.js`` — the two
#: have no shared source, so any change must be applied to both.
SPEECH_RE = re.compile(
    r"<Subject\s+(?P<subject>\d+)\s*>\s*"
    r"\(\s*S(?P<speaker>\d+)\s*\)\s*says\s*:\s*"
    r"<d>\s*\[(?P<lang>[^\]]*)\]\s*(?P<text>.*?)\s*</d>",
    re.IGNORECASE | re.DOTALL,
)

#: Anything in these ranges is scored per character (CJK / kana / hangul).
_CJK_RE = re.compile(
    "["
    "\u3040-\u30ff"      # kana
    "\u3400-\u4dbf"      # CJK ext A
    "\u4e00-\u9fff"      # CJK
    "\uf900-\ufaff"      # compat ideographs
    "\uac00-\ud7af"      # hangul
    "]"
)

#: Error rate at or below which a speaker's line counts as delivered.
PASS_RATE = 0.25


@dataclass(frozen=True)
class SpeechLine:
    """One ``<Subject N> (SN) says: <d>[lang] text</d>`` block."""

    subject: int
    speaker: int
    lang: str
    text: str
    #: 0-based order of appearance across the whole prompt run.
    order: int
    #: Verbatim source block, so a report can quote it exactly.
    raw: str


# --------------------------------------------------------------------------
# Prompt side
# --------------------------------------------------------------------------

def parse_speech_lines(prompt: str, *, order_offset: int = 0) -> list[SpeechLine]:
    """Every well-formed speaking line in ``prompt``, in appearance order."""
    if not prompt:
        return []
    out: list[SpeechLine] = []
    for i, match in enumerate(SPEECH_RE.finditer(str(prompt))):
        text = re.sub(r"\s+", " ", match.group("text") or "").strip()
        out.append(
            SpeechLine(
                subject=int(match.group("subject")),
                speaker=int(match.group("speaker")),
                lang=(match.group("lang") or "").strip(),
                text=text,
                order=order_offset + i,
                raw=match.group(0),
            )
        )
    return out


def collect_prompt_texts(plan: Any) -> list[tuple[int, str]]:
    """``(segment index, effective prompt)`` for every segment that ran.

    Timeline order, so「说话人首次说话的顺序」is the order the audience hears
    them — not the order of a partial run list.
    """
    wanted = getattr(plan, "run_indices", None)
    wanted = {int(i) for i in wanted} if wanted else None
    rows: list[tuple[int, str]] = []
    for seg in (getattr(plan, "segments", None) or []):
        idx = int(getattr(seg, "index", -1))
        if wanted is not None and idx not in wanted:
            continue
        text = str(getattr(seg, "prompt", "") or "")
        if not text.strip():
            text = str(getattr(plan, "global_prompt", "") or "")
        rows.append((idx, text))
    return rows


def collect_speech_lines(plan: Any) -> list[SpeechLine]:
    """Speaking lines of every running segment, numbered across the whole run."""
    out: list[SpeechLine] = []
    for _idx, text in collect_prompt_texts(plan):
        out.extend(parse_speech_lines(text, order_offset=len(out)))
    return out


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """CJK → one token per character, other scripts → one token per word."""
    out: list[str] = []
    buf: list[str] = []
    for ch in str(text or ""):
        if _CJK_RE.match(ch):
            if buf:
                out.append("".join(buf).lower())
                buf = []
            out.append(ch)
        elif ch.isalnum():
            buf.append(ch)
        elif buf:
            out.append("".join(buf).lower())
            buf = []
    if buf:
        out.append("".join(buf).lower())
    return out


def _levenshtein(ref: list[str], hyp: list[str]) -> int:
    if ref == hyp:
        return 0
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(
                min(
                    prev[j] + 1,
                    cur[j - 1] + 1,
                    prev[j - 1] + (0 if r == h else 1),
                )
            )
        prev = cur
    return prev[-1]


def error_rate(expected: str, recognized: str) -> tuple[int, int, float]:
    """``(errors, reference_length, rate)`` — CER for CJK, WER otherwise.

    Tokenisation drops every punctuation mark (see :func:`tokenize`), so
    「，」vs「、」是零成本差异。参考长度是**token 数**（中文即字数），错误数是一次
    不带对齐的编辑距离。
    """
    ref = tokenize(expected)
    hyp = tokenize(recognized)
    if not ref:
        return (len(hyp), 0, 0.0 if not hyp else 1.0)
    errs = _levenshtein(ref, hyp)
    return (errs, len(ref), errs / len(ref))


#: Backtracking the edit distance is O(n·m) in memory; past this the breakdown is
#: skipped and only the rate is reported (a paragraph-long line still fits).
_OPS_LIMIT = 250_000


def edit_op_counts(expected: str, recognized: str) -> tuple[int, int, int] | None:
    """``(substitutions, deletions, insertions)`` behind the error rate.

    Worth reporting next to the number because「错误率 43%」reads as「听错了」while the
    usual cause is a **missing tail** — the recogniser dropped it or handed it to
    another speaker, and that shows up as deletions, not substitutions.
    ``None`` when the input is too large to break down.
    """
    ref = tokenize(expected)
    hyp = tokenize(recognized)
    if len(ref) * len(hyp) > _OPS_LIMIT:
        return None
    n, m = len(ref), len(hyp)
    dist = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dist[i][0] = i
    for j in range(m + 1):
        dist[0][j] = j
    for i in range(1, n + 1):
        row, prev = dist[i], dist[i - 1]
        for j in range(1, m + 1):
            row[j] = min(
                prev[j] + 1,
                row[j - 1] + 1,
                prev[j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1),
            )
    subs = dels = inss = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and dist[i][j] == dist[i - 1][j - 1]:
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and dist[i][j] == dist[i - 1][j - 1] + 1:
            subs += 1
            i, j = i - 1, j - 1
        elif i > 0 and dist[i][j] == dist[i - 1][j] + 1:
            dels += 1
            i -= 1
        else:
            inss += 1
            j -= 1
    return subs, dels, inss


# --------------------------------------------------------------------------
# Audio side — concatenation
# --------------------------------------------------------------------------

def _waveform_of(audio: Any):
    if not isinstance(audio, dict):
        return None
    wave = audio.get("waveform")
    try:
        import torch

        if not isinstance(wave, torch.Tensor) or int(wave.numel()) <= 0:
            return None
    except Exception:  # pragma: no cover - defensive
        return None
    return wave


def _pick_sample_rate(audios: list[Any]) -> int:
    """First non-zero ``sample_rate`` across the clips — the timeline's rate.

    Taking it from the first clip alone leaves ``sr`` at 0 whenever that clip
    carries no rate, and the clip already appended is then concatenated at the
    next clip's rate — a silently stretched timeline.
    """
    for item in audios or ():
        if not isinstance(item, dict):
            continue
        rate = int(item.get("sample_rate") or 0)
        if rate > 0:
            return rate
    return 0


def concat_audio(
    audios: list[Any],
    *,
    warnings: list[str] | None = None,
    spans: dict[int, tuple[int, int]] | None = None,
) -> dict[str, Any] | None:
    """Concatenate AUDIO dicts into one timeline (``None`` when all silent).

    ``warnings`` — when given, collects one line per clip that could not be
    merged. A dropped clip has to be visible in the report: a verdict computed
    over a partial soundtrack looks exactly like a real one.

    ``spans`` — when given, records ``{position in audios: (first, last+1) sample}``
    for every clip that made it into the merge. The timeline is a plain
    concatenation, so these are what map a recognised segment's timestamps back to
    the clip it came from (see :func:`segment_asr_details`).
    """
    import torch

    sr = _pick_sample_rate(audios)
    waves: list[Any] = []
    dropped = 0
    cursor = 0
    for pos, item in enumerate(audios or ()):
        wave = _waveform_of(item)
        if wave is None:
            continue
        item_sr = int(item.get("sample_rate") or 0)
        if item_sr and sr and item_sr != sr:
            resampled = _resample(wave, item_sr, sr)
            if resampled is None:
                dropped += 1
                continue
            wave = resampled
        wave = wave.detach().cpu().float()
        if spans is not None:
            spans[pos] = (cursor, cursor + int(wave.shape[-1]))
        cursor += int(wave.shape[-1])
        waves.append(wave)

    if dropped and warnings is not None:
        warnings.append(f"{dropped} 段音频因采样率不一致且重采样失败，未计入识别")

    if not waves:
        return None

    channels = max(int(w.shape[1]) if w.ndim == 3 else 1 for w in waves)
    normed = []
    for w in waves:
        if w.ndim == 2:                      # [C, T] → [1, C, T]
            w = w.unsqueeze(0)
        if w.ndim != 3:
            w = w.reshape(1, 1, -1)
        c = int(w.shape[1])
        if c < channels:
            if c == 1:
                w = w.expand(1, channels, int(w.shape[-1])).contiguous()
            else:
                w = torch.cat(
                    [w, w.new_zeros(1, channels - c, int(w.shape[-1]))], dim=1
                )
        elif c > channels:
            w = w[:, :channels, :]
        normed.append(w)
    merged = torch.cat(normed, dim=-1)
    return {"waveform": merged.contiguous(), "sample_rate": int(sr or 32000)}


def _as_mono_2d(wave):
    """``[1, T]`` float tensor — the shape both resamplers accept.

    Only used on the resample path: a clip that needs resampling is mixed down
    instead of being dropped, and :func:`concat_audio` re-expands it to the
    timeline's channel count afterwards.
    """
    if wave.ndim == 3:
        wave = wave[0]
    if wave.ndim == 1:
        return wave.unsqueeze(0)
    if wave.ndim == 2 and int(wave.shape[0]) > 1:
        return wave.mean(dim=0, keepdim=True)
    return wave


def _resample(wave, src_sr: int, dst_sr: int):
    """Resample to ``dst_sr``, or ``None`` (caller skips the clip) on failure.

    torchaudio first, then ``soxr`` — the same fallback the sibling pack's
    ``resample_waveform`` uses. Without the second route a missing torchaudio
    silently drops every mismatched clip, and a verdict over a partial
    soundtrack is indistinguishable from a real one.
    """
    if src_sr == dst_sr or src_sr <= 0 or dst_sr <= 0:
        return wave
    try:
        import torchaudio

        return torchaudio.functional.resample(wave, src_sr, dst_sr)
    except Exception as exc:  # pragma: no cover - torchaudio is usually present
        log.warning(
            "音频有效性校验: %dHz → %dHz torchaudio 重采样失败 (%s)，改用 soxr。",
            src_sr, dst_sr, exc,
        )
    try:
        import numpy as np
        import soxr
        import torch

        mono = _as_mono_2d(wave).numpy().astype(np.float32, copy=False)
        values = soxr.resample(mono, src_sr, dst_sr)
        return torch.from_numpy(np.asarray(values, dtype=np.float32)).unsqueeze(0)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning(
            "音频有效性校验: %dHz → %dHz 重采样失败 (%s)，该段音频未计入识别。",
            src_sr, dst_sr, exc,
        )
        return None


# --------------------------------------------------------------------------
# ASR side — the sibling pack
# --------------------------------------------------------------------------

def _asr_package_dir() -> str:
    """Absolute path of the sibling ASR pack.

    ``.../custom_nodes/<this pack>/director/asr_check.py`` →
    ``.../custom_nodes/<ASR_PACKAGE>`` — both packs share the ``custom_nodes``
    parent, so this survives a renamed / relocated ComfyUI root.
    """
    here = os.path.dirname(os.path.abspath(__file__))        # .../director
    pack_root = os.path.dirname(here)                        # .../<this pack>
    return os.path.join(os.path.dirname(pack_root), ASR_PACKAGE)


def _ensure_asr_package():
    """Make ``ASR_PACKAGE`` importable under its own name.

    ComfyUI loads a custom node under a *path-derived* ``sys.modules`` key — see
    ``nodes.load_custom_node``, which registers ``module_path.replace(".", "_x_")``
    — and never puts ``custom_nodes/`` on ``sys.path``. ``importlib.import_module``
    therefore could never resolve the directory name, so every ASR run died with
    ``No module named 'Comfyui-MOSS-Transcribe-Diarize-T8'`` even though the pack
    was installed and its model socket was wired.
    Binding a namespace package whose ``__path__`` points at that directory
    restores the plain ``ASR_PACKAGE.<sub>`` imports — and, unlike putting
    ``custom_nodes/`` on ``sys.path``, it does *not* execute the pack's
    ``__init__.py`` a second time (which would re-register all of its nodes).
    """
    existing = sys.modules.get(ASR_PACKAGE)
    if existing is not None and getattr(existing, "__path__", None):
        return existing
    pkg_dir = _asr_package_dir()
    if not os.path.isdir(pkg_dir):
        raise ImportError(f"未找到 ASR 节点目录 {ASR_PACKAGE}（期望位置：{pkg_dir}）")
    spec = importlib.machinery.ModuleSpec(ASR_PACKAGE, None, is_package=True)
    module = importlib.util.module_from_spec(spec)
    module.__path__ = [pkg_dir]
    sys.modules[ASR_PACKAGE] = module
    return module


def _load_t8_module(suffix: str):
    """``ASR_PACKAGE.suffix`` module, or raise ``ImportError``."""
    try:
        _ensure_asr_package()
        return importlib.import_module(f"{ASR_PACKAGE}.{suffix}")
    except Exception as exc:
        raise ImportError(
            f"未找到 ASR 节点 {ASR_PACKAGE}（或其 {suffix} 不可用）：{exc}"
        ) from exc


def _load_t8_runtime():
    """``runtime.inference`` of :data:`ASR_PACKAGE`, or raise ``ImportError``."""
    return _load_t8_module("runtime.inference")


def audio_seconds(audio: dict[str, Any]) -> float:
    """Timeline length of an AUDIO dict, ``0.0`` when it cannot be read."""
    try:
        wave = audio.get("waveform")
        sr = int(audio.get("sample_rate") or 0)
        return float(wave.shape[-1]) / float(sr) if sr > 0 else 0.0
    except Exception:  # pragma: no cover - defensive
        return 0.0


def _progress_sink() -> tuple[Any, Any]:
    """Our own ``(progress, cancellation)`` callbacks for the ASR runtime.

    The pack reports progress through ComfyUI's plumbing by default
    (``runtime.inference._comfy_runtime_callbacks``): first the new
    ``ComfyAPISync().execution.set_progress``, then — when that raises — the
    legacy ``comfy.utils.ProgressBar``. The legacy route goes through ComfyUI's
    global progress hook, which reads ``PromptServer.last_prompt_id``
    (``main.py:hijack_progress``), an attribute the queue loop only assigns once a
    prompt starts. We run from an HTTP request with no prompt context, so that
    raises ``AttributeError`` — and it lands *inside* transcription, killing the
    check. Handing the pack our own sink keeps it independent of the UI's progress
    plumbing: nobody is waiting on a progress bar for a button-triggered check.

    Cancellation stays wired to ComfyUI's interrupt flag, which is what makes
    「取消」able to abort a long transcription.
    """
    def report(_value: int, _total: int) -> None:
        return None

    def cancelled() -> bool:
        try:
            import comfy.model_management
        except Exception:  # pragma: no cover - no ComfyUI runtime
            return False
        comfy.model_management.throw_exception_if_processing_interrupted()
        return False

    return report, cancelled


def _ensure_progress_context() -> None:
    """Make ComfyUI's progress hook usable from outside a prompt (idempotent).

    The chunked path (:func:`_transcribe_long`) has no callback injection point, so
    it always reports through that hook. ``last_prompt_id`` is missing on a server
    that has not executed a prompt yet — exactly the state right after a restart,
    which is when the button gets used. Seeding it with an empty id is honest (we
    are not a prompt) and only fills a hole: the queue loop overwrites it with the
    real id as soon as a run starts.
    """
    try:
        from server import PromptServer

        server = getattr(PromptServer, "instance", None)
    except Exception:  # pragma: no cover - no ComfyUI runtime
        return
    if server is None:
        return
    if not hasattr(server, "last_prompt_id"):
        server.last_prompt_id = ""


def _run_single_shot(model_handle: Any, audio: dict[str, Any]) -> Any:
    """Single-shot transcription with injectable callbacks.

    ``runtime.run_transcription`` hard-codes the pack's ComfyUI-progress callbacks,
    so we call the layer below it, where ``progress_callback`` /
    ``cancellation_callback`` are parameters (the pack skips its own callbacks when
    both are given). Falls back to the public entry point if that layer is not
    available in the installed pack version.
    """
    runtime = _load_t8_runtime()
    entry = getattr(runtime, "run_transcription_samples", None)
    to_numpy = getattr(runtime, "comfy_audio_to_numpy", None)
    if entry is None or to_numpy is None:
        return _transcribe_pack_default(runtime, model_handle, audio)
    report, cancelled = _progress_sink()
    try:
        return entry(
            model_handle,
            to_numpy(audio, getattr(runtime, "TARGET_SAMPLE_RATE", 16000)),
            None,
            max_new_tokens=0,
            silence_policy="warn",
            preflight_backend="webrtc",
            vad_aggressiveness=2,
            retry_policy="invalid_format",
            progress_callback=report,
            cancellation_callback=cancelled,
        )
    except TypeError:  # pragma: no cover - older/newer pack signature
        log.info("音频有效性校验: 该版本 T8 不支持注入回调，回退其默认入口。")
        return _transcribe_pack_default(runtime, model_handle, audio)


def _transcribe_pack_default(runtime: Any, model_handle: Any, audio: dict[str, Any]) -> Any:
    """The pack's public single-shot entry point (uses its own progress plumbing)."""
    return runtime.run_transcription(
        model_handle,
        audio,
        None,
        max_new_tokens=0,
        silence_policy="warn",
        preflight_backend="webrtc",
        vad_aggressiveness=2,
        retry_policy="invalid_format",
    )


def _transcribe_long(model_handle: Any, audio: dict[str, Any]) -> Any:
    """Chunked path for a long timeline — the pack's own long-audio runtime."""
    long_audio = _load_t8_module("runtime.long_audio")
    adapter = _load_t8_module("vendor.moss_transcribe_diarize.audio_adapter")
    samples = adapter.comfy_audio_to_numpy(audio, adapter.TARGET_SAMPLE_RATE)
    payload, _report = long_audio.transcribe_long_audio(
        model_handle,
        samples,
        None,
        sample_rate=adapter.TARGET_SAMPLE_RATE,
        max_new_tokens_per_chunk=0,
        silence_policy="warn",
        preflight_backend="webrtc",
        vad_aggressiveness=2,
        retry_policy="invalid_format",
        checkpoint_mode="off",
        # Chunking namespaces speakers per chunk (``S001001``), so the same
        # voice would arrive as several speakers. Overlap linking stitches the
        # shared seams back together — see :func:`map_speakers`.
        speaker_link_mode="overlap_only",
    )
    return payload


def transcribe(model_handle: Any, audio: dict[str, Any]) -> tuple[Any, str]:
    """Transcribe with the sibling pack. Returns ``(payload, mode)``.

    A timeline longer than :data:`LONG_AUDIO_SECONDS` goes through the chunked
    path: the single-shot ``run_transcription`` truncates long audio, and a
    truncated transcript reads as「这句台词没说」— a false failure. Any problem
    on the chunked route falls back to the single-shot one rather than losing
    the check entirely.
    """
    seconds = audio_seconds(audio)
    # The chunked path cannot take our callbacks, so make sure the progress hook
    # it will use can actually run (see _ensure_progress_context).
    _ensure_progress_context()
    if seconds > LONG_AUDIO_SECONDS:
        try:
            return _transcribe_long(model_handle, audio), f"分块长音频（{seconds:.1f}s）"
        except Exception as exc:
            log.warning(
                "音频有效性校验: 分块长音频不可用（%.1fs），回退单次识别 (%s)",
                seconds, exc,
            )
    return _run_single_shot(model_handle, audio), f"单次（{seconds:.1f}s）"


def _asr_segments(payload: Any) -> list[dict[str, Any]]:
    raw = getattr(payload, "segments", None)
    if raw is None and isinstance(payload, dict):
        raw = payload.get("segments")
    if not raw:
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            out.append(item)
        else:  # dataclass row
            out.append(
                {
                    "start": float(getattr(item, "start", 0.0) or 0.0),
                    "end": float(getattr(item, "end", 0.0) or 0.0),
                    "speaker": str(getattr(item, "speaker", "") or ""),
                    "text": str(getattr(item, "text", "") or ""),
                }
            )
    return out


def _speaker_key(speaker: Any) -> str:
    """Normalise a recogniser speaker tag; unassigned voices collapse to ``S00``.

    The pack writes ``S00`` for「未分配到说话人」. Normalising it to ``S0`` used
    to let it win rank 0 in :func:`map_speakers` and push every real speaker one
    slot down — the whole verdict came out shifted.
    """
    text = str(speaker or "").strip()
    match = re.search(r"(\d+)", text)
    if not match:
        return text or UNKNOWN_SPEAKER
    value = int(match.group(1))
    return UNKNOWN_SPEAKER if value == 0 else f"S{value}"


# --------------------------------------------------------------------------
# Mapping recognised speakers onto prompt speakers
# --------------------------------------------------------------------------

def map_speakers(
    lines: list[SpeechLine],
    segments: list[dict[str, Any]],
) -> dict[str, int]:
    """``{recognised speaker key: prompt speaker id}``, matched by first-speech rank.

    Prompt speakers are ordered by the first line they speak, recognised speakers
    by their first utterance in time; rank *i* meets rank *i*. Extra recognised
    speakers (more voices than the prompt wrote) are simply left unmapped and
    reported as unexpected. :data:`UNKNOWN_SPEAKER` never takes part: it is not
    a person and must not consume a rank.
    """
    prompt_order: dict[int, int] = {}
    for line in lines:
        prompt_order.setdefault(int(line.speaker), int(line.order))
    ranked_prompt = [s for s, _ in sorted(prompt_order.items(), key=lambda kv: kv[1])]

    asr_first: dict[str, float] = {}
    for seg in segments:
        key = _speaker_key(seg.get("speaker"))
        if key == UNKNOWN_SPEAKER:
            continue
        start = float(seg.get("start") or 0.0)
        if key not in asr_first or start < asr_first[key]:
            asr_first[key] = start
    ranked_asr = [k for k, _ in sorted(asr_first.items(), key=lambda kv: (kv[1], kv[0]))]

    return {
        asr_key: ranked_prompt[i]
        for i, asr_key in enumerate(ranked_asr)
        if i < len(ranked_prompt)
    }


# --------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------

def _clip(text: str, limit: int = 40) -> str:
    flat = re.sub(r"\s+", " ", str(text or "")).strip()
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def build_asr_report(
    plan: Any,
    audio: dict[str, Any] | None,
    payload: Any,
    *,
    mode: str = "单次",
    warnings: list[str] | None = None,
) -> str:
    """Human-readable verdict for the node's ``report`` output."""
    lines = collect_speech_lines(plan)
    segments = _asr_segments(payload)
    metadata = getattr(payload, "metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    duration = 0.0
    try:
        duration = float(metadata.get("audio_duration_seconds", 0.0))
    except Exception:  # pragma: no cover - defensive
        duration = 0.0

    head = [
        "音频有效性校验 (ASR)",
        f"- 识别方式: {mode}",
        f"- 识别音频: {duration:.2f}s"
        + (f" @ {int(audio.get('sample_rate') or 0)}Hz" if audio else ""),
        f"- 识别段落: {len(segments)} 段",
        f"- 提示词台词: {len(lines)} 条",
    ]
    for note in warnings or ():
        head.append(f"- 注意: {note}")
    # A truncated transcript is missing tail speech, which the per-speaker
    # comparison would score as「这句台词没说」. Flag it instead of failing
    # silently — otherwise the verdict looks like a real one.
    if metadata.get("possibly_truncated"):
        head.append("- 注意: 识别结果可能被截断，后半段台词的判定不可信。")

    if not lines:
        head.append(
            "- 提示词里没有符合格式的台词块，跳过文本比对（格式：<Subject 1> (S1) says: "
            "<d>[Chinese] 台词</d>）。"
        )
        if not segments:
            head.append("- 识别结果为空：生成的音轨没有可辨识的语音。")
        return "\n".join(head)

    if not segments:
        head.append("- 识别结果为空：生成的音轨没有可辨识的语音，台词全部未命中。")
        return "\n".join(head)

    mapping = map_speakers(lines, segments)

    expected: dict[int, list[str]] = {}
    for line in lines:
        expected.setdefault(int(line.speaker), []).append(line.text)

    heard: dict[int, list[str]] = {}
    unexpected: dict[str, list[str]] = {}
    unknown = 0
    for seg in segments:
        key = _speaker_key(seg.get("speaker"))
        if key == UNKNOWN_SPEAKER:
            unknown += 1
            continue
        spoke = int(mapping.get(key, -1))
        if spoke < 0:
            unexpected.setdefault(key, []).append(str(seg.get("text") or ""))
        else:
            heard.setdefault(spoke, []).append(str(seg.get("text") or ""))

    head.append(
        "- 说话人对齐: "
        + (
            ", ".join(f"{key} → S{spoke}" for key, spoke in sorted(mapping.items()))
            if mapping
            else "无"
        )
    )
    if unexpected:
        head.append(
            "- 多余说话人: "
            + ", ".join(sorted(unexpected))
            + "（提示词里没有对应的台词，不参与判定）"
        )
        # Show what they said: a「多余」speaker is where a missing tail usually went,
        # and without its text the per-speaker failure below cannot be attributed.
        for key in sorted(unexpected):
            head.append(f"  · {key} 识别到：「{_clip(''.join(unexpected[key]), 60)}」")
    if unknown:
        head.append(f"- 未知说话人: {unknown} 段（识别器未归类，未计入判定）")

    body: list[str] = []
    worst = 0.0
    ok = True
    for speaker in sorted(expected):
        want = "".join(expected[speaker])
        got = "".join(heard.get(speaker, []))
        if not want.strip():
            # An empty <d></d> block is not a line: scoring it would hand out a
            # free pass (zero errors over zero expected characters).
            body.append(f"- S{speaker}: 期望台词为空，跳过比对")
            continue
        if speaker not in heard:
            ok = False
            body.append(f"- S{speaker}: 未识别到该说话人（期望「{_clip(want)}」）")
            continue
        errs, total, rate = error_rate(want, got)
        worst = max(worst, rate)
        failed = rate > PASS_RATE
        if failed:
            ok = False
        detail = ""
        if failed:
            ops = edit_op_counts(want, got)
            if ops is not None:
                subs, dels, inss = ops
                detail = f" — 替换 {subs}、漏识 {dels}、多出 {inss}"
        # 不达标的一行给更长的摘录：原因通常落在 40 字截断之外。
        limit = 90 if failed else 40
        body.append(
            f"- S{speaker}: 错误率 {rate * 100:.1f}% ({errs}/{total}){detail} "
            f"期望「{_clip(want, limit)}」 识别「{_clip(got, limit)}」"
        )

    verdict = "通过" if ok else f"不通过（阈值 {PASS_RATE * 100:.0f}%，最高 {worst * 100:.1f}%）"
    return "\n".join([*head, *body, f"- 综合判定: {verdict}"])


def _node_outputs(out: Any) -> tuple[Any, ...]:
    """Payload of an ``io.NodeOutput`` across ComfyUI API spellings."""
    result = getattr(out, "result", None)
    if result is None:
        result = getattr(out, "args", None)
    return tuple(result or ())


def default_model_handle() -> Any:
    """Build the ASR model handle from the sibling pack's own loader node.

    Why a route may do this: the handle is **not** a loaded model.
    ``T8_MOSS_ModelLoader`` only resolves the model directory and records the
    inference settings (device / precision / memory policy / attention backend) —
    the weights are loaded lazily inside :func:`transcribe`. So the handle can be
    constructed from an HTTP request without executing a graph, touching the GPU,
    or generating anything, which is what lets「音频有效性校验」work before the
    node has ever run.

    Nothing is downloaded: a missing model raises the pack's own
    「请运行 scripts/download_models.py」message.
    """
    nodes = _load_t8_module("nodes_v3")
    loader = nodes.T8MossModelLoader
    options = list(nodes.model_options())
    model_name = options[0] if options else ""
    if not model_name or model_name == nodes.MISSING_MODEL_OPTION:
        raise RuntimeError(
            "未找到 MOSS 转写模型：请运行 ASR 节点目录中的 scripts/download_models.py，"
            "或把模型放到 models/moss_transcribe_diarize。"
        )
    invalid = loader.validate_inputs(model_name)
    if invalid is not True:
        raise RuntimeError(str(invalid))
    out = loader.execute(
        model_name=model_name,
        device="auto",
        precision="auto",
        release_after_run=False,
        verify_hashes=False,
    )
    handle = _node_outputs(out)[0] if _node_outputs(out) else None
    if handle is None:
        raise RuntimeError("ASR 加载器没有返回模型句柄。")
    log.info("音频有效性校验: 未接线的 ASR 模型按默认参数自动构造句柄（%s）", model_name)
    return handle


def auto_model_available() -> bool:
    """Cheap probe (no hashing, no GPU): could :func:`default_model_handle` work?"""
    try:
        nodes = _load_t8_module("nodes_v3")
        options = list(nodes.model_options())
        if not options or options[0] == nodes.MISSING_MODEL_OPTION:
            return False
        return bool(nodes.resolve_model(options[0]).is_dir())
    except Exception:
        return False


def segment_asr_details(
    plan: Any,
    seg_indices: list[int] | None,
    spans: dict[int, tuple[int, int]],
    sample_rate: int,
    asr_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Per-segment「本段提示词说了什么 / 本段识别到什么」。

    The checked audio is a plain concatenation, so a recognised segment belongs to
    whichever source clip its time range overlaps most. That assignment is what
    lets each card show its own lines instead of the batch verdict repeated on
    every one of them — the batch view pools a speaker's text across all checked
    segments, so it cannot answer「这一段说对了吗」.
    """
    if not seg_indices or sample_rate <= 0:
        return []

    prompt_lines: dict[int, list[SpeechLine]] = {}
    for idx, text in collect_prompt_texts(plan):
        prompt_lines[int(idx)] = parse_speech_lines(text, order_offset=0)

    windows: list[tuple[int, float, float]] = []
    for pos, idx in enumerate(seg_indices):
        span = spans.get(pos)
        if not span:
            continue
        windows.append((pos, span[0] / sample_rate, span[1] / sample_rate))

    mapping = map_speakers(collect_speech_lines(plan), asr_segments)

    heard: dict[int, list[dict[str, Any]]] = {pos: [] for pos, _s, _e in windows}
    for seg in asr_segments:
        try:
            start = float(seg.get("start") or 0.0)
            end = float(seg.get("end") or start)
        except (TypeError, ValueError):
            continue
        best: int | None = None
        best_overlap = 0.0
        for pos, w_start, w_end in windows:
            overlap = min(end, w_end) - max(start, w_start)
            if overlap > best_overlap:
                best, best_overlap = pos, overlap
        if best is None:
            continue
        key = _speaker_key(seg.get("speaker"))
        heard[best].append({
            "speaker": key,
            "mapped": int(mapping.get(key, -1)),
            "text": str(seg.get("text") or ""),
            "start": round(start, 2),
            "end": round(end, 2),
        })

    out: list[dict[str, Any]] = []
    for pos, idx in enumerate(seg_indices):
        span = spans.get(pos)
        out.append({
            "index": int(idx),
            "start": round(span[0] / sample_rate, 2) if span else None,
            "end": round(span[1] / sample_rate, 2) if span else None,
            "prompt": [
                {"speaker": int(line.speaker), "text": line.text}
                for line in prompt_lines.get(int(idx), [])
            ],
            "heard": heard.get(pos, []),
        })
    return out


def run_asr_check_detailed(
    plan: Any,
    audios: list[Any],
    model_handle: Any,
    *,
    seg_indices: list[int] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Report text **and** per-segment attribution. Never raises.

    ``seg_indices`` are the timeline positions of ``audios``, one per entry, in the
    same order — without them the recognised speech cannot be assigned back to the
    segment that produced it.
    """
    if model_handle is None:
        return "音频有效性校验 (ASR): 未接线 ASR 模型，已跳过。", []
    warnings: list[str] = []
    spans: dict[int, tuple[int, int]] = {}
    try:
        audio = concat_audio(audios, warnings=warnings, spans=spans)
    except Exception as exc:  # pragma: no cover - defensive
        return f"音频有效性校验 (ASR): 音频拼接失败 — {type(exc).__name__}: {exc}", []
    if audio is None:
        return "音频有效性校验 (ASR): 没有可识别的音轨（全部静音），已跳过。", []

    try:
        payload, mode = transcribe(model_handle, audio)
    except Exception as exc:
        log.warning("音频有效性校验: 识别失败 (%s)", exc, exc_info=True)
        return f"音频有效性校验 (ASR): 识别失败 — {type(exc).__name__}: {exc}", []

    try:
        report = build_asr_report(plan, audio, payload, mode=mode, warnings=warnings)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("音频有效性校验: 判定失败 (%s)", exc, exc_info=True)
        return f"音频有效性校验 (ASR): 判定失败 — {type(exc).__name__}: {exc}", []

    try:
        details = segment_asr_details(
            plan,
            seg_indices,
            spans,
            int(audio.get("sample_rate") or 0),
            _asr_segments(payload),
        )
    except Exception as exc:  # pragma: no cover - defensive
        # The verdict is still valid without the per-segment split; losing the
        # whole check over a presentation detail would be worse.
        log.warning("音频有效性校验: 分段时间归属失败 (%s)", exc, exc_info=True)
        details = []
    return report, details


def run_asr_check(
    plan: Any,
    audios: list[Any],
    model_handle: Any,
) -> str:
    """Report text only (see :func:`run_asr_check_detailed`)."""
    return run_asr_check_detailed(plan, audios, model_handle)[0]


__all__ = [
    "ASR_MODEL_TYPE",
    "LONG_AUDIO_SECONDS",
    "PASS_RATE",
    "SpeechLine",
    "UNKNOWN_SPEAKER",
    "audio_seconds",
    "auto_model_available",
    "build_asr_report",
    "collect_speech_lines",
    "concat_audio",
    "default_model_handle",
    "error_rate",
    "map_speakers",
    "parse_speech_lines",
    "run_asr_check",
    "run_asr_check_detailed",
    "segment_asr_details",
    "tokenize",
    "transcribe",
]
