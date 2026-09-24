# Source 模式音频优化改造说明

## 概述

**目标模式**: `audio_mode = "source"`（声音=使用原声）

source 模式下，成片音轨来自**源视频**，而不是 AV latent。优化只做一件事：把源 PCM 在
Phase 1 提前提取一次，Phase 3 与分段 mp4 封装直接复用内存中的张量，避免每段重复调用
`extract_timeline_audio`（内部要跑 ffmpeg）。

> 上一版实现曾声称"编码 audio latent 用于采样锁定保护"。该部分并未接入采样流程，
> 属于死代码，已删除。**本次改造不涉及任何 audio latent 锁定**，采样路径与改造前一致。

---

## 改造前后对比

### 改造前

```
Source 模式:

Phase 1  文本编码 / Video VAE / 参考音频 VAE 编码（ref_audios 等，采样必需）
Phase 2  UNet 采样
Phase 3  Video VAE 解码，audio_dict = 空
         └─ save_segment_cache → clip.mp4 封装 → extract_timeline_audio  ← 第 1 次提取
         └─ maybe_export_segment_mp4        → extract_timeline_audio  ← 第 2 次提取
```

Phase 3 本身并不提取音频，重复提取发生在每个 mp4 封装点，且每段都会发生。

### 改造后

```
Source 模式:

Phase 1  文本编码 / Video VAE / 参考音频 VAE 编码（不变）
         └─ 提取各段源 PCM → source_audio_cache（唯一一次提取）
Phase 2  UNet 采样（不变）
Phase 3  Video VAE 解码，audio_dict = 缓存 PCM（对齐到导出帧数）
         └─ save_segment_cache / maybe_export_segment_mp4 直接用该 PCM，不再读文件
```

---

## 具体改动

### 新增 `director/batch_source_audio.py`

| 函数 | 作用 |
|------|------|
| `build_source_audio_cache(run_list, plan, fps)` | Phase 1 批量为所有运行段提取源 PCM |
| `extract_segment_source_pcm(seg, plan, fps)` | 单段提取（`plan.raw` 时间线 → PCM） |
| `align_pcm_to_frames(pcm, frame_count, fps)` | 把 PCM 裁/补到指定视频帧数 |

要点：

- 时间线取自 `plan.raw`（解析后的 dict），与 `audio_export` 的取法完全一致；
  不是 JSON 字符串。
- 直接接收 `seg` 对象，不再用 `plan.segments[seg.index]` 反查。
- 只缓存 PCM，不编码 latent。

### `director/batch_executor.py`

- 仅 `AUDIO_MODE_SOURCE` 构建 `source_audio_cache`（mute 无音轨、generate 用模型音频，
  都不需要，避免 mute 模式白跑一次提取）。
- 保留 `encode_audio_vae_batch`：它编码的是**参考音频**（`seg.ref_audios`、
  参考视频音轨），属于采样输入 conditioning，任何模式都必需。source 模式只是不解码
  *输出* 音频，不能跳过它。

### `director/batch_phases.py`

Phase 3 `_decode_export_one_segment`：

- 命中缓存时跳过 audio VAE 解码，直接用缓存 PCM。
- 音频对齐统一在本处完成：先按导出口径裁视频，再把 PCM 对齐到 `decoded.shape[0]` 帧。
- **只裁尾、不裁头**：源 PCM 锚定在 `seg.start_frame`，而连续性头部 pin 位于该点**之前**
  的时间线上，不属于本段的源音频；模型生成音频才需要随 pin 一起丢掉头部。

### `director/audio_export.py`

`prepare_segment_audio_for_file_export` 在 source 模式下优先复用传入的 PCM，
缺失时仍回退到原提取逻辑（generate 模式的空模型音频回退路径不受影响）。

### `director/h3_motion_context.py`

source 模式的段缓存现在带有真实源音频，连续性 pin 因此可能走到"编码上下文音频"
分支。该分支改为**失败即降级**（只 pin 视频），不再因重采样/编码失败中断整个运行。

---

## 未改变的行为

- **generate 模式**：全流程不变。
- **mute 模式**：不提取、不缓存，输出静音。
- **参考音频编码**：不变（改造前一度被跳过，属缺陷，已恢复）。
- **最终合并 AUDIO 输出**：source 模式仍由 `build_director_audio_outputs` 按整条时间线
  提取，不使用分段缓存。
- **采样流程**：除上述降级保护外，与改造前一致；没有引入 audio latent 锁定。

---

## 验证要点

- `ruff check .`（配置见 `pyproject.toml`，规则 `F821/F822/F823`）——用于拦截
  "未导入/未定义名称"这一类会让整个 phase 崩溃或在 `try/except` 中静默失效的错误。
- source 模式跑多段（含「选择运行」部分运行）后，确认：
  - 报告里 `Phase 1` 出现 `Source audio: N/M segment(s) cached for reuse`；
  - 每段 mp4 与合并成片音轨均正常，且与改造前一致。
