# 保留音频：待确认项 —— 音频 latent 形状是否依赖画布

> 状态：**未确认**。本文记录「保留音频」功能里唯一一个没有从上游源码核实的假设，
> 以及如果假设不成立需要改哪些地方。
>
> 功能本身（提取音频 / 音频 tab / 保留音频）已实现并可用；这里的结论只影响
> 跨画布复用时的边界情况，不影响默认路径。

---

## 1. 背景

「保留音频」在采样前把用户选中的 PCM 用音频 VAE 编码，写进 AV latent 的**音频流**，
并把该流的 `noise_mask` 置 0（锁定，采样不重绘）。出片时不解码音频，直接复用原 PCM。

核心实现：`director/audio_retain.py`
- `apply_retain_audio(latent, audio_vae, pcm, *, sample_len=None, fps=24.0, seconds=None)`
- `_fit_waveform()`
- `_latent_audio_seconds()`

接入点：
| 阶段 | 文件:位置 | 说明 |
|---|---|---|
| Phase 1 | `director/batch_executor.py` `build_retain_audio_cache()` | 读 PCM 进内存 |
| Phase 2（一采） | `director/batch_phases.py` `_sample_one_segment`，`sample_single_stage` 之前 | 写音频流 + 置 mask |
| Phase 3（一采） | `director/batch_phases.py` `_decode_export_one_segment` | `decode_audio=False` + 复用 PCM |
| 二采 | `director/second_sampling.py` `_sample_one` / `_decode_one` | 同上 |

---

## 2. 待确认的假设

**假设：AV latent 的音频流形状 `[1, C, 2, T]` 中的 `T` 只由帧数决定，与 width / height 无关。**

即 `T ≈ round(帧数 / fps × AUDIO_HZ)`，`AUDIO_HZ = 40.0`（`h3_motion_context.py:29`）。

### 支撑证据（都是本仓库内部的，不是上游源码）

1. `director/h3_motion_context.py:130-131`
   ```python
   FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
   def pixel_frames_for_latent_t(latent_t: int) -> int:
       return sum(FRAME_PER_TOKEN[k % 5] for k in range(int(latent_t)))
   ```
   视频 latent 的**时间步**只由帧数决定；H/W 只影响空间维。

2. `director/h3_motion_context.py:323-372` `_audio_tail_from_latent` 里的运行时一致性断言：
   ```python
   total_t = int(audio.shape[-1])
   frames = pixel_frames_for_latent_t(int(video.shape[2]))
   overhang = total_t - FRAME_RESCALE * frames      # FRAME_RESCALE = 5/3 = 40/24
   if not (0.0 <= overhang < 1.0):
       log.warning("Director continuity: unexpected audio grid ...")
   ```
   这条在段间延续时持续在跑。如果 `T` 还受 H/W 影响，它应该会频繁告警。

### 为什么不能算已确认

- `_empty_av_latent(width, height, length)` 的实现在 ComfyUI 核心
  （`comfy_extras/nodes_minimax_h3.py`），**本项目不含该文件，本机也没有 ComfyUI 安装**，
  上游源码未能拉取核实。
- 上面两条都是"本仓库对上游的假设"，不是上游实现本身。

---

## 3. 如果假设不成立，需要改什么

**结论先行：`apply_retain_audio` 大概率不需要改。** 它不是"先算出 T 再往里写"，而是：
1. 读目标 latent 自己的音频流长度 `audio.shape[-1]`；
2. 需要多长就从 PCM 编多长（由这个 T 反推 seconds）；
3. 写回只用 `t = min(T_latent, T_encoded)`，不足补零。

所以 T 由帧数还是由画布决定，编码长度都跟着目标 latent 走，自洽。

真正需要检查的是另外两处：

### 3.1 通道布局（`apply_retain_audio`）

当前假设音频流是 `[B, C, 2, T]`（2 = 立体声）。已有两种兼容：
- 编码结果形状完全相等 → 直接整体替换
- 编码结果是 `[B, 2C, T]` → reshape 成 `[B, C, 2, T]`
- 都不是 → 抛异常，调用方降级为正常生成（不中断）

若实际布局是第三种，需要在这里补分支。

### 3.2 `seconds` 的反推（`_latent_audio_seconds`）

```python
seconds = audio_T / AUDIO_HZ      # AUDIO_HZ = 40.0
```
如果音频网格不是 40 Hz（或不是线性于时间），这里算出的 PCM 长度会偏，
表现为成片音频时长对不上画面（短 → 尾部静音；长 → 被 `align_pcm_to_frames` 截掉）。

更稳的替代：由视频流反推帧数再除 fps——
```python
from .h3_motion_context import pixel_frames_for_latent_t
frames = pixel_frames_for_latent_t(int(video.shape[2]))
seconds = frames / float(fps)
```
这条路完全不依赖 `AUDIO_HZ`，只依赖本仓库已经在用的 `pixel_frames_for_latent_t`。
**如果确认假设不成立，优先换成这个写法。**

### 3.3 二采的跨画布假设

`second_sampling.py` 里我写了"二采换了画布，音频流没变，所以同一条音频可直接复用"。
若音频流实际受画布影响，二采的 `new_av` 音频流 T 会和一采不同——不过因为
`apply_retain_audio` 是按目标 latent 现算现编的，结果仍然正确，只是这句注释要改掉。

---

## 4. 怎么一次性确认

### 4.1 最直接

把 `comfy_extras/nodes_minimax_h3.py` 里 `_empty_av_latent` 的实现贴出来对照即可。

### 4.2 加诊断日志（换两个画布各跑一段）

在 `director/batch_phases.py` `_sample_one_segment`、latent 重建之后加：

```python
_v, _a = latent["samples"].unbind()
log.info("latent shape diag: ctx=%dx%d len=%d video=%s audio=%s",
         ctx_w, ctx_h, sample_len, tuple(_v.shape), tuple(_a.shape))
```

判定方法：
- 固定 `sample_len`，只改画布 → `audio` 形状**不变** ⇒ 假设成立
- `audio` 形状随画布变 ⇒ 假设不成立，按 §3.2 换反推方式

---

## 5. 顺带澄清：`shift video` / `shift audio` 不是画布参数

容易被误当成画布相关，实际是 **sigma 调度偏移**，按视频/音频两条流分别设置。

```python
# director/core_sampling.py:99-103
if apply_shift:
    shifted = MiniMaxH3SigmaShift.execute(model, float(shift_video), float(shift_audio))
    model_use = _unpack_node_output(shifted)[0]
```

- 定义：`nodes/director.py:220-227`，默认 `shift_video=12.0`、`shift_audio=3.0`
- 语义：把噪声调度按 `σ' = shift·σ / (1 + (shift−1)·σ)` 重新分布。
  值大 → 更多步落在高噪声区（先定构图/运动）；值小 → 更多步落在低噪声区（细节）
- 为什么两个：H3 的 DiT 是双流（视频 + 音频），但 `KSampler` 只有一条 sigmas，
  官方于是按流分别 patch。视频流 token 多、需要更强的结构偏置，所以 12；音频流维度低，3 就够
- 返回的是**打过补丁的 MODEL**，不是 per-call 参数；先 shift 再 sample
- 二采透传同一对值：`nodes/director.py:503-504`

---

## 6. 相关文件速查

| 文件 | 作用 |
|---|---|
| `director/audio_retain.py` | 保留音频全部逻辑（解析开关 / 读 PCM / 注入 latent + mask） |
| `director/audio_extract.py` | 提取音频的持久缓存（音频 tab 的数据源） |
| `director/batch_phases.py` | 一采采样注入 + 出片复用 PCM |
| `director/second_sampling.py` | 二采采样注入 + 出片复用 PCM |
| `director/batch_executor.py` | 构造 retain 缓存并传给 Phase 2 / 3 |
| `director/h3_motion_context.py:28-31` | `FPS` / `AUDIO_HZ` / `FRAME_RESCALE` / `FRAME_PER_TOKEN` |
| `director/h3_motion_context.py:130-131` | `pixel_frames_for_latent_t()` |
| `director/h3_motion_context.py:323-372` | `_audio_tail_from_latent()`（音频网格一致性断言） |
| `director/h3_latent_continue.py:264-286` | AV 双流 `noise_mask` 的先例 |
| `director/core_sampling.py:69-188` | `sample_single_stage()`，`noise_mask` 唯一入口 |

开关字段：`timeline.segments[i].retainAudioId`（存 audio_extract 的 entry id，空 = 不保留）
