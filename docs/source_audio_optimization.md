# 源音频链路（模式 / 抽取 / 保留 / 排错）

**相关代码**：`director/audio_export.py`（模式与导出）、`director/audio_extract.py`（提取音频存储）、
`director/audio_retain.py`（保留音频）、`lib/audio_io.py`（抽取与诊断）。

## 1. 三种声音模式

写在 `timeline.output.audioMode`，前端对应「声音」下拉：

| 模式 | 别名 | 行为 |
|---|---|---|
| `generate`（默认） | `generated`、`model` | 用模型生成的音轨 |
| `source` | `original`、`passthrough` | 用源视频的原声；**仅 `v2v` / `rv2v` 生效**，其他任务自动回退 `generate` |
| `mute` | `silent`、`silence` | 静音，输出 44.1 kHz 空音轨 |

`generate` 模式下如果模型音轨为空，而任务属于 `{i2v, fl2v, r2v, v2v, rv2v}`，会自动退回到
源音频（不会静音）。报告里会说明本次实际用的是哪种。

「全部导出」合并画面时，各组音频会按时间轴拼接；否则只会用第一组音频、后半段静音。

## 2. 原声是怎么抽出来的

`lib/audio_io.py::extract_timeline_audio` 把**逻辑时间轴区间**映射回源文件的真实时间轴：

1. 由 `frameMap` 把 `[logical_start, logical_end)` 换算成若干段 `(文件路径, 起秒, 时长, 原生 fps)`；
2. 用 ffmpeg 解码出 PCM（44.1 kHz 起，实际以源文件采样率为准）；
3. 按目标时长裁剪 / 补零，使音轨长度与画面帧数严格对齐（避免音画漂移）。

无 ffmpeg、源文件没有音轨、或 ffmpeg 解码失败时返回 `None`，并由
`diagnose_source_audio_failure()` 给出人话原因（见第 5 节）。

## 3. 提取音频（独立于渲染缓存）

`director/audio_extract.py`：把一段源 PCM 存成可复用的条目。

- 目录：`<节点缓存目录>/audio_extract/`；
- 命名：按 **entry id**（`audio_<id>.pt` / `audio_<id>_latent.pt`），**不是内容哈希、不是时间轴位置**；
- 因此：改提示词、重跑、甚至改画布都不会让它失效；「清空缓存」「清空节点所有缓存」和槽位表的
  孤儿回收都碰不到它（前缀 `audio_` 既不匹配 `seg_*` / `seg2_*`，也不是 `cache_layout.stem_of_filename`
  认可的后缀）。
- 绑定方式：条目记的是时间轴卡片上的 **segment id**，位置是推导出来的 ——
  重排 / 中间插入时按当前 id 顺序重算索引（音频跟着卡片走），删除卡片时条目与文件一起删。
  旧时间轴若卡片没有 id，则跳过对账而不是猜（猜错会删掉用户还要用的音频）。
- **每段每来源只有一条**（`(seg_id, variant)` 唯一）：重新提取同一个片段会**覆盖**上一条 ——
  沿用原 **entry id**（文件名相同，字节就地替换），所以「音频」tab 不会堆重复条目，
  已经在时间轴上钉住的「保留音频」也不会指向一个被覆盖掉的旧文件。旧 latent 若本次没切出来，
  会顺手删掉，不留孤儿文件。
  这条规则**对磁盘上已有的老条目同样生效**：`sync_audio_slots()`（每次打开「音频」tab、删除卡片、
  提取音频时都会跑）会先调 `collapse_duplicate_takes()` 把同一 `(seg_id, variant)` 的多条收敛成
  一条 —— 优先保留被「保留音频」钉住的那条（前端在载荷里带 `retainIds`，`_retain_ids()` 读它），
  否则保留 `created` 最新的那条，并把被丢弃条目的 WAV/latent 一起删掉（只从索引里剔除会留下
  永远不再被引用的文件）。重提取时同样会清掉同键的旧条目文件。
  注意粒度是 **每段 × 每来源一条**：如果你对同一段分别从一采、二采各提取一次，会看到 **2 条**
  （一条 `variant=1st`、一条 `variant=2nd`）。而「保留音频」是**每段一个**（`retainAudioId` 单值）。
- **清空范围**：`audio_extract/` 在自己的子目录、文件名是 `audio_*`，`clear_cache` 的
  `SEGMENT_GLOBS`（`seg_*` / `seg2_*`）扫不到它 —— 所以「清空缓存」**不动**提取音频（原设计），
  而「清空节点所有缓存」（`clear_all=true`）会额外调 `clear_audio_store()` 把条目文件与
  `index.json` 一起删掉。删完前端会调 `clearRetainedAudioPins()` 解掉时间轴上的
  `retainAudioId`，否则运行时会去找一个已经不存在的条目。
- **提取不出来的原因**：`_extract_one` 失败时 `run_audio_extract` 用 `skip_code()` 回一个**码**
  （`no-cache` / `latent-only` / `no-audio-track` / `unreadable` / `not-in-plan` / `error`），
  前端在结果区逐段翻成人话。码由 `audio_extract_availability` 的 `hasWave / hasClipAudio /
  hasClip / hasAvLatent` 判定，所以「显示可提取」和「实际跳过」用的是同一份依据。

## 4. 保留音频（retainAudioId）

在**卡片头部**（「引用上段」旁边）打开「保留音频」开关即可，该段运行时：

- **Phase 2**：把这段 PCM 用 audio VAE 编码进 AV latent 的**音频流**，并把该流的 `noise_mask` 置零。
  UNet 看得到它、以它为条件生成画面，但永远不会重绘它。
- **Phase 3**：跳过音频 VAE 解码，直接把**同一份 PCM** 混流回去 —— 听到的就是你选的那一段，
  不经过音频 VAE 往返。

为什么不直接复用存下来的 audio latent：那份 latent 的 `T` 是按「上次导出长度」算的，而本次需要
`T = round(sample_len / fps * AUDIO_HZ)`；开启「段间引导」时 `sample_len` 会比导出长度多出被 pin 的
头部帧，两者不一致。从 PCM 重新编码只是一次廉价的 `audio_vae.encode`，永远落在当前网格上。

音频流与画布无关（形状 `[1, C, 2, T]`，T 只由帧数决定），所以保留的音频在卡片换到别的画布后仍然
有效 —— 这一点视频 pin 做不到（换画布会降级到像素路径）。

**头部偏移**：开启段间引导时，一段是「采样得比导出长」，头部 pin 的前缀解码后被裁掉。保留音频因此
要从「导出起点」而不是 t=0 开始写（调用方传 `head_seconds`），否则混进去的声轨会比 UNet 实际听到的
早 `trim_frames / fps` 秒；头部区域保留连续段 pin 的内容。

## 5. 排错

`diagnose_source_audio_failure()` 会返回具体原因，报告里原样显示：

| 报告文字 | 处理 |
|---|---|
| `ffmpeg unavailable (install FFmpeg on PATH or pip install imageio-ffmpeg)` | 装 ffmpeg；ComfyUI 便携版一般自带，`imageio-ffmpeg` 是兜底 |
| `could not map timeline frames to a source video path` | 时间轴的 `frameMap` / 源视频路径不对，重新上传或重建时间轴 |
| `input video has no audio track` | 源视频本身没有音轨，改用 `generate` |
| `audio extraction failed – ffprobe not found …` | 装带 `ffprobe` 的完整 FFmpeg |
| `ffmpeg failed to extract audio despite an audio stream being present` | 源音轨编码 ffmpeg 解不了，先转码成 AAC / PCM 再上传 |

其他常见问题：

- **原声听着有延迟 / 提前**：确认输出帧率与源视频原生 fps 是否一致；上传视频时前端会默认跟随源 fps，
  手动改帧率会保持真实时长并重算帧数。
- **后半段没声音**：检查导出方式是不是「全部导出」（分段导出时每段各自一条音轨）；
  合并路径下各组音频是按时间轴拼接的，若仍无声，看报告里 `Audio:` 那一段写的是哪种来源。
- **换源视频后原声不对**：源视频身份（相对路径 + 大小 + mtime）参与缓存指纹，换源会让相关片段重算。
