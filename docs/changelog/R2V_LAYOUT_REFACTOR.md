# R2V 模式布局改造方案与进度

> 目标节点：`ComfyUI\custom_nodes\ComfyUI_MiniMaxH3_Director`
> 主要文件：`web/js/minimax_image_batch.js`、`web/js/minimax_timeline.js`、`web/js/minimax_i18n.js`
> 后端：`director/plan.py`、`director/gen_timeline.py`、`director/segment_cache.py`、`director/http_routes.py`

---

## 0. 已确认的需求口径

| # | 需求 | 确认结论 |
|---|---|---|
| 1 | 公共参数模块 | r2v 模式下整体移除（`timeline.js` 的 `.bd-split` 仅对非 r2v 模式保留） |
| 2 | 全显模式 | 移除，恒为单显；`batchDetailMode` 相关 UI/状态全部删除 |
| 3 | 左侧三模块 | 参考图片(3 列×3 行＝9/页)、参考视频(3 列×2 行＝6/页)、参考音频(3 列×2 行＝6/页) |
| 4 | 模块头 | 最右＝展开/收起；展开态显示 `公共` / `片段`（互斥）、`选已有` |
| 5 | 素材组页码 | **不省略的方框+数字**，位于原「素材组 N」处；该组引用溢出时页码框置红 |
| 6 | 素材模块页码 | 可复用组件，**带省略号**（`‹ 1 … 4 5 6 … 12 ›`） |
| 7 | 上传数量 | 组内/公共**均不限量**（翻页展示）；实际生效仍受 MiniMax 官方上限 9/3/3 |
| 8 | 拖动排序 | 取消（图片/视频/音频全部取消） |
| 9 | 素材 tile | 空→点击上传；有素材→上半 hover「预览」(点击弹窗)、下半 hover「复制 id」(`<Picture 1>`) |
| 10 | 公共素材页 | 与素材组页**结构完全一致**，只是没有专有素材；公共提示词**固定显示在公共页** |
| 11 | 提示词引用量 | 标题右侧 `图片 1/9 视频 1/3 音频 1/3`；分子＝组内提示词引用 ∪ 公共提示词引用（去重） |
| 12 | 溢出表现 | 提示词框红边 + 该项指标红 + 整素材组卡片红边 + 组标题上方红字 |
| 13 | 预览模块 | 加 `一采`/`二采` tab；有视频缓存时二采直连缓存 |
| 14 | 旧工作流兼容 | **不考虑**，不做 schema 版本分支 |

---

## 1. 关键设计决策

### 1.1 编号方案：沿用统一绝对编号（不重编号）

- 公共素材：`global.refs` / `refVideos` / `refAudios`，`index` 从 `0` 递增
- 组内素材：`seg.refs` / `refVideos` / `refAudios`，`index` 从 `offset = 公共已上传数` 开始
- UI 标签与 token 均为绝对编号：`图片 N` = `index + 1`，token `<Picture N>`
- `rebaseR2vGroupSlotsForCommon()` **保留**（公共增删后重排组内编号）

**为什么可以"不限量"**：后端 `plan.py:370/411/467` 对 `index >= MAX` 直接 `continue`，
超出素材仅存在于 UI/存档层，不会进图。前端红色溢出提示即告知用户"多出的不生效"。

### 1.2 引用量计算口径

```js
usage.picture = new Set([
  ...matchTags(global.prompt, "Picture"),   // 公共提示词
  ...matchTags(seg.prompt,    "Picture"),   // 组内提示词（已含绝对编号）
]).size
// 分母：9 / 3 / 3
```

### 1.3 页码组件（两个，职责不同）

| 组件 | 位置 | 形态 | 状态 |
|---|---|---|---|
| `createGroupPaginator` | 卡片头（原「素材组 N」） | `[公共素材] [1][2][3]...` 方框+数字，不省略 | 溢出→该框红 |
| `createMiniPager` | 每个素材模块底部 | `‹ 1 … 4 [5] 6 … 12 ›` 带省略 | 可复用，纯 UI |

### 1.4 页面切换状态

```js
editor.r2vPage = "common" | <groupIndex:number>   // 默认 0
editor.r2vScope = { image:"common"|"segment", video:..., audio:... }  // 各模块独立
editor.r2vFold  = { image:bool, video:bool, audio:bool }              // 展开/收起
editor.r2vAssetPage = { image:0, video:0, audio:0 }                   // 各模块页码
```

---

## 2. 实施清单（状态追踪）

图例：`[ ]` 待做 · `[~]` 进行中 · `[x]` 已完成 · `[-]` 已取消

### P1 — 删除公共参数区 / 全显模式 / 参考图尺寸
- [x] P1.1 `timeline.js`：r2v 下隐藏 `.bd-split`（保存 `this.splitEl`，`updateModeUI` 里 toggle）
- [x] P1.2 `timeline.js:6195`：`isR2vCommonEnabled()` 在 r2v 下恒返回 `true`（公共素材页常驻）
- [x] P1.3 `timeline.js`：删除 `r2v-common-toggle` / `r2v-common-fold` 相关逻辑与 CSS
- [x] P1.4 `image_batch.js`：删除全显模式按钮（HTML/绑定/`toggleBatchDetailMode`/`isBatchDetailSolo`）
- [x] P1.5 `image_batch.js:2468-2495`：删除卡片头「参考图尺寸」select
- [x] P1.6 `timeline.js:2645-2651`：删除片段头「参考图尺寸」select
- [x] P1.7 清理 `.bd-batch-solo` / `.bd-split` 相关 CSS

### P2 — 素材组页码组件（替换「素材组 N」）
- [x] P2.1 新增 `createGroupPaginator()`：公共素材按钮 + 不省略方框数字
- [x] P2.2 卡片头接入 `editor.r2vPage`，替换 `batch.groupTitle.asset`
- [x] P2.3 `renderImageBatchGroups` 按 `r2vPage` 渲染单页（公共页用 global 代理 `r2vCommonCardSeg`）
- [x] P2.4 溢出时对应页码框置红

### P3 — 左侧三模块统一形态
- [x] P3.1 新增可复用 `createMiniPager()`（带省略号）+ 页码 CSS
- [x] P3.2 重写 `createR2vSection()`：展开/收起(最右) + 公共/片段(互斥) + 选已有
- [x] P3.3 重写 `appendR2vMediaSections()`：三模块统一、3×3 / 3×2 / 3×2、无限素材+翻页
- [x] P3.4 删除「公共继承只读区」
- [x] P3.5 删除 `R2V_PICTURE_STEP` 渐进展开逻辑（改由页码控制）
- [x] P3.6 素材槽渲染改为「实际素材 + 1 空位」，不再固定 slots 循环

### P4 — 素材 tile 交互改造
- [x] P4.1 删除 r2v 拖动排序（不再调用 `bindBatchRefDrop`；`moveBatchRefSlot` 置空实现）
- [x] P4.2 tile 上下 hover 热区（预览 / 复制 id）
- [x] P4.3 `copyAssetTag()` 复制到剪贴板（`<Picture 1>` / `<Video 1>` / `<Audio 1>`）
- [x] P4.4 `openAssetPreviewModal()` 预览弹窗（图/视频/音频）

### P5 — 提示词引用量与溢出告警
- [x] P5.1 新增 `computePromptRefUsage(editor, seg)` / `computeCommonRefUsage(editor)`
- [x] P5.2 提示词标题右侧渲染 `图片 1/9 视频 1/3 音频 1/3`
- [x] P5.3 输入/素材变更后实时刷新（重渲染即重算）
- [x] P5.4 溢出：提示词框红边 + 单项红 + 整卡红边 + 组标题上方红字 + 页码框红
- [x] P5.5 公共提示词编辑器（公共页）同步接入引用量

### P6 — 预览模块「一采/二采」tab
- [x] P6.1 tab 条 UI（仅 r2v 视频任务显示）
- [x] P6.2 二采可用性探测（HEAD `/minimax/director/segment_clip`，30s 缓存）
- [x] P6.3 新增 `GET /minimax/director/segment_clip` 只读路由，直连缓存
- [x] P6.4 无缓存时 tab 置灰 + 提示

### P7 — 选已有：默认列表 + 批量选择
- [x] P7.1 `showInputMediaPicker` 增加 `multi` 选项（checkbox + 返回数组 + 全选/清空）
- [x] P7.2 新增 `chooseImageInputs/chooseVideoInputs/chooseAudioInputs`
- [x] P7.3 `pickExistingR2vAssets` + `appendR2vPicked` 批量写入，去掉 `slotsFull` 硬拦截
- [x] P7.4 模块头「选已有」按钮接入批量

### P8 — i18n 与收尾
- [x] P8.1 中英双语新增词条
- [x] P8.2 语法检查（node --check 三个 JS + python AST 后端）
- [ ] P8.3 浏览器冒烟验证（需重启 ComfyUI 后人工确认）

---

## 3. 关键代码位置速查

| 功能 | 文件:行 |
|---|---|
| 卡片渲染入口 | `minimax_image_batch.js:2288 renderImageBatchGroups` |
| 卡片头（组标题/参考图尺寸） | `minimax_image_batch.js:2392-2495` |
| 左列三模块 | `minimax_image_batch.js:1651 appendR2vMediaSections` |
| 模块头 | `minimax_image_batch.js:1413 createR2vSection` |
| 图片/视频/音频槽渲染 | `image_batch.js:1940 / 1527 / 1447` |
| 提示词 + 预览挂载 | `minimax_image_batch.js:2618-2670` |
| 拖动排序 | `image_batch.js:1111 moveBatchRefSlot`、`1131 bindBatchRefDrop` |
| 单显/全显 | `image_batch.js:2175/2189`、`timeline.js:1544/1646` |
| 公共参数面板 HTML | `minimax_timeline.js:2566-2693` |
| 公共参数开关 | `timeline.js:6190-6225`、`3023-3055` |
| offset 计算 | `image_batch.js:78-100 r2vCommonPicOffset/…` |
| 重排 | `image_batch.js:174 rebaseR2vGroupSlotsForCommon` |
| 素材选择器（单选） | `timeline.js:7825 showInputMediaPicker` |
| token 正则 | `minimax_prompt_mentions.js:19 TAG_RE` |
| token 生成 | `minimax_gen_timeline.js:232/244/255` |
| 后端上限截断 | `plan.py:370/411/467` |
| 后端合并 | `plan.py:190/203`、`gen_timeline.py:392-470` |
| 二采状态 | `timeline.js:4091/4117`、`http_routes.py:662` |
| 分段缓存 clip | `segment_cache.py:2329 run_segment_export`、`2655 clip_path` |

---

## 4. 验收清单

- [ ] r2v 模式下不再出现「公共参数」区块
- [ ] 工具栏不再出现「单显模式/全显模式」按钮
- [ ] 卡片头为 `[公共素材] [1][2][3]` 方框页码，无省略
- [ ] 三模块头：最右折叠按钮；展开态有 公共/片段/选已有
- [ ] 公共页与素材组页结构一致，公共页无「片段」专有素材
- [ ] 图片 3×3、视频 3×2、音频 3×2；上传不限量并可翻页
- [ ] 素材模块页码带省略号
- [ ] 素材不可拖动
- [ ] 空 tile 点击上传；有素材时上下 hover 分别为预览/复制 id
- [ ] 复制得到 `<Picture 1>` 形式，可直接粘贴进提示词
- [ ] 提示词标题右侧显示 `图片 x/9 视频 x/3 音频 x/3`
- [ ] 溢出时：提示词框红、指标红、整卡红边、标题上方红字
- [ ] 预览区有 一采/二采 tab，二采有缓存时可播放
- [ ] 选已有默认进入已有列表，支持多选批量写入
