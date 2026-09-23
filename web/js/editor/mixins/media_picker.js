/** media_picker mixin for the Director editor (media_picker).
 *
 * Extracted verbatim from editor.js; methods are unchanged apart from the
 * trailing comma an object literal needs.
 */

import { uploadToInput, uploadToInputSmart } from "../../core/upload.js";
import { relPath } from "../../core/utils.js";
import { inputViewUrl, videoRelativePath } from "../urls.js";
import { t } from "../../minimax_i18n.js";
import { extractReferenceAudioFromExistingVideo, prepareLocalReferenceAudio } from "../../minimax_ref_audio.js";
export const media_pickerMixin = {
    showInputMediaPicker({ kind, title, accept, currentValue = "", multi = false, includeCache = false } = {}) {
        return new Promise((resolve) => {
            this._closeBdModal();

            const overlay = document.createElement("div");
            overlay.className = "bd-modal-overlay";
            const panel = document.createElement("div");
            panel.className = "bd-modal bd-media-modal";
            panel.innerHTML = `
                <div class="bd-media-head">
                    <div class="bd-modal-title"></div>
                    <div class="bd-modal-actions bd-media-head-actions"></div>
                </div>
                <div class="bd-media-status"></div>
                <div class="bd-media-body">
                    <div class="bd-media-left">
                        <div class="bd-media-table" tabindex="0">
                            <div class="bd-media-thead">
                                <button type="button" class="bd-media-th" data-sort="name">
                                    <span></span><i class="bd-media-sort"></i>
                                </button>
                                <button type="button" class="bd-media-th" data-sort="dims">
                                    <span></span><i class="bd-media-sort"></i>
                                </button>
                                <button type="button" class="bd-media-th" data-sort="time">
                                    <span></span><i class="bd-media-sort"></i>
                                </button>
                            </div>
                            <div class="bd-media-tbody"></div>
                        </div>
                    </div>
                    <div class="bd-media-right">
                        <div class="bd-media-preview">
                            <div class="bd-media-preview-empty"></div>
                        </div>
                        <div class="bd-media-meta"></div>
                    </div>
                </div>`;

            panel.querySelector(".bd-modal-title").textContent = title || "";
            const statusEl = panel.querySelector(".bd-media-status");
            const actionsTop = panel.querySelector(".bd-media-head-actions");
            const tableEl = panel.querySelector(".bd-media-table");
            const tbodyEl = panel.querySelector(".bd-media-tbody");
            const previewEl = panel.querySelector(".bd-media-preview");
            const previewEmptyEl = panel.querySelector(".bd-media-preview-empty");
            const metaEl = panel.querySelector(".bd-media-meta");
            const showDims = kind !== "audio" && kind !== "reference_audio";
            if (!showDims) {
                tableEl.classList.add("bd-media-nodims");
                tableEl.querySelector('.bd-media-th[data-sort="dims"]')?.remove();
            }
            if (multi) {
                // 批量模式：表格多一列复选框，给 thead 补一个占位 cell 保持列对齐，
                // 否则复选框会把 name 列挤掉，time 列还会换到下一行。
                tableEl.classList.add("bd-media-multi");
                // 注意：不要用 .bd-media-th —— 它会被列头文案/排序逻辑选中，
                // 而这里没有可填充的 span，会抛 "Cannot set properties of null"。
                const cbHead = document.createElement("span");
                cbHead.className = "bd-media-th-cb";
                cbHead.appendChild(document.createElement("span"));
                cbHead.setAttribute("aria-hidden", "true");
                const theadEl = tableEl.querySelector(".bd-media-thead");
                theadEl?.insertBefore(
                    cbHead,
                    theadEl?.querySelector(".bd-media-th[data-sort='name']"),
                );
            }
            const thEls = [...panel.querySelectorAll(".bd-media-th")];
            thEls.forEach((th) => {
                const key = th.dataset.sort;
                const label = key === "dims" ? t("mediaPicker.dims")
                    : key === "time" ? t("mediaPicker.time")
                    : t("mediaPicker.file");
                // 缺 span 只跳过自己，不要让整个弹窗初始化失败。
                const labelEl = th.querySelector("span");
                if (labelEl) labelEl.textContent = label;
            });

            let selectedValue = currentValue || "";
            let itemsByPath = new Map();
            let listedItems = [];
            let sortKey = "time";
            let sortDir = "desc";
            // 批量选择：选中集合（multi=true 时生效，返回数组）
            const multiSel = new Set();

            const finish = (val) => {
                this._closeBdModal();
                resolve(val);
            };

            const choiceFor = (relPath) => {
                const item = itemsByPath.get(relPath || "");
                if (!item) return null;
                return {
                    source: "existing",
                    relPath: item.relPath,
                    fileName: item.fileName || item.name || item.relPath,
                    subfolder: item.subfolder || "",
                    type: item.type || "input",
                    mediaKind: item.mediaKind || kind,
                };
            };

            const selectedChoice = () => choiceFor(selectedValue);

            const selectedChoices = () => {
                // 保持列表顺序，用户勾选顺序无关紧要
                return sortedItems()
                    .map((it) => it.relPath)
                    .filter((p) => multiSel.has(p))
                    .map((p) => choiceFor(p))
                    .filter(Boolean);
            };

            const formatMediaTime = (unixSec) => {
                const n = Number(unixSec);
                if (!Number.isFinite(n) || n <= 0) return "—";
                const d = new Date(n * 1000);
                if (Number.isNaN(d.getTime())) return "—";
                const pad = (v) => String(v).padStart(2, "0");
                return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
            };

            const dimsText = (item) => {
                const w = Number(item?.width) || 0;
                const h = Number(item?.height) || 0;
                return w > 0 && h > 0 ? `${w}x${h}` : "—";
            };

            const dimScore = (item) => {
                const w = Number(item?.width) || 0;
                const h = Number(item?.height) || 0;
                return w > 0 && h > 0 ? w * 100000 + h : -1;
            };

            const sortedItems = () => {
                const copy = listedItems.slice();
                copy.sort((a, b) => {
                    if (sortKey === "name") {
                        const av = (a.fileName || a.name || a.relPath || "").toLowerCase();
                        const bv = (b.fileName || b.name || b.relPath || "").toLowerCase();
                        const c = av.localeCompare(bv, undefined, { numeric: true, sensitivity: "base" });
                        if (c) return sortDir === "asc" ? c : -c;
                    } else if (sortKey === "dims") {
                        const as = dimScore(a);
                        const bs = dimScore(b);
                        const aMiss = as < 0;
                        const bMiss = bs < 0;
                        if (aMiss !== bMiss) return aMiss ? 1 : -1;
                        if (as !== bs) return sortDir === "asc" ? as - bs : bs - as;
                    } else {
                        const av = Number(a.modified) || 0;
                        const bv = Number(b.modified) || 0;
                        if (av !== bv) return sortDir === "asc" ? av - bv : bv - av;
                    }
                    return (a.relPath || "").localeCompare(b.relPath || "");
                });
                return copy;
            };

            const syncHeaderState = () => {
                thEls.forEach((th) => {
                    const active = th.dataset.sort === sortKey;
                    th.classList.toggle("is-active", active);
                    th.classList.toggle("is-asc", active && sortDir === "asc");
                });
            };

            const renderPreview = (item) => {
                previewEl.innerHTML = "";
                metaEl.innerHTML = "";
                if (!item?.relPath) {
                    previewEmptyEl.textContent = t("mediaPicker.previewEmpty");
                    previewEl.appendChild(previewEmptyEl);
                    return;
                }
                const relPath = item.relPath;
                const type = item.type || "input";
                const previewKind = kind === "reference_audio" ? item.mediaKind : kind;
                if (previewKind === "image") {
                    const img = document.createElement("img");
                    img.src = inputViewUrl(relPath, type);
                    img.alt = item.fileName || item.name || relPath;
                    previewEl.appendChild(img);
                } else if (previewKind === "audio") {
                    const audio = document.createElement("audio");
                    audio.src = inputViewUrl(relPath, type);
                    audio.controls = true;
                    audio.preload = "metadata";
                    previewEl.appendChild(audio);
                } else {
                    const video = document.createElement("video");
                    video.src = inputViewUrl(relPath, type);
                    video.controls = true;
                    video.preload = "metadata";
                    video.muted = true;
                    video.playsInline = true;
                    previewEl.appendChild(video);
                }
                const fileEl = document.createElement("div");
                fileEl.textContent = `${t("mediaPicker.file")}: ${item.fileName || item.name || relPath}`;
                metaEl.appendChild(fileEl);
                const pathEl = document.createElement("div");
                pathEl.textContent = `${t("mediaPicker.path")}: ${relPath}`;
                metaEl.appendChild(pathEl);
            };

            const selectRow = (relPath, { scroll = false, toggleCheck = false } = {}) => {
                if (multi) {
                    if (!relPath) return;
                    // 行点击 = 预览；只有点复选框才切换勾选。
                    if (toggleCheck) {
                        if (multiSel.has(relPath)) multiSel.delete(relPath);
                        else multiSel.add(relPath);
                    }
                    selectedValue = relPath;
                    tbodyEl.querySelectorAll(".bd-media-tr").forEach((row) => {
                        const on = row.dataset.path === selectedValue;
                        row.classList.toggle("selected", on);
                        if (on && scroll) row.scrollIntoView({ block: "nearest" });
                    });
                    renderPreview(itemsByPath.get(relPath));
                    syncMultiRows();
                    return;
                }
                selectedValue = relPath || "";
                tbodyEl.querySelectorAll(".bd-media-tr").forEach((row) => {
                    const on = row.dataset.path === selectedValue;
                    row.classList.toggle("selected", on);
                    if (on && scroll) row.scrollIntoView({ block: "nearest" });
                });
                renderPreview(itemsByPath.get(selectedValue));
            };

            const syncMultiRows = () => {
                tbodyEl.querySelectorAll(".bd-media-tr").forEach((row) => {
                    const on = multiSel.has(row.dataset.path);
                    // checked = 已勾选（左侧蓝条）；selected = 当前预览项（底色）。
                    row.classList.toggle("checked", on);
                    row.classList.toggle("selected", row.dataset.path === selectedValue);
                    const cb = row.querySelector(".bd-media-cb");
                    if (cb) cb.checked = on;
                });
                const n = multiSel.size;
                statusEl.textContent = n
                    ? t("mediaPicker.multiSelected", { n })
                    : t("mediaPicker.count", { n: listedItems.length });
                okBtn.disabled = n === 0;
            };

            const renderRows = () => {
                tbodyEl.innerHTML = "";
                const rows = sortedItems();
                if (!rows.length) {
                    const empty = document.createElement("div");
                    empty.className = "bd-media-empty-row";
                    empty.textContent = t("mediaPicker.empty");
                    tbodyEl.appendChild(empty);
                    selectedValue = "";
                    renderPreview(null);
                    syncHeaderState();
                    return;
                }
                if (!multi && (!selectedValue || !itemsByPath.has(selectedValue))) {
                    selectedValue = rows[0].relPath || "";
                }
                for (const item of rows) {
                    const row = document.createElement("div");
                    row.className = "bd-media-tr";
                    row.dataset.path = item.relPath;
                    if (multi) {
                        if (multiSel.has(item.relPath)) row.classList.add("checked");
                        if (item.relPath === selectedValue) row.classList.add("selected");
                    } else if (item.relPath === selectedValue) {
                        row.classList.add("selected");
                    }
                    if (multi) {
                        const cbTd = document.createElement("div");
                        cbTd.className = "bd-media-td bd-media-td-cb";
                        const cb = document.createElement("input");
                        cb.type = "checkbox";
                        cb.className = "bd-media-cb";
                        cb.checked = multiSel.has(item.relPath);
                        cb.onclick = (e) => e.stopPropagation();
                        cb.onchange = () => selectRow(item.relPath, { toggleCheck: true });
                        cbTd.appendChild(cb);
                        row.appendChild(cbTd);
                    }
                    const nameTd = document.createElement("div");
                    nameTd.className = "bd-media-td bd-media-td-name";
                    const mediaPrefix = kind === "reference_audio"
                        ? (item.mediaKind === "video" ? "🎞 " : "♪ ")
                        : "";
                    nameTd.textContent = `${mediaPrefix}${item.relPath || item.fileName || item.name || ""}`;
                    nameTd.title = nameTd.textContent;
                    const timeTd = document.createElement("div");
                    timeTd.className = "bd-media-td bd-media-td-time";
                    timeTd.textContent = formatMediaTime(item.modified);
                    if (showDims) {
                        const dimsTd = document.createElement("div");
                        dimsTd.className = "bd-media-td bd-media-td-dims";
                        dimsTd.textContent = dimsText(item);
                        row.append(nameTd, dimsTd, timeTd);
                    } else {
                        row.append(nameTd, timeTd);
                    }
                    row.addEventListener("click", () => selectRow(item.relPath));
                    row.addEventListener("dblclick", () => {
                        if (multi) return; // 批量模式下双击不提交，避免误关
                        const choice = selectedChoice();
                        if (choice) finish(choice);
                    });
                    tbodyEl.appendChild(row);
                }
                syncHeaderState();
                if (multi) {
                    // 默认预览第一项，避免刚打开弹窗时右侧预览区一片空白。
                    if (!selectedValue && rows.length) {
                        selectedValue = rows[0].relPath || "";
                        renderPreview(itemsByPath.get(selectedValue));
                    }
                    syncMultiRows();
                } else {
                    renderPreview(itemsByPath.get(selectedValue));
                    const selectedRow = tbodyEl.querySelector(".bd-media-tr.selected");
                    selectedRow?.scrollIntoView({ block: "nearest" });
                }
            };

            const moveSelection = (delta) => {
                if (multi) return; // 批量模式下方向键不应改动勾选集合
                const rows = sortedItems();
                if (!rows.length) return;
                const idx = rows.findIndex((item) => item.relPath === selectedValue);
                const next = rows[Math.max(0, Math.min(rows.length - 1, (idx < 0 ? 0 : idx) + delta))];
                if (next) selectRow(next.relPath, { scroll: true });
            };

            const loadItems = async () => {
                statusEl.textContent = t("mediaPicker.loading");
                tbodyEl.innerHTML = "";
                renderPreview(null);
                try {
                    listedItems = await this.listInputMedia(kind, { includeCache });
                    itemsByPath = new Map(listedItems.map((item) => [item.relPath, item]));
                    renderRows();
                    if (multi) {
                        syncMultiRows();
                    } else {
                        statusEl.textContent = listedItems.length
                            ? t("mediaPicker.count", { n: listedItems.length })
                            : t("mediaPicker.empty");
                    }
                } catch (err) {
                    listedItems = [];
                    itemsByPath = new Map();
                    tbodyEl.innerHTML = "";
                    statusEl.textContent = err?.message || String(err);
                }
            };

            thEls.forEach((th) => {
                th.addEventListener("click", () => {
                    const key = th.dataset.sort || "time";
                    if (sortKey === key) {
                        sortDir = sortDir === "asc" ? "desc" : "asc";
                    } else {
                        sortKey = key;
                        sortDir = key === "name" ? "asc" : "desc";
                    }
                    renderRows();
                });
            });

            const refreshBtn = document.createElement("button");
            refreshBtn.type = "button";
            refreshBtn.className = "bd-btn";
            refreshBtn.textContent = t("mediaPicker.refresh");
            refreshBtn.onclick = () => { void loadItems(); };
            actionsTop.appendChild(refreshBtn);

            const uploadBtn = document.createElement("button");
            uploadBtn.type = "button";
            uploadBtn.className = "bd-btn";
            uploadBtn.textContent = t("mediaPicker.upload");
            uploadBtn.onclick = async () => {
                const file = await this.pickLocalFile(accept || "");
                if (!file) return;
                finish(multi ? [{ source: "file", file }] : { source: "file", file });
            };
            actionsTop.appendChild(uploadBtn);

            if (multi) {
                const selAll = document.createElement("button");
                selAll.type = "button";
                selAll.className = "bd-btn";
                selAll.textContent = t("mediaPicker.selectAll");
                selAll.onclick = () => {
                    for (const it of listedItems) multiSel.add(it.relPath);
                    syncMultiRows();
                    renderRows();
                };
                actionsTop.appendChild(selAll);
                const clearSel = document.createElement("button");
                clearSel.type = "button";
                clearSel.className = "bd-btn";
                clearSel.textContent = t("mediaPicker.clearSelection");
                clearSel.onclick = () => {
                    multiSel.clear();
                    syncMultiRows();
                    renderRows();
                };
                actionsTop.appendChild(clearSel);
            }

            const actionsBottom = document.createElement("div");
            actionsBottom.className = "bd-modal-actions";
            const cancelBtn = document.createElement("button");
            cancelBtn.type = "button";
            cancelBtn.className = "bd-btn";
            cancelBtn.textContent = t("dialog.cancel");
            cancelBtn.onclick = () => finish(null);
            actionsBottom.appendChild(cancelBtn);

            const okBtn = document.createElement("button");
            okBtn.type = "button";
            okBtn.className = "bd-btn bd-btn-primary";
            okBtn.textContent = multi
                ? t("mediaPicker.useSelectedMulti")
                : t("mediaPicker.useSelected");
            okBtn.onclick = () => {
                if (multi) {
                    const choices = selectedChoices();
                    if (choices.length) finish(choices);
                    return;
                }
                const choice = selectedChoice();
                if (choice) finish(choice);
            };
            actionsBottom.appendChild(okBtn);
            panel.appendChild(actionsBottom);

            overlay.onclick = (e) => {
                if (e.target === overlay) finish(null);
            };
            panel.onclick = (e) => e.stopPropagation();

            this._modalKeyHandler = (e) => {
                if (e.key === "Escape") {
                    e.preventDefault();
                    e.stopPropagation();
                    finish(null);
                    return;
                }
                if (e.key === "ArrowDown" || e.key === "ArrowUp") {
                    if (!panel.contains(e.target) && e.target !== document.body) return;
                    e.preventDefault();
                    moveSelection(e.key === "ArrowDown" ? 1 : -1);
                    return;
                }
                if (e.key !== "Enter") return;
                if (e.target?.closest?.(".bd-media-th, .bd-media-head-actions, button.bd-btn:not(.bd-btn-primary)")) return;
                e.preventDefault();
                okBtn.click();
            };
            window.addEventListener("keydown", this._modalKeyHandler, true);

            overlay.appendChild(panel);
            this.root.appendChild(overlay);
            this._modalEl = overlay;
            void loadItems();
            tableEl.focus();
        });
    },
    async chooseImageInput(opts = {}) {
        const choice = await this.showInputMediaPicker({
            kind: "image",
            title: opts.title || t("mediaPicker.pickImage"),
            accept: "image/*,.jpg,.jpeg,.png,.webp,.bmp,.gif,.tif,.tiff",
            currentValue: opts.currentValue || "",
        });
        return this._resolveImageChoice(choice);
    },
    /** 批量版：返回数组（元素结构同 chooseImageInput）。 */
    async chooseImageInputs(opts = {}) {
        const choices = await this.showInputMediaPicker({
            kind: "image",
            title: opts.title || t("mediaPicker.pickReferenceImage"),
            accept: "image/*,.jpg,.jpeg,.png,.webp,.bmp,.gif,.tif,.tiff",
            currentValue: opts.currentValue || "",
            multi: true,
        });
        if (!Array.isArray(choices) || !choices.length) return [];
        const out = [];
        for (const c of choices) {
            const r = await this._resolveImageChoice(c);
            if (r) out.push(r);
        }
        return out;
    },
    async _resolveImageChoice(choice) {
        if (!choice) return null;
        if (choice.source === "file" && choice.file) {
            const uploaded = await uploadToInput(choice.file);
            const relPath = videoRelativePath(uploaded);
            const dims = await this.probeInputImageDimensions(relPath, uploaded.type || "input");
            return {
                imageFile: relPath,
                fileName: uploaded?.name || choice.file.name || relPath,
                subfolder: uploaded?.subfolder || "",
                type: uploaded?.type || "input",
                width: dims.width || 0,
                height: dims.height || 0,
            };
        }
        const dims = await this.probeInputImageDimensions(choice.relPath, choice.type || "input");
        return {
            imageFile: choice.relPath,
            fileName: choice.fileName || choice.relPath,
            subfolder: choice.subfolder || "",
            type: choice.type || "input",
            width: dims.width || 0,
            height: dims.height || 0,
        };
    },
    async chooseVideoInput(opts = {}) {
        const choice = await this.showInputMediaPicker({
            kind: "video",
            title: opts.title || t("mediaPicker.pickVideo"),
            accept: "video/*,.mp4,.mov,.webm,.mkv,.avi,.m4v,.mpg,.mpeg,.mts,.ts",
            currentValue: opts.currentValue || "",
            // Rendered clips count as pickable source material here — that is
            // the whole point of reusing an earlier render as a reference.
            // Entries render as ordinary rows; nothing about the picker's
            // look changes.
            includeCache: true,
        });
        return this._resolveVideoChoice(choice);
    },
    /** 批量版：返回数组（元素结构同 chooseVideoInput）。 */
    async chooseVideoInputs(opts = {}) {
        const choices = await this.showInputMediaPicker({
            kind: "video",
            title: opts.title || t("mediaPicker.pickReferenceVideo"),
            accept: "video/*,.mp4,.mov,.webm,.mkv,.avi,.m4v,.mpg,.mpeg,.mts,.ts",
            currentValue: opts.currentValue || "",
            multi: true,
            includeCache: true,
        });
        if (!Array.isArray(choices) || !choices.length) return [];
        const out = [];
        for (const c of choices) {
            const r = await this._resolveVideoChoice(c);
            if (r) out.push(r);
        }
        return out;
    },
    async _resolveVideoChoice(choice) {
        if (!choice) return null;
        if (choice.source === "file" && choice.file) {
            const uploaded = await uploadToInputSmart(choice.file);
            return {
                relPath: videoRelativePath(uploaded),
                fileName: uploaded?.name || choice.file.name || "",
                subfolder: uploaded?.subfolder || "",
                type: uploaded?.type || "input",
            };
        }
        return {
            relPath: choice.relPath,
            fileName: choice.fileName || choice.relPath,
            subfolder: choice.subfolder || "",
            type: choice.type || "input",
        };
    },
    async chooseAudioInput(opts = {}) {
        const choice = await this.showInputMediaPicker({
            kind: "reference_audio",
            title: opts.title || t("mediaPicker.pickAudio"),
            accept: "audio/*,video/*,.wav,.mp3,.flac,.ogg,.m4a,.aac,.wma,.mp4,.mov,.webm,.mkv,.avi,.m4v,.mpg,.mpeg,.mts,.ts",
            currentValue: opts.currentValue || "",
        });
        return this._resolveAudioChoice(choice);
    },
    /** 批量版：返回数组（元素结构同 chooseAudioInput）。 */
    async chooseAudioInputs(opts = {}) {
        const choices = await this.showInputMediaPicker({
            kind: "reference_audio",
            title: opts.title || t("mediaPicker.pickReferenceAudio"),
            accept: "audio/*,video/*,.wav,.mp3,.flac,.ogg,.m4a,.aac,.wma,.mp4,.mov,.webm,.mkv,.avi,.m4v,.mpg,.mpeg,.mts,.ts",
            currentValue: opts.currentValue || "",
            multi: true,
        });
        if (!Array.isArray(choices) || !choices.length) return [];
        const out = [];
        for (const c of choices) {
            const r = await this._resolveAudioChoice(c);
            if (r) out.push(r);
        }
        return out;
    },
    async _resolveAudioChoice(choice) {
        if (!choice) return null;
        if (choice.source === "file" && choice.file) {
            return prepareLocalReferenceAudio(choice.file);
        }
        if (choice.mediaKind === "video") {
            return extractReferenceAudioFromExistingVideo(choice);
        }
        return {
            relPath: choice.relPath,
            fileName: choice.fileName || choice.relPath,
            subfolder: choice.subfolder || "",
            type: choice.type || "input",
        };
    }
};
