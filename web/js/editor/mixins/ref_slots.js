/** ref_slots mixin for the Director editor (ref_slots).
 *
 * Extracted verbatim from editor.js; methods are unchanged apart from the
 * trailing comma an object literal needs.
 */

import { uploadToInput, uploadToInputSmart } from "../../core/upload.js";
import { relPath } from "../../core/utils.js";
import { refViewUrl, videoRelativePath } from "../urls.js";
import { MAX_REFERENCE_AUDIOS, MAX_REFERENCE_IMAGES, MAX_REFERENCE_VIDEOS, refAudioLabel, refImageLabel, refVideoLabel, resolveTaskKey } from "../../minimax_gen_timeline.js";
import { t } from "../../minimax_i18n.js";
import { bindR2vMediaPlayback, formatMediaDuration, rebaseR2vGroupSlotsForCommon, wireMediaDuration } from "../../minimax_image_batch.js";
import { refreshPromptTokenEditors } from "../../minimax_prompt_mentions.js";
import { hasDuplicateReferenceAudio, prepareLocalReferenceAudio } from "../../minimax_ref_audio.js";
export const ref_slotsMixin = {
    renderRefSlots(refs, box, isGlobal) {
        if (!box) return;
        box.innerHTML = "";
        const target = isGlobal
            ? this.timeline.global
            : this.timeline.segments[this.selectedIndex];
        const taskKey = isGlobal
            ? this.getTaskKey()
            : resolveTaskKey(
                target?.taskType || this.timeline.global?.taskType || this.globalTask?.value || this.getTaskKey(),
            );
        const polished = this.usesRv2vRefStyle(taskKey);
        const wrap = isGlobal ? this.globalRefsImagesWrap : this.segRefsImagesWrap;
        const countEl = isGlobal ? this.globalRefsCount : this.segRefsCount;
        const PIC_STEP = 3;
        const PIC_SLOTS = MAX_REFERENCE_IMAGES;

        let filled = 0;
        let highestFilled = -1;
        for (const r of refs || []) {
            const idx = Number(r.index ?? r.slot);
            const has = !!(r?.imageFile || r?.imageB64);
            if (!has || !Number.isFinite(idx) || idx < 0 || idx >= PIC_SLOTS) continue;
            filled += 1;
            highestFilled = Math.max(highestFilled, idx);
        }
        if (countEl) countEl.textContent = polished ? `${filled}/${PIC_SLOTS}` : "";
        this._syncPickExistingDisabled(
            isGlobal ? '[data-r="global-refs-pick"]' : '[data-r="seg-refs-pick"]',
            filled >= PIC_SLOTS,
        );

        if (!this._rv2vPicsVisible) this._rv2vPicsVisible = {};
        const visKey = isGlobal ? "global" : `seg:${target?.id ?? this.selectedIndex}`;
        const minVisible = highestFilled >= 0
            ? Math.min(PIC_SLOTS, Math.ceil((highestFilled + 1) / PIC_STEP) * PIC_STEP)
            : PIC_STEP;
        let visible = polished
            ? (Number(this._rv2vPicsVisible[visKey]) || PIC_STEP)
            : PIC_SLOTS;
        if (polished) {
            visible = Math.max(PIC_STEP, Math.min(PIC_SLOTS, visible));
            if (visible < minVisible) visible = minVisible;
            this._rv2vPicsVisible[visKey] = visible;
        }

        for (let i = 0; i < PIC_SLOTS; i++) {
            const el = document.createElement("div");
            el.className = "bd-ref";
            if (polished && i >= visible) el.classList.add("bd-r2v-pic-hidden");
            el.dataset.refSlot = String(i);
            el.dataset.refKind = "image";
            el.dataset.refIndex = String(i);
            el.dataset.refScope = isGlobal ? "global" : "seg";
            const label = refImageLabel(i);
            el.title = t("ref.slotTitle", { label });
            const ref = (refs || []).find((r) => Number(r.index ?? r.slot) === i);
            const tag = document.createElement("span");
            tag.className = polished ? "cap" : "bd-ref-tag";
            tag.textContent = label;
            el.appendChild(tag);
            if (ref?.imageFile) {
                el.classList.add("has-img");
                const img = document.createElement("img");
                img.src = refViewUrl(ref.imageFile);
                img.draggable = false;
                el.appendChild(img);
                if (polished) {
                    const dot = document.createElement("span");
                    dot.className = "dot";
                    el.appendChild(dot);
                }
                const x = document.createElement("span");
                x.className = "x";
                x.textContent = "×";
                x.onclick = (e) => {
                    e.stopPropagation();
                    this.removeRef(target, i);
                };
                el.appendChild(x);
            } else if (ref?.imageB64) {
                el.classList.add("has-img");
                const img = document.createElement("img");
                img.src = ref.imageB64.startsWith("data:") ? ref.imageB64 : `data:image/png;base64,${ref.imageB64}`;
                img.draggable = false;
                el.appendChild(img);
                if (polished) {
                    const dot = document.createElement("span");
                    dot.className = "dot";
                    el.appendChild(dot);
                }
                const x = document.createElement("span");
                x.className = "x";
                x.textContent = "×";
                x.onclick = (e) => {
                    e.stopPropagation();
                    this.removeRef(target, i);
                };
                el.appendChild(x);
            }
            this._bindRefSlotDnD(el, target, i, isGlobal);
            el.onclick = () => {
                if (this._refDragMoved) {
                    this._refDragMoved = false;
                    return;
                }
                this.pickRef(target, i, isGlobal);
            };
            box.appendChild(el);
        }

        wrap?.querySelectorAll(".bd-r2v-pics-toggle").forEach((btn) => btn.remove());
        if (polished && wrap) {
            const toggle = document.createElement("button");
            toggle.type = "button";
            toggle.className = "bd-r2v-pics-toggle";
            const syncToggleLabel = () => {
                if (visible < PIC_SLOTS) {
                    const next = Math.min(PIC_STEP, PIC_SLOTS - visible);
                    toggle.textContent = t("batch.r2v.expandPics", { n: next });
                } else {
                    toggle.textContent = t("batch.r2v.collapsePics");
                }
            };
            syncToggleLabel();
            toggle.onclick = (e) => {
                e.stopPropagation();
                if (visible < PIC_SLOTS) {
                    visible = Math.min(PIC_SLOTS, visible + PIC_STEP);
                } else {
                    visible = Math.max(PIC_STEP, minVisible);
                }
                this._rv2vPicsVisible[visKey] = visible;
                box.querySelectorAll(".bd-ref").forEach((el, i) => {
                    el.classList.toggle("bd-r2v-pic-hidden", i >= visible);
                });
                syncToggleLabel();
                this.updateDomWidgetHeight?.();
            };
            wrap.appendChild(toggle);
        }
        refreshPromptTokenEditors(this.root || document);
    },
    _bindRefSlotDnD(el, target, slotIndex, isGlobal) {
        const hasImg = el.classList.contains("has-img");
        el.draggable = hasImg;
        el.addEventListener("dragstart", (e) => {
            if (!hasImg) {
                e.preventDefault();
                return;
            }
            this._refDragMoved = false;
            const payload = JSON.stringify({
                scope: isGlobal ? "global" : "seg",
                segIndex: isGlobal ? -1 : this.selectedIndex,
                from: slotIndex,
            });
            e.dataTransfer.setData("application/x-minimax-ref-slot", payload);
            e.dataTransfer.setData("text/plain", payload);
            e.dataTransfer.effectAllowed = "move";
        });
        el.addEventListener("dragend", () => {
            // click may fire after dragend; keep suppress for one tick
            setTimeout(() => { this._refDragMoved = false; }, 0);
        });
        el.addEventListener("dragover", (e) => {
            const types = e.dataTransfer?.types || [];
            if (![...types].includes("application/x-minimax-ref-slot") && ![...types].includes("Files")) {
                return;
            }
            e.preventDefault();
            e.stopPropagation();
            e.dataTransfer.dropEffect = [...types].includes("application/x-minimax-ref-slot")
                ? "move"
                : "copy";
        });
        el.addEventListener("drop", (e) => {
            e.preventDefault();
            e.stopPropagation();
            const raw = e.dataTransfer.getData("application/x-minimax-ref-slot")
                || e.dataTransfer.getData("text/plain");
            if (raw) {
                try {
                    const data = JSON.parse(raw);
                    const scope = isGlobal ? "global" : "seg";
                    if (data.scope !== scope) return;
                    if (!isGlobal && data.segIndex !== this.selectedIndex) return;
                    this._refDragMoved = true;
                    this.moveRefSlot(target, Number(data.from), slotIndex, isGlobal);
                    return;
                } catch (_) { /* fall through to file drop */ }
            }
            const f = e.dataTransfer.files?.[0];
            if (f?.type?.startsWith("image/")) {
                this.addRefFromFile(f, target, slotIndex, isGlobal);
            }
        });
    },
    moveRefSlot(target, fromIndex, toIndex, isGlobal) {
        if (!target || fromIndex === toIndex) return;
        const refs = [...(target.refs || [])];
        const fromRef = refs.find((r) => Number(r.index ?? r.slot) === fromIndex);
        if (!fromRef) return;
        const toRef = refs.find((r) => Number(r.index ?? r.slot) === toIndex);
        target.refs = refs.filter((r) => {
            const idx = Number(r.index ?? r.slot);
            return idx !== fromIndex && idx !== toIndex;
        });
        target.refs.push({ ...fromRef, index: toIndex, slot: undefined });
        if (toRef) {
            target.refs.push({ ...toRef, index: fromIndex, slot: undefined });
        }
        if (isGlobal) {
            this.timeline.global = target;
            if (this.isR2vCommonEnabled()) rebaseR2vGroupSlotsForCommon(this);
        }
        this.commit();
    },
    removeRef(target, index) {
        target.refs = (target.refs || []).filter((r) => Number(r.index ?? r.slot) !== index);
        if (this.isR2vCommonEnabled() && target === this.timeline.global) {
            rebaseR2vGroupSlotsForCommon(this);
        }
        this.commit();
    },
    renderRefAudioSlots() {
        const isGlobal = this.usesGlobalRefPanel();
        const box = isGlobal ? this.globalRefAudiosBox : this.segRefAudiosBox;
        if (!box) return;
        const target = isGlobal
            ? (this.timeline.global = this.timeline.global || { refs: [], refAudios: [] })
            : this.timeline.segments[this.selectedIndex];
        if (!target) return;
        target.refAudios = target.refAudios || [];
        const taskKey = isGlobal
            ? this.getTaskKey()
            : resolveTaskKey(
                target?.taskType || this.timeline.global?.taskType || this.globalTask?.value || this.getTaskKey(),
            );
        const polished = this.usesRv2vRefStyle(taskKey);
        const countEl = isGlobal ? this.globalAudiosCount : this.segAudiosCount;
        let filled = 0;
        for (const r of target.refAudios) {
            if (r?.audioFile || r?.fileName) filled += 1;
        }
        if (countEl) countEl.textContent = polished ? `${filled}/${MAX_REFERENCE_AUDIOS}` : "";
        this._syncPickExistingDisabled(
            isGlobal ? '[data-r="global-audios-pick"]' : '[data-r="seg-audios-pick"]',
            filled >= MAX_REFERENCE_AUDIOS,
        );

        box.innerHTML = "";
        for (let i = 0; i < MAX_REFERENCE_AUDIOS; i++) {
            const el = document.createElement("div");
            el.className = "bd-ref-audio";
            el.dataset.audioSlot = String(i);
            el.dataset.refKind = "audio";
            el.dataset.refIndex = String(i);
            const label = refAudioLabel(i);
            const ref = (target.refAudios || []).find((r) => Number(r.index ?? r.slot) === i);
            const file = ref?.audioFile || ref?.fileName || "";
            el.title = file
                ? t("ref.audioTitleFilled", { label, file })
                : t("ref.audioTitleEmpty", { label });
            if (polished) {
                const thumb = document.createElement("div");
                thumb.className = "bd-r2v-thumb";
                const meta = document.createElement("div");
                meta.className = "bd-r2v-meta";
                const tag = document.createElement("span");
                tag.className = "tag";
                tag.textContent = label;
                meta.appendChild(tag);
                el.appendChild(thumb);
                el.appendChild(meta);
                if (file) {
                    el.classList.add("has-audio");
                    const playBtn = document.createElement("button");
                    playBtn.type = "button";
                    playBtn.className = "bd-r2v-play";
                    playBtn.title = t("batch.r2v.play");
                    playBtn.textContent = "▶";
                    thumb.appendChild(playBtn);
                    const dur = document.createElement("span");
                    dur.className = "bd-r2v-dur";
                    dur.textContent = ref?.durationSec != null
                        ? formatMediaDuration(ref.durationSec)
                        : "--:--";
                    meta.appendChild(dur);
                    const progress = document.createElement("div");
                    progress.className = "bd-r2v-progress";
                    progress.title = t("batch.r2v.seek");
                    progress.innerHTML = `<div class="bd-r2v-progress-fill"></div>`;
                    el.appendChild(progress);
                    const audio = document.createElement("audio");
                    audio.preload = "metadata";
                    audio.src = refViewUrl(file);
                    audio.className = "bd-r2v-media";
                    el.appendChild(audio);
                    bindR2vMediaPlayback(audio, playBtn, progress);
                    wireMediaDuration(audio, dur, (sec) => {
                        if (ref) ref.durationSec = sec;
                    });
                    const x = document.createElement("span");
                    x.className = "x";
                    x.textContent = "×";
                    x.onclick = (e) => {
                        e.stopPropagation();
                        this.removeRefAudio(target, i);
                    };
                    el.appendChild(x);
                } else {
                    thumb.textContent = "♪";
                    const hint = document.createElement("span");
                    hint.className = "name";
                    hint.textContent = t("batch.r2v.uploadHint");
                    meta.appendChild(hint);
                }
            } else if (file) {
                el.classList.add("has-audio");
                const tag = document.createElement("span");
                tag.textContent = label;
                el.appendChild(tag);
                const name = document.createElement("span");
                name.className = "bd-ref-audio-name";
                name.textContent = file.split("/").pop() || file;
                el.appendChild(name);
                const x = document.createElement("span");
                x.className = "x";
                x.textContent = "×";
                x.onclick = (e) => {
                    e.stopPropagation();
                    this.removeRefAudio(target, i);
                };
                el.appendChild(x);
            } else {
                el.textContent = t("ref.audioUpload", { label });
            }
            el.onclick = (e) => {
                if (e.target?.closest?.(".bd-r2v-play, .bd-r2v-progress, .x")) return;
                this.pickRefAudio(target, i);
            };
            box.appendChild(el);
        }
        refreshPromptTokenEditors(this.root || document);
    },
    removeRefAudio(target, index) {
        if (!target) return;
        target.refAudios = (target.refAudios || []).filter((r) => Number(r.index ?? r.slot) !== index);
        if (this.isR2vCommonEnabled() && target === this.timeline.global) {
            rebaseR2vGroupSlotsForCommon(this);
        }
        this.commit();
        this.renderRefAudioSlots();
    },
    pickRefAudio(target, index) {
        const input = document.createElement("input");
        input.type = "file";
        input.accept = "audio/*,video/*,.wav,.mp3,.flac,.ogg,.m4a,.aac,.wma,.mp4,.mov,.webm,.mkv,.avi,.m4v,.mpg,.mpeg,.mts,.ts";
        input.onchange = () => {
            const file = input.files?.[0];
            if (file) this.addRefAudioFromFile(file, target, index);
        };
        input.click();
    },
    async addRefAudioFromFile(file, target, slotIndex = null) {
        if (!target || !file) return;
        target.refAudios = target.refAudios || [];
        let index = slotIndex;
        if (index == null) {
            index = Array.from({ length: MAX_REFERENCE_AUDIOS }, (_, i) => i)
                .find((i) => !target.refAudios.some((r) => Number(r.index ?? r.slot) === i));
            if (index == null) return;
        }
        try {
            const prepared = await prepareLocalReferenceAudio(file);
            const relPath = prepared.relPath;
            if (hasDuplicateReferenceAudio(target.refAudios, relPath, index)) {
                alert(t("ref.audioDuplicate"));
                return;
            }
            target.refAudios = target.refAudios.filter((r) => Number(r.index ?? r.slot) !== index);
            target.refAudios.push({
                index,
                audioFile: relPath,
                fileName: prepared.fileName || file.name,
                type: prepared.type || "input",
                subfolder: prepared.subfolder || "",
            });
            if (this.isR2vCommonEnabled() && target === this.timeline.global) {
                rebaseR2vGroupSlotsForCommon(this);
                this.renderImageBatchGroups?.();
            }
            this.commit();
            this.renderRefAudioSlots();
        } catch (err) {
            console.error("[MiniMax H3Director] ref audio upload failed:", err);
            alert(t("upload.refAudioFailed", { err: err?.message || err }));
        }
    },
    /** r2v common panel: multi-slot global.refVideos (1–3), merged into groups at run time. */
    renderR2vCommonVideoSlots() {
        const box = this.globalRefVideosBox;
        if (!box || !this.usesR2vCommonPanel()) return;
        const target = (this.timeline.global = this.timeline.global || {
            refs: [], refAudios: [], refVideos: [],
        });
        target.refVideos = target.refVideos || [];
        let filled = 0;
        for (const r of target.refVideos) {
            if (r?.videoFile || r?.fileName || r?.previewImageFile || r?.previewImageUrl || r?.linked) {
                filled += 1;
            }
        }
        if (this.globalVideosCount) {
            this.globalVideosCount.textContent = `${filled}/${MAX_REFERENCE_VIDEOS}`;
        }
        this._syncPickExistingDisabled('[data-r="global-videos-pick"]', filled >= MAX_REFERENCE_VIDEOS);
        box.innerHTML = "";
        for (let i = 0; i < MAX_REFERENCE_VIDEOS; i++) {
            const el = document.createElement("div");
            el.className = "bd-ref-video";
            el.dataset.videoSlot = String(i);
            el.dataset.refKind = "video";
            el.dataset.refIndex = String(i);
            const label = refVideoLabel(i);
            const ref = (target.refVideos || []).find((r) => Number(r.index ?? r.slot) === i);
            const file = ref?.videoFile || "";
            const posterSrc = ref?.previewImageUrl
                || (ref?.previewImageFile ? refViewUrl(ref.previewImageFile) : "");
            const hasMedia = !!(file || posterSrc || ref?.linked);
            const titleFile = file || ref?.fileName || ref?.previewImageFile || "";
            el.title = hasMedia
                ? t("ref.videoTitleFilled", { label, file: titleFile || label })
                : t("ref.videoTitleEmpty", { label });
            const thumb = document.createElement("div");
            thumb.className = "bd-r2v-thumb bd-r2v-thumb-video";
            const meta = document.createElement("div");
            meta.className = "bd-r2v-meta";
            const tag = document.createElement("span");
            tag.className = "tag";
            tag.textContent = label;
            meta.appendChild(tag);
            el.appendChild(thumb);
            el.appendChild(meta);
            if (file) {
                el.classList.add("has-video");
                const video = document.createElement("video");
                video.preload = "metadata";
                video.muted = true;
                video.playsInline = true;
                video.src = refViewUrl(file);
                video.className = "bd-r2v-media";
                thumb.appendChild(video);
                const playBtn = document.createElement("button");
                playBtn.type = "button";
                playBtn.className = "bd-r2v-play";
                playBtn.title = t("batch.r2v.play");
                playBtn.textContent = "▶";
                thumb.appendChild(playBtn);
                const dur = document.createElement("span");
                dur.className = "bd-r2v-dur";
                dur.textContent = ref?.durationSec != null
                    ? formatMediaDuration(ref.durationSec)
                    : "--:--";
                meta.appendChild(dur);
                bindR2vMediaPlayback(video, playBtn);
                playBtn.addEventListener("click", () => { video.muted = false; });
                wireMediaDuration(video, dur, (sec) => {
                    if (ref) ref.durationSec = sec;
                });
                video.addEventListener("loadeddata", () => {
                    if (video.readyState >= 2 && video.currentTime < 0.05) {
                        try {
                            video.currentTime = Math.min(0.1, (video.duration || 1) * 0.05);
                        } catch (_) { /* ignore */ }
                    }
                }, { once: true });
                const x = document.createElement("span");
                x.className = "x";
                x.textContent = "×";
                x.onclick = (e) => {
                    e.stopPropagation();
                    this.removeR2vCommonVideo(i);
                };
                el.appendChild(x);
            } else if (posterSrc) {
                el.classList.add("has-video");
                const img = document.createElement("img");
                img.className = "bd-r2v-media";
                img.src = posterSrc;
                img.alt = label;
                thumb.appendChild(img);
                const hint = document.createElement("span");
                hint.className = "name";
                hint.textContent = t("batch.r2v.externalPoster");
                meta.appendChild(hint);
            } else {
                thumb.textContent = "▶";
                const hint = document.createElement("span");
                hint.className = "name";
                hint.textContent = t("batch.r2v.uploadHint");
                meta.appendChild(hint);
            }
            el.onclick = (e) => {
                if (e.target?.closest?.(".bd-r2v-play, .bd-r2v-dur, .x, video")) return;
                if (file && e.target?.closest?.(".bd-r2v-thumb")) {
                    el.querySelector(".bd-r2v-play")?.click();
                    return;
                }
                this.pickR2vCommonVideo(i);
            };
            box.appendChild(el);
        }
        refreshPromptTokenEditors(this.root || document);
    },
    removeR2vCommonVideo(index) {
        const target = this.timeline.global;
        if (!target) return;
        target.refVideos = (target.refVideos || []).filter((r) => Number(r.index ?? r.slot) !== index);
        if (this.isR2vCommonEnabled()) {
            rebaseR2vGroupSlotsForCommon(this);
            this.renderImageBatchGroups?.();
        }
        this.commit();
        this.renderR2vCommonVideoSlots();
    },
    pickR2vCommonVideo(index) {
        const input = document.createElement("input");
        input.type = "file";
        input.accept = "video/*,.mp4,.mov,.webm,.mkv";
        input.onchange = () => {
            const file = input.files?.[0];
            if (file) this.addR2vCommonVideoFromFile(file, index);
        };
        input.click();
    },
    async pickExistingR2vCommonVideo() {
        const target = (this.timeline.global = this.timeline.global || {
            refs: [], refAudios: [], refVideos: [],
        });
        target.refVideos = target.refVideos || [];
        const index = Array.from({ length: MAX_REFERENCE_VIDEOS }, (_, i) => i)
            .find((i) => !target.refVideos.some((r) => Number(r.index ?? r.slot) === i && (r.videoFile || r.fileName)));
        if (index == null) {
            alert(t("mediaPicker.slotsFull"));
            return;
        }
        try {
            const picked = await this.chooseVideoInput({
                title: t("mediaPicker.pickReferenceVideo"),
            });
            if (!picked?.relPath) return;
            target.refVideos = target.refVideos.filter((r) => Number(r.index ?? r.slot) !== index);
            target.refVideos.push({
                index,
                videoFile: picked.relPath,
                fileName: picked.fileName || picked.relPath,
                type: picked.type || "input",
                subfolder: picked.subfolder || "",
            });
            if (this.isR2vCommonEnabled()) {
                rebaseR2vGroupSlotsForCommon(this);
                this.renderImageBatchGroups?.();
            }
            this.commit();
            this.renderR2vCommonVideoSlots();
        } catch (err) {
            console.error("[MiniMax H3Director] common ref video pick failed:", err);
            alert(t("upload.refVideoBatchFailed", { err: err?.message || err }));
        }
    },
    async addR2vCommonVideoFromFile(file, slotIndex = null) {
        if (!file) return;
        const target = (this.timeline.global = this.timeline.global || {
            refs: [], refAudios: [], refVideos: [],
        });
        target.refVideos = target.refVideos || [];
        let index = slotIndex;
        if (index == null) {
            index = Array.from({ length: MAX_REFERENCE_VIDEOS }, (_, i) => i)
                .find((i) => !target.refVideos.some((r) => Number(r.index ?? r.slot) === i));
            if (index == null) return;
        }
        try {
            const uploaded = await uploadToInputSmart(file);
            const relPath = videoRelativePath(uploaded);
            target.refVideos = target.refVideos.filter((r) => Number(r.index ?? r.slot) !== index);
            target.refVideos.push({
                index,
                videoFile: relPath,
                fileName: uploaded?.name || file.name,
                type: "input",
                subfolder: uploaded?.subfolder || "",
            });
            if (this.isR2vCommonEnabled()) {
                rebaseR2vGroupSlotsForCommon(this);
                this.renderImageBatchGroups?.();
            }
            this.commit();
            this.renderR2vCommonVideoSlots();
        } catch (err) {
            console.error("[MiniMax H3Director] common ref video upload failed:", err);
            alert(t("upload.refVideoBatchFailed", { err: err?.message || err }));
        }
    },
    pickRef(target, index, isGlobal) {
        const input = document.createElement("input");
        input.type = "file"; input.accept = "image/*";
        input.onchange = () => {
            const file = input.files?.[0];
            if (file) this.addRefFromFile(file, target, index, isGlobal);
        };
        input.click();
    },
    _nextEmptyMediaSlot(items, max, hasFn) {
        for (let i = 0; i < max; i++) {
            const hit = (items || []).find((r) => Number(r.index ?? r.slot) === i);
            if (!hasFn(hit)) return i;
        }
        return -1;
    },
    _syncPickExistingDisabled(selector, disabled) {
        const btn = this.root?.querySelector(selector);
        if (!btn) return;
        btn.disabled = !!disabled;
        btn.title = disabled ? t("mediaPicker.slotsFull") : t("mediaPicker.pickExistingHint");
    },
    async pickExistingRef(isGlobal) {
        const target = isGlobal
            ? (this.timeline.global = this.timeline.global || { refs: [] })
            : this.timeline.segments[this.selectedIndex];
        if (!target) return;
        target.refs = target.refs || [];
        const index = this._nextEmptyMediaSlot(
            target.refs,
            MAX_REFERENCE_IMAGES,
            (r) => !!(r?.imageFile || r?.imageB64),
        );
        if (index < 0) {
            alert(t("mediaPicker.slotsFull"));
            return;
        }
        try {
            const picked = await this.chooseImageInput({
                title: t("mediaPicker.pickReferenceImage"),
            });
            if (!picked?.imageFile) return;
            target.refs = target.refs.filter((r) => Number(r.index ?? r.slot) !== index);
            target.refs.push({ index, imageFile: picked.imageFile, imageB64: "" });
            if (isGlobal) {
                this.timeline.global = target;
                if (this.isR2vCommonEnabled()) {
                    rebaseR2vGroupSlotsForCommon(this);
                    this.renderImageBatchGroups?.();
                }
            }
            this.commit();
            this.renderRefSlots(
                target.refs,
                isGlobal ? this.globalRefsBox : this.segRefsBox,
                isGlobal,
            );
        } catch (err) {
            console.error("[MiniMax H3Director] ref pick failed:", err);
        }
    },
    async pickExistingRefAudio(isGlobal) {
        const target = isGlobal
            ? (this.timeline.global = this.timeline.global || { refs: [], refAudios: [] })
            : this.timeline.segments[this.selectedIndex];
        if (!target) return;
        target.refAudios = target.refAudios || [];
        const index = this._nextEmptyMediaSlot(
            target.refAudios,
            MAX_REFERENCE_AUDIOS,
            (r) => !!(r?.audioFile || r?.fileName),
        );
        if (index < 0) {
            alert(t("mediaPicker.slotsFull"));
            return;
        }
        try {
            const picked = await this.chooseAudioInput({
                title: t("mediaPicker.pickReferenceAudio"),
            });
            if (!picked?.relPath) return;
            if (hasDuplicateReferenceAudio(target.refAudios, picked.relPath, index)) {
                alert(t("ref.audioDuplicate"));
                return;
            }
            target.refAudios = target.refAudios.filter((r) => Number(r.index ?? r.slot) !== index);
            target.refAudios.push({
                index,
                audioFile: picked.relPath,
                fileName: picked.fileName || picked.relPath,
                type: picked.type || "input",
                subfolder: picked.subfolder || "",
            });
            if (this.isR2vCommonEnabled() && isGlobal) {
                rebaseR2vGroupSlotsForCommon(this);
                this.renderImageBatchGroups?.();
            }
            this.commit();
            this.renderRefAudioSlots();
        } catch (err) {
            console.error("[MiniMax H3Director] ref audio pick failed:", err);
            alert(t("upload.refAudioFailed", { err: err?.message || err }));
        }
    },
    async addRefFromFile(file, target, slotIndex = null, isGlobal = null) {
        target.refs = target.refs || [];
        let index = slotIndex;
        if (index == null) {
            index = Array.from({ length: MAX_REFERENCE_IMAGES }, (_, i) => i)
                .find((i) => !target.refs.some((r) => Number(r.index ?? r.slot) === i));
            if (index == null) return;
        }
        try {
            const uploaded = await uploadToInput(file);
            const relPath = videoRelativePath(uploaded);
            target.refs = target.refs.filter((r) => Number(r.index ?? r.slot) !== index);
            target.refs.push({ index, imageFile: relPath, imageB64: "" });
            if (isGlobal) {
                this.timeline.global = target;
                if (this.isR2vCommonEnabled()) {
                    rebaseR2vGroupSlotsForCommon(this);
                    this.renderImageBatchGroups?.();
                }
            }
            this.commit();
        } catch (err) {
            console.error("[MiniMax H3Director] ref upload failed:", err);
        }
    }
};
