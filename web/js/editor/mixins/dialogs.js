/** dialogs mixin for MiniMaxH3DirectorOptEditor.
 *
 * Extracted verbatim from editor.js; methods are unchanged apart from the
 * trailing comma an object literal needs.
 */

import { relPath } from "../../core/utils.js";
import { MiniMaxH3DirectorOptEditor } from "../editor.js";
import { inputViewUrl } from "../urls.js";
import { t } from "../../minimax_i18n.js";
export const dialogsMixin = {
    _closeBdModal() {
        if (this._modalKeyHandler) {
            window.removeEventListener("keydown", this._modalKeyHandler, true);
            this._modalKeyHandler = null;
        }
        if (this._modalEl) {
            this._modalEl.remove();
            this._modalEl = null;
        }
    },
    showBdMessage(title, message) {
        return this.showBdDialog({ title, message, confirmText: t("dialog.confirm"), cancelText: null });
    },
    showBdDialog(opts = {}) {
        const { title, message, items } = opts;
        const confirmText = opts.confirmText ?? t("dialog.confirm");
        const cancelText = Object.prototype.hasOwnProperty.call(opts, "cancelText")
            ? opts.cancelText
            : t("dialog.cancel");
        return new Promise((resolve) => {
            this._closeBdModal();

            const overlay = document.createElement("div");
            overlay.className = "bd-modal-overlay";
            const panel = document.createElement("div");
            panel.className = "bd-modal";
            panel.innerHTML = `
                <div class="bd-modal-title"></div>
                <div class="bd-modal-body hidden"></div>
                <div class="bd-modal-list hidden"></div>
                <div class="bd-modal-actions"></div>`;

            panel.querySelector(".bd-modal-title").textContent = title || "";

            const bodyEl = panel.querySelector(".bd-modal-body");
            const listEl = panel.querySelector(".bd-modal-list");
            const actionsEl = panel.querySelector(".bd-modal-actions");

            let selectedValue = items?.length ? items[0].value : null;

            const finish = (val) => {
                this._closeBdModal();
                resolve(val);
            };

            if (message) {
                bodyEl.textContent = message;
                bodyEl.classList.remove("hidden");
            }

            if (items?.length) {
                listEl.classList.remove("hidden");
                for (const item of items) {
                    const row = document.createElement("div");
                    row.className = "bd-modal-item";
                    row.textContent = item.label ?? item.value;
                    row.title = item.label ?? item.value;
                    row.dataset.value = item.value;
                    if (item.value === selectedValue) row.classList.add("selected");
                    row.onclick = () => {
                        selectedValue = item.value;
                        for (const el of listEl.querySelectorAll(".bd-modal-item")) {
                            el.classList.toggle("selected", el === row);
                        }
                    };
                    row.ondblclick = () => finish(item.value);
                    listEl.appendChild(row);
                }
            }

            if (cancelText) {
                const cancelBtn = document.createElement("button");
                cancelBtn.type = "button";
                cancelBtn.className = "bd-btn";
                cancelBtn.textContent = cancelText;
                cancelBtn.onclick = () => finish(null);
                actionsEl.appendChild(cancelBtn);
            }

            const okBtn = document.createElement("button");
            okBtn.type = "button";
            okBtn.className = "bd-btn bd-btn-primary";
            okBtn.textContent = confirmText;
            okBtn.onclick = () => finish(items?.length ? selectedValue : true);
            actionsEl.appendChild(okBtn);

            overlay.onclick = (e) => {
                if (e.target === overlay && cancelText) finish(null);
            };
            panel.onclick = (e) => e.stopPropagation();

            this._modalKeyHandler = (e) => {
                if (e.key === "Escape") {
                    e.preventDefault();
                    e.stopPropagation();
                    finish(cancelText ? null : true);
                } else if (e.key === "Enter" && items?.length) {
                    e.preventDefault();
                    finish(selectedValue);
                }
            };
            window.addEventListener("keydown", this._modalKeyHandler, true);

            overlay.appendChild(panel);
            this.root.appendChild(overlay);
            this._modalEl = overlay;
            okBtn.focus();
        });
    },
    pickLocalFile(accept = "") {
        return new Promise((resolve) => {
            const input = document.createElement("input");
            input.type = "file";
            if (accept) input.accept = accept;
            input.style.cssText = "position:fixed;left:-9999px;top:0;opacity:0;pointer-events:none";
            const cleanup = () => input.remove();
            input.onchange = () => {
                const file = input.files?.[0] || null;
                cleanup();
                resolve(file);
            };
            input.addEventListener("cancel", () => {
                cleanup();
                resolve(null);
            }, { once: true });
            document.body.appendChild(input);
            input.click();
        });
    },
    async listInputMedia(kind, { includeCache = false } = {}) {
        // ``includeCache`` pulls in Director's own rendered clips from output/.
        // Opt-in per call so image/audio pickers keep their current contents —
        // this is only ever useful for video, and only when the user is hunting
        // for something they generated earlier.
        const params = new URLSearchParams({ kind });
        if (includeCache) params.set("includeCache", "1");
        const resp = await api.fetchApi(`/minimax/director_opt/list_input_media?${params.toString()}`);
        if (!resp.ok) {
            const text = (await resp.text()).trim();
            if (resp.status === 404) throw new Error(t("mediaPicker.needRestart"));
            throw new Error(text || `HTTP ${resp.status}`);
        }
        const data = await resp.json();
        return Array.isArray(data?.items) ? data.items : [];
    },
    probeInputImageDimensions(relPath, type = "input") {
        return new Promise((resolve) => {
            const img = new Image();
            img.onload = () => resolve({
                width: img.naturalWidth || img.width || 0,
                height: img.naturalHeight || img.height || 0,
            });
            img.onerror = () => resolve({ width: 0, height: 0 });
            img.src = inputViewUrl(relPath, type || "input");
        });
    }
};
