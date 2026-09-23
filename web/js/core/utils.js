/** Shared frontend helpers for the Director Opt UI.
 *
 * ``clamp`` / ``uid`` / ``relPath`` / ``viewUrl`` used to be re-implemented in
 * every feature module (timeline, image batch, fl2v, prompt mentions) under
 * slightly different names. One copy lives here so they cannot drift apart.
 */

import { api } from "../../../scripts/api.js";

export function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
}

export function uid() {
    return Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
}

/** ``subfolder/name`` for a ComfyUI upload response (relative-path form). */
export function relPath(upload) {
    const name = upload.name || upload.filename;
    const sub = (upload.subfolder || "").replace(/\\/g, "/").replace(/\/$/, "");
    return sub ? `${sub}/${name}` : name;
}

/** ``/api/view`` URL for a relative path.
 *
 * Must echo the record's own ``type``: ``/api/view`` resolves the file under
 * whichever directory that names. Hardcoding ``input`` made any non-input media
 * 404 in the preview even though the backend could read it — Director's own
 * renders live in ``output/``. ComfyUI only serves input/output/temp.
 */
export function viewUrl(relativePath, type = "input") {
    const norm = String(relativePath || "").replace(/\\/g, "/");
    const slash = norm.lastIndexOf("/");
    const filename = slash >= 0 ? norm.slice(slash + 1) : norm;
    const subfolder = slash >= 0 ? norm.slice(0, slash) : "";
    const dirType = ["input", "output", "temp"].includes(String(type || "").toLowerCase())
        ? String(type).toLowerCase()
        : "input";
    const params = new URLSearchParams({ filename, type: dirType });
    if (subfolder) params.set("subfolder", subfolder);
    return api.apiURL(`/view?${params.toString()}`);
}
