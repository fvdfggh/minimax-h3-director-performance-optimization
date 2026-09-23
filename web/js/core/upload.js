/** Uploads into ComfyUI's input directory (direct, or sliced into chunks).
 *
 * The "POST to ``/upload/image``; if the file is too big, slice it and post one
 * part per chunk until the server answers with a name" flow was re-implemented
 * in the timeline, image-batch and pack modules, each with its own copy of the
 * 8 MiB / 95 MiB constants. It lives here now.
 */

import { api } from "../../../scripts/api.js";
import { fileForComfyUpload, safeUploadFilename } from "../minimax_gen_timeline.js";
import { t } from "../minimax_i18n.js";

/** Chunk size for the resumable upload endpoint. */
export const CHUNK_SIZE = 8 * 1024 * 1024;
/** Files at or below this go straight to ``/upload/image``. */
export const UPLOAD_SOFT_LIMIT = 95 * 1024 * 1024;

/** POST a file to ComfyUI's own ``/upload/image`` endpoint. */
export async function uploadToInput(file) {
    const uploadFile = fileForComfyUpload(file);
    const body = new FormData();
    body.append("image", uploadFile, uploadFile.name);
    body.append("type", "input");
    body.append("overwrite", "false");
    const resp = await api.fetchApi("/upload/image", { method: "POST", body });
    if (!resp.ok) {
        const text = await resp.text();
        throw new Error(text || `Upload failed (${resp.status})`);
    }
    return resp.json();
}

/** Slice ``file`` and POST the parts to ``endpoint`` until it returns a name. */
export async function uploadChunked(file, {
    endpoint = "/minimax/director_opt/upload_chunk",
    filename = safeUploadFilename(file?.name, file?.type),
    chunkSize = CHUNK_SIZE,
    onProgress,
} = {}) {
    const uploadId = crypto.randomUUID();
    const totalChunks = Math.ceil(file.size / chunkSize);
    for (let i = 0; i < totalChunks; i++) {
        const start = i * chunkSize;
        const end = Math.min(start + chunkSize, file.size);
        const body = new FormData();
        body.append("upload_id", uploadId);
        body.append("chunk_index", String(i));
        body.append("total_chunks", String(totalChunks));
        body.append("filename", filename);
        body.append("chunk", file.slice(start, end), `${filename}.part`);
        const resp = await api.fetchApi(endpoint, { method: "POST", body });
        if (!resp.ok) {
            const text = await resp.text();
            throw new Error(text || t("upload.chunkFailed", { status: resp.status }));
        }
        onProgress?.((i + 1) / totalChunks, i + 1, totalChunks);
        const data = await resp.json();
        if (data.name) return data;
    }
    throw new Error(t("upload.chunkIncomplete"));
}
