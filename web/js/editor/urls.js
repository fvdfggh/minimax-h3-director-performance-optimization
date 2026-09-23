/** URL helpers for the editor, aliased away from core/utils.js names.
 *
 * These are deliberately thin aliases rather than direct use of ``relPath`` /
 * ``viewUrl``: the editor class has methods with local ``const relPath`` /
 * ``const viewUrl`` bindings, so calling those names from inside such a method
 * would either be shadowed by the local variable or (in the module scope) read as a
 * self-reference. Keeping the aliases makes every call site unambiguous.
 *
 * Moved verbatim from minimax_timeline.js, comment included.
 */

import { relPath, viewUrl } from "../core/utils.js";

// Thin aliases over web/js/core/utils.js. Kept as aliases (rather than renamed
// call sites) because this file has local ``const relPath`` / ``const viewUrl``
// variables that would shadow — or self-reference — a straight rename.
export const videoRelativePath = relPath;
export const inputViewUrl = viewUrl;
export const refViewUrl = viewUrl;
