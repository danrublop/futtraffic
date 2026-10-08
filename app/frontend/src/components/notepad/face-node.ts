import { Node, mergeAttributes } from "@tiptap/core";

/** A PLAYER'S FACE, beside his name.
 *
 *  Every panel in a scouting report is headed by one, and it is not decoration:
 *  a coach reading a squad he does not know yet learns the faces off these
 *  pages before he learns the numbers. So the notepad has the block, and the
 *  club has somewhere to put the photographs.
 *
 *  UNTIL SOMEBODY UPLOADS ONE, IT IS A SILHOUETTE. Not a blank, not a hidden
 *  block: the page has the same shape whether the photographs have arrived or
 *  not, so a report written today does not get re-laid-out the day they do.
 *
 *  The photo is BAKED IN as a data URI at insert, like every other figure in
 *  this notepad. The document is read offline - printed, mailed, opened from a
 *  folder - and a note whose faces are `<img src="/api/...">` is a note full of
 *  broken images the moment it leaves the machine that wrote it. `src` is
 *  narrowed to a data URI or the API's own path and nothing else, because this
 *  string is written into markup.
 */

/** A data URI for one of the three image types a browser will certainly draw,
 *  or the API path the app itself serves, or nothing. Anything else - a remote
 *  URL, a `javascript:`, an SVG data URI that can carry script - is dropped. */
export function safeSrc(value: string | null | undefined): string {
  const v = String(value ?? "").trim();
  if (/^data:image\/(png|jpeg|jpg|webp);base64,[A-Za-z0-9+/=\s]+$/.test(v)) return v;
  if (/^\/api\/players\/[A-Za-z0-9_-]{1,64}\/photo$/.test(v)) return v;
  return "";
}

const esc = (s: string) =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

/** Head and shoulders, drawn rather than shipped as a file: it is two shapes,
 *  it inherits the document's ink, and it survives being mailed. */
export const SILHOUETTE =
  `<svg viewBox="0 0 40 40" width="100%" height="100%" aria-hidden="true" ` +
  `style="display:block">` +
  `<circle cx="20" cy="15.5" r="7.6" fill="currentColor" fill-opacity="0.5"/>` +
  `<path d="M4.5 40a15.5 15.5 0 0 1 31 0z" fill="currentColor" fill-opacity="0.5"/>` +
  `</svg>`;

export const FaceNode = Node.create({
  name: "face",
  group: "block",
  atom: true,
  draggable: true,
  selectable: true,

  addAttributes() {
    const attr = (name: string, fallback = "") => ({
      default: fallback,
      parseHTML: (el: HTMLElement) => el.getAttribute(`data-${name}`) ?? fallback,
      renderHTML: (attrs: Record<string, unknown>) =>
        attrs[name] ? { [`data-${name}`]: String(attrs[name]) } : {},
    });
    return {
      /** The player this is a picture of. The join to the roster, so a photo
       *  uploaded later can be found and baked in. */
      player: attr("player"),
      name: attr("name"),
      /** The line under the name: age, height, foot, position. */
      sub: attr("sub"),
      src: {
        default: "",
        parseHTML: (el: HTMLElement) => safeSrc(el.getAttribute("data-src")),
        renderHTML: (attrs: Record<string, unknown>) => {
          const src = safeSrc(String(attrs.src ?? ""));
          return src ? { "data-src": src } : {};
        },
      },
    };
  },

  parseHTML() {
    return [{ tag: "div[data-face]" }];
  },

  renderHTML({ HTMLAttributes, node }) {
    const el = document.createElement("div");
    for (const [k, v] of Object.entries(
      mergeAttributes(HTMLAttributes, { class: "ft-face", "data-face": "1" }),
    )) {
      if (v != null) el.setAttribute(k, String(v));
    }
    const name = String(node.attrs.name ?? "");
    const sub = String(node.attrs.sub ?? "");
    const src = safeSrc(String(node.attrs.src ?? ""));
    el.innerHTML =
      `<span class="ft-face__pic">` +
      (src ? `<img src="${esc(src)}" alt="${esc(name)}"/>` : SILHOUETTE) +
      `</span>` +
      (name || sub
        ? `<span class="ft-face__who">` +
          (name ? `<span class="ft-face__name">${esc(name)}</span>` : "") +
          (sub ? `<span class="ft-face__sub">${esc(sub)}</span>` : "") +
          `</span>`
        : "");
    return el;
  },
});
