/** Strict-selector helper used across the workbench modules. Throws if
 *  the element isn't in the DOM — surfaces missing-id bugs immediately
 *  during init instead of silently producing null-ref errors later. */
export const $ = <T extends HTMLElement = HTMLElement>(sel: string): T => {
  const el = document.querySelector<T>(sel);
  if (!el) throw new Error(`[wb] missing element: ${sel}`);
  return el;
};
