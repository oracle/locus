/** CodeMirror editor module — owns the editor lifecycle and exposes a
 *  small `setSource / getSource` API that the rest of the workbench
 *  uses. Also publishes the same API on `window.__wb` so playwright /
 *  devtools can drive the editor from outside. */
import { EditorState } from "@codemirror/state";
import {
  EditorView,
  highlightActiveLine,
  keymap,
  lineNumbers,
} from "@codemirror/view";
import { defaultKeymap, history, historyKeymap } from "@codemirror/commands";
import { python } from "@codemirror/lang-python";
import { oneDark } from "@codemirror/theme-one-dark";
import { $ } from "./dom";

let view: EditorView | null = null;

export function ensureEditor(initial = ""): EditorView {
  if (view) return view;
  const mount = $<HTMLElement>("#wb-editor");
  const state = EditorState.create({
    doc: initial,
    extensions: [
      lineNumbers(),
      highlightActiveLine(),
      history(),
      python(),
      oneDark,
      keymap.of([...defaultKeymap, ...historyKeymap]),
      // Match the output panel's surface so editor + output share the
      // same dark canvas.
      EditorView.theme({
        "&": { fontSize: "13px", height: "100%", backgroundColor: "#1b1a18" },
        ".cm-scroller": { fontFamily: "JetBrains Mono, ui-monospace, Menlo, monospace" },
        ".cm-gutters": { backgroundColor: "#1b1a18", borderRight: "1px solid #2a2823" },
        ".cm-content": { caretColor: "#f0cc71" },
        ".cm-cursor, .cm-dropCursor": { borderLeftColor: "#f0cc71" },
        ".cm-activeLine": { backgroundColor: "rgba(240, 204, 113, 0.06)" },
        ".cm-activeLineGutter": { backgroundColor: "rgba(240, 204, 113, 0.06)" },
      }),
    ],
  });
  view = new EditorView({ state, parent: mount });
  // Hook for playwright / programmatic edits.
  (window as unknown as Record<string, unknown>).__wb = {
    setSource: setEditorContent,
    getSource: getEditorContent,
  };
  return view;
}

export function setEditorContent(text: string): void {
  const ed = ensureEditor(text);
  ed.dispatch({
    changes: { from: 0, to: ed.state.doc.length, insert: text },
    // Park cursor at the very top + scroll line 1 into view so picking
    // a notebook drops you on its first line, not wherever the cursor
    // was in the previous source.
    selection: { anchor: 0 },
    effects: EditorView.scrollIntoView(0, { y: "start" }),
  });
}

export function getEditorContent(): string {
  return view?.state.doc.toString() ?? "";
}

export function requestEditorMeasure(): void {
  view?.requestMeasure();
}
