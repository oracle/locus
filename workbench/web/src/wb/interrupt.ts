/** Inline interrupt input form. The bootstrap monkey-patches
 *  `locus.core.interrupt` to emit an `InterruptEvent` __LE__ line and
 *  block on stdin; the workbench shows this form, the user types,
 *  the answer is POSTed to /api/notebooks/runs/<id>/respond which writes
 *  it to the subprocess's stdin. */
import { respondToInterrupt } from "../api";
import { appendInterruptForm } from "./output";

export function promptInterruptResponse(runId: string, payload: unknown): void {
  const wrap = document.createElement("div");
  wrap.className = "wb-interrupt";
  wrap.dataset.testid = "wb-interrupt";
  const q = typeof payload === "object" && payload !== null
    ? (payload as Record<string, unknown>)
    : { question: String(payload) };
  const question = (q.question as string) ?? (q.message as string) ?? String(payload);
  const options = Array.isArray(q.options) ? (q.options as string[]) : null;
  wrap.innerHTML = `
    <div class="wb-interrupt__head">
      <span class="event__kind event__kind--query">Interrupt</span>
      <strong></strong>
    </div>
    <div class="wb-interrupt__body">
      ${
        options
          ? options
              .map((o) => `<button class="btn wb-interrupt__opt" data-val="${o.replace(/"/g, "&quot;")}">${o}</button>`)
              .join("")
          : ""
      }
      <input type="text" class="input wb-interrupt__text" placeholder="type a response, press Enter..." />
      <button class="btn btn--primary wb-interrupt__send">Send</button>
    </div>
  `;
  (wrap.querySelector(".wb-interrupt__head strong") as HTMLElement).textContent = question;
  appendInterruptForm(wrap);
  const text = wrap.querySelector<HTMLInputElement>(".wb-interrupt__text")!;
  const send = wrap.querySelector<HTMLButtonElement>(".wb-interrupt__send")!;
  text.focus();

  const submit = async (value: string) => {
    console.info("[wb/interrupt] respond", { runId, value });
    wrap.classList.add("wb-interrupt--sent");
    text.disabled = true;
    send.disabled = true;
    wrap.querySelectorAll<HTMLButtonElement>(".wb-interrupt__opt").forEach((b) => (b.disabled = true));
    try {
      await respondToInterrupt(runId, value);
      const ack = document.createElement("div");
      ack.className = "wb-interrupt__ack";
      ack.textContent = `→ replied ${JSON.stringify(value)}`;
      wrap.appendChild(ack);
    } catch (err) {
      console.error("[wb/interrupt] respond failed", err);
      const ack = document.createElement("div");
      ack.className = "wb-interrupt__err";
      ack.textContent = `failed: ${(err as Error).message}`;
      wrap.appendChild(ack);
    }
  };

  text.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && text.value.trim()) void submit(text.value.trim());
  });
  send.addEventListener("click", () => {
    if (text.value.trim()) void submit(text.value.trim());
  });
  wrap.querySelectorAll<HTMLButtonElement>(".wb-interrupt__opt").forEach((btn) => {
    btn.addEventListener("click", () => void submit(btn.dataset.val ?? ""));
  });
}
