import { useEffect, useMemo, useState } from "react";
import type { Peer, StreamedEvent } from "./types";
import { fetchCard, invoke, stream } from "./api";

const PEERS: Peer[] = [
  {
    url: "http://127.0.0.1:8001",
    proxy: "/api/research",
    fallbackName: "research-agent",
    status: "checking",
  },
  {
    url: "http://127.0.0.1:8002",
    proxy: "/api/finance",
    fallbackName: "finance-agent",
    status: "checking",
  },
];

const TICKER_RE = /\b[A-Z]{2,5}\b/;
const FINANCE_HINTS = ["buy", "sell", "valuation", "price", "stock", "ticker"];

function suggestSkill(query: string): string {
  if (!query) return "research";
  if (TICKER_RE.test(query)) return "valuation";
  if (FINANCE_HINTS.some((w) => query.toLowerCase().includes(w))) return "valuation";
  return "research";
}

export default function App() {
  const [peers, setPeers] = useState<Peer[]>(PEERS);
  const [selected, setSelected] = useState<string | null>(null);
  const [query, setQuery] = useState("Should I buy TSLA?");
  const [streaming, setStreaming] = useState(false);
  const [busy, setBusy] = useState(false);
  const [events, setEvents] = useState<StreamedEvent[]>([]);
  const [reply, setReply] = useState("");
  const [error, setError] = useState<string | null>(null);

  const suggestedSkill = useMemo(() => suggestSkill(query), [query]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const updated = await Promise.all(
        PEERS.map(async (p) => {
          try {
            const card = await fetchCard(p.proxy);
            return { ...p, card, status: "up" as const };
          } catch {
            return { ...p, status: "down" as const };
          }
        }),
      );
      if (!cancelled) setPeers(updated);
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Auto-pick the first reachable peer that advertises the suggested skill.
  useEffect(() => {
    if (selected) return;
    const match = peers.find(
      (p) => p.status === "up" && p.card?.skills.includes(suggestedSkill),
    );
    if (match) setSelected(match.url);
  }, [peers, suggestedSkill, selected]);

  async function run() {
    const peer = peers.find((p) => p.url === selected);
    if (!peer || !peer.card) return;
    setBusy(true);
    setError(null);
    setEvents([]);
    setReply("");

    if (streaming) {
      await stream(
        peer.proxy,
        query,
        (e) => setEvents((prev) => [...prev, e]),
        (final) => setReply(final),
        (msg) => setError(msg),
      );
    } else {
      try {
        const answer = await invoke(peer.proxy, query);
        setReply(answer);
      } catch (err) {
        setError((err as Error).message);
      }
    }
    setBusy(false);
  }

  return (
    <div className="app">
      <header className="app__header">
        <div className="app__brand">
          <span className="app__brand-mark">locus</span>
          <span className="app__brand-tag">a2a mesh console</span>
        </div>
        <div className="app__header-spacer" />
        <span className="app__header-meta">Oracle Generative AI · Multi-Agent Reasoning Orchestrator SDK</span>
      </header>

      <aside className="app__side">
        <div className="side__title">Workspace</div>
        <div className="side__item side__item--active">
          <span className="side__dot side__dot--up" />
          <span>A2A Console</span>
        </div>
        <div className="side__item">
          <span className="side__dot" />
          <span>Specialists</span>
        </div>
        <div className="side__item">
          <span className="side__dot" />
          <span>Audit log</span>
        </div>
        <div className="side__item">
          <span className="side__dot" />
          <span>Settings</span>
        </div>

        <div className="side__title" style={{ marginTop: "1.5rem" }}>
          Peers
        </div>
        {peers.map((p) => (
          <div key={p.url} className="side__item">
            <span
              className={
                "side__dot " +
                (p.status === "up"
                  ? "side__dot--up"
                  : p.status === "down"
                    ? "side__dot--down"
                    : "")
              }
            />
            <span style={{ fontSize: "0.85rem" }}>
              {p.card?.name ?? p.fallbackName}
            </span>
          </div>
        ))}
      </aside>

      <main className="app__main">
        <div className="page__crumbs">Workflows · Multi-agent · A2A Console</div>
        <h1 className="page__title">A2A mesh console</h1>
        <p className="page__lede">
          Discover Locus agents over HTTP+SSE, route a query by capability tag, and
          watch the remote agent's typed event stream — the same shape your
          back-end emits for a single in-process agent.
        </p>

        <div className="split">
          <section>
            <div className="card">
              <div className="card__header">
                <div>
                  <h2 className="card__title">Peers</h2>
                  <div className="card__sub">
                    Auto-discovered via <code>GET /agent-card</code>
                  </div>
                </div>
                <span
                  className={
                    "pill " +
                    (peers.every((p) => p.status === "up")
                      ? "pill--up"
                      : peers.some((p) => p.status === "up")
                        ? "pill--busy"
                        : "pill--down")
                  }
                >
                  <span className="pill__dot" />
                  {peers.filter((p) => p.status === "up").length}/{peers.length} reachable
                </span>
              </div>

              {peers.map((p) => {
                const card = p.card;
                const isSelected = selected === p.url;
                const isFinance = (card?.name ?? p.fallbackName).includes("finance");
                return (
                  <div
                    key={p.url}
                    className={"peer " + (isSelected ? "peer--selected" : "")}
                    onClick={() => p.status === "up" && setSelected(p.url)}
                    role="button"
                    tabIndex={0}
                  >
                    <div
                      className={"peer__icon " + (isFinance ? "peer__icon--finance" : "")}
                    >
                      {(card?.name ?? p.fallbackName).slice(0, 2).toUpperCase()}
                    </div>
                    <div style={{ flex: 1 }}>
                      <div className="peer__name">{card?.name ?? p.fallbackName}</div>
                      <div className="peer__url">{p.url}</div>
                      {card && (
                        <div className="peer__skills">
                          {card.skills.map((s) => (
                            <span
                              key={s}
                              className={
                                "skill " + (s === suggestedSkill ? "skill--match" : "")
                              }
                            >
                              {s}
                            </span>
                          ))}
                        </div>
                      )}
                      {p.status === "down" && (
                        <div
                          style={{
                            fontSize: "0.75rem",
                            color: "var(--or-red-deep)",
                            marginTop: "0.3rem",
                          }}
                        >
                          unreachable — is the service running?
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </section>

          <section>
            <div className="card">
              <div className="card__header">
                <div>
                  <h2 className="card__title">Send a query</h2>
                  <div className="card__sub">
                    Routed to <strong>{selected ?? "no peer"}</strong>{" "}
                    <span style={{ color: "var(--or-text-mute)" }}>
                      · suggested skill <em>{suggestedSkill}</em>
                    </span>
                  </div>
                </div>
              </div>

              <div className="field" style={{ marginBottom: "0.7rem" }}>
                <label className="field__label">Prompt</label>
                <textarea
                  className="textarea"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Ask the mesh anything…"
                />
              </div>

              <div className="form-row">
                <label className="checkbox">
                  <input
                    type="checkbox"
                    checked={streaming}
                    onChange={(e) => setStreaming(e.target.checked)}
                  />
                  Stream events (SSE)
                </label>
                <div style={{ flex: 1 }} />
                <button
                  className="btn"
                  onClick={() => {
                    setEvents([]);
                    setReply("");
                    setError(null);
                  }}
                  disabled={busy}
                >
                  Clear
                </button>
                <button
                  className="btn btn--primary"
                  onClick={run}
                  disabled={busy || !selected}
                >
                  {busy ? "Running…" : "Send"}
                </button>
              </div>
            </div>

            <div className="card reply">
              <div className="card__header">
                <h2 className="card__title">Response</h2>
                {streaming && events.length > 0 && (
                  <span className="pill pill--busy">
                    <span className="pill__dot" />
                    {events.length} events
                  </span>
                )}
              </div>

              {error && (
                <div className="event">
                  <span className="event__kind event__kind--error">Error</span>
                  <span className="event__body">{error}</span>
                </div>
              )}

              {!error && events.length === 0 && !reply && !busy && (
                <div className="reply__empty">
                  No request yet. Type a prompt and hit <strong>Send</strong>.
                </div>
              )}

              {events.map((e) => {
                const css =
                  e.kind === "TerminateEvent"
                    ? "event__kind--terminate"
                    : e.kind.startsWith("Tool")
                      ? "event__kind--tool"
                      : "";
                return (
                  <div key={e.id} className="event">
                    <span className={"event__kind " + css}>{e.kind.replace("Event", "")}</span>
                    <span className="event__body">{e.text}</span>
                  </div>
                );
              })}

              {reply && <div className="reply__final">{reply}</div>}
            </div>
          </section>
        </div>
      </main>

      <footer className="app__footer">
        Built on Oracle Generative AI · locus
      </footer>
    </div>
  );
}
