export type AgentCard = {
  name: string;
  description: string;
  skills: string[]; // skill ids, normalised from the card's AgentSkill objects
  url: string; // synthesized client-side: where we fetched it from
  protocolVersion?: string;
  interfaces: string[]; // e.g. "JSONRPC v1.0", from supportedInterfaces
};

export type AgentStatus = "up" | "down" | "checking";

export type Peer = {
  url: string;
  proxy: string; // path the Vite proxy rewrites
  fallbackName: string;
  card?: AgentCard;
  status: AgentStatus;
};

// A2A v1.0 SendStreamingMessage frames: the task, then status and artifact updates.
export type EventKind = "Task" | "StatusUpdate" | "ArtifactUpdate" | "Message" | "Error" | "Other";

export type StreamedEvent = {
  id: string;
  kind: EventKind;
  raw: Record<string, unknown>;
  text?: string;
};
