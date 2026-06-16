export type AgentCard = {
  name: string;
  description: string;
  skills: string[];
  url: string; // synthesized client-side: where we fetched it from
};

export type AgentStatus = "up" | "down" | "checking";

export type Peer = {
  url: string;
  proxy: string; // path the Vite proxy rewrites
  fallbackName: string;
  card?: AgentCard;
  status: AgentStatus;
};

export type EventKind =
  | "ToolStartEvent"
  | "ToolEndEvent"
  | "ThinkEvent"
  | "ModelChunkEvent"
  | "TerminateEvent"
  | "AgentStartEvent"
  | "ErrorEvent"
  | "Other";

export type StreamedEvent = {
  id: string;
  kind: EventKind;
  raw: Record<string, unknown>;
  text?: string;
};
