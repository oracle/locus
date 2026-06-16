export type Pattern = {
  id: string;
  title: string;
  notebook: number | null;
  summary: string;
  streamable: boolean;
};

export type ProviderType = "openai" | "anthropic" | "oci-session" | "oci-apikey";

export type OciTransport = "auto" | "v1" | "sdk" | "responses";

export type ProviderConfig = {
  provider: ProviderType;
  model: string;
  // Optional secondary models. Empty/undefined means "fall back to model".
  // Same provider + credentials as the primary slot.
  model_b?: string;
  model_c?: string;
  api_key?: string;
  profile?: string;
  region?: string;
  compartment_id?: string;
  oci_transport?: OciTransport;
  // OCI Responses transport only: optional GenAI Project OCID.
  project_ocid?: string;
};

export type DatabaseConfig = {
  dsn: string;
  user: string;
  password: string;
  wallet_location: string;
  wallet_password: string;
};

export type DatabaseTestResult = {
  ok: boolean;
  detail: string;
  dsn?: string;
};

export type RunEvent = {
  kind: string;
  text: string;
  extra?: Record<string, unknown>;
};

export type RunResponse = {
  reply: string;
  events: RunEvent[];
  model?: string;
  provider?: string;
};
