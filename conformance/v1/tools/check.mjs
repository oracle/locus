// Cross-language check of the conformance fixtures with nothing but Node's
// standard library: recomputes every canonical-json digest and verifies the
// Ed25519 signature on every signed audit record with the committed public
// keys. Prints one JSON line with the counts; exits non-zero on any mismatch.
//
//   node conformance/v1/tools/check.mjs [path/to/conformance/v1]
import { createHash, createPublicKey, verify } from "node:crypto";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = process.argv[2] ?? dirname(dirname(fileURLToPath(import.meta.url)));
const read = (...parts) => readFileSync(join(root, ...parts), "utf8");
const sha256 = (data) => createHash("sha256").update(data).digest("hex");

let vectors = 0;
for (const vector of JSON.parse(read("canonical-json", "vectors.json"))) {
  if (sha256(Buffer.from(vector.canonical, "utf8")) !== vector.sha256) {
    throw new Error(`digest mismatch: ${vector.name}`);
  }
  vectors += 1;
}

const keys = {};
let signatures = 0;
for (const trail of ["signed", "rotated"]) {
  for (const line of read("audit", trail, "trail.jsonl").split("\n").filter(Boolean)) {
    const record = JSON.parse(line);
    keys[record.key_id] ??= createPublicKey(read("audit", "keys", `${record.key_id}.pub.pem`));
    const signature = Buffer.from(record.signature, "base64");
    if (!verify(null, Buffer.from(record.hash, "ascii"), keys[record.key_id], signature)) {
      throw new Error(`signature does not verify: ${trail} record ${record.seq}`);
    }
    signatures += 1;
  }
}

// The default key id is the first 16 hex characters of SHA-256 over the raw
// 32-byte public key (the last 32 bytes of the SubjectPublicKeyInfo DER).
const derived = Object.fromEntries(
  Object.entries(keys).map(([id, key]) => [
    id,
    sha256(key.export({ type: "spki", format: "der" }).subarray(-32)).slice(0, 16),
  ]),
);
console.log(JSON.stringify({ vectors, signatures, derived_key_ids: derived }));
