### Adversary Tradecraft

- **Malleable C2**: Cobalt Strike beacons use malleable C2 profiles to blend in with normal web traffic (intel-01).
- **Living off the land**: LOLBins such as `certutil` are abused to download payloads (intel-03).
- **Fast-flux DNS**: Rotating the IPs behind a domain rapidly frustrates takedown efforts (intel-06).
- **Lateral movement**: Movement over SMB and remote service creation maps to ATT&CK technique T1021 (intel-10).

### Vulnerability Landscape

- **Log4Shell (CVE-2021-44228)**: Remote code execution via JNDI lookups in Log4j (cve-03).
- **Deserialization**: Deserializing untrusted data can lead to remote code execution, CWE-502 (cve-09).
- **Real-world risk**: Patching cadence and asset exposure drive risk more than CVSS alone (cve-07).
- **Default credentials**: Default credentials on internet-facing services are a recurring source of compromise (cve-10).
