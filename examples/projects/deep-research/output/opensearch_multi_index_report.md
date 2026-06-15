### Adversary Tradecraft (Threat-Intel Index)

- **Malleable C2**: Cobalt Strike beacons use malleable C2 profiles to blend in with normal web traffic. (intel-01)
- **PowerShell execution**: Adversary use of PowerShell for execution maps to ATT&CK technique T1059.001. (intel-02)
- **Credential dumping**: Dumping credentials from LSASS memory maps to ATT&CK technique T1003.001. (intel-05)
- **Beaconing periodicity**: C2 beaconing shows a regular periodicity that stands out from human-driven sessions. (intel-07)

### Notable Vulnerabilities (CVE Index)

- **Log4Shell (CVE-2021-44228)**: Remote code execution via JNDI lookups in Log4j. (cve-03)
- **Severity bands**: CVSS v3.1 scores 9.0-10.0 are classified as Critical. (cve-01)
- **Exploit likelihood**: The EPSS score estimates the probability a vulnerability is exploited in the wild. (cve-06)
- **Exposure over score**: Patching cadence and asset exposure drive real-world risk more than CVSS alone. (cve-07)
