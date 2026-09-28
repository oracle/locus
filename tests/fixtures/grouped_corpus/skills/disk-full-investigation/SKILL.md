---
name: disk-full-investigation
description: Establish blast radius for disk-full errors.
allowed-tools: query_metrics
min-tool-calls: 2
max-tool-calls: 10
required_probes:
  - name: disk_full_count
    match: "node_disk_full_error_count"
    description: The error counts by host.
metadata:
  skill-id: disk-full-investigation
---

# Disk-full investigation

Query the error counts by host, then free bytes per volume.
