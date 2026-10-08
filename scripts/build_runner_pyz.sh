#!/usr/bin/env bash
# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0
#
# Build tulip-runner.pyz: the box runner (`python -m tulip.runner`) as one file
# that runs on any CPython of the same version and platform it was built with.
#
#   scripts/build_runner_pyz.sh <tulip_agents wheel> <output .pyz>
#
# The wheel is installed with the `openai` extra into a fresh virtualenv, and
# what that resolved to is frozen: the .pyz carries exactly those versions,
# listed in <output>.requirements.txt beside it. pydantic-core is a compiled
# extension, so the .pyz is tied to the interpreter that built it (the release
# builds it on CPython 3.12, linux x86_64 — the runner image's Python). shiv
# unpacks it once into $SHIV_ROOT (~/.shiv) on first run.
set -euo pipefail

wheel=${1:?usage: build_runner_pyz.sh <wheel> <output.pyz>}
out=${2:?usage: build_runner_pyz.sh <wheel> <output.pyz>}
python=${PYTHON:-python3}
shiv_version=${SHIV_VERSION:-1.0.8}

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

"$python" -m venv "$work/venv"
"$work/venv/bin/pip" install --quiet --disable-pip-version-check "${wheel}[openai]"
"$work/venv/bin/pip" freeze --disable-pip-version-check --exclude tulip-agents \
  > "$work/requirements.txt"

"$python" -m venv "$work/build"
"$work/build/bin/pip" install --quiet --disable-pip-version-check "shiv==${shiv_version}"

mkdir -p "$(dirname "$out")"
"$work/build/bin/shiv" \
  --console-script tulip-runner \
  --python "/usr/bin/env python3" \
  --compressed \
  --reproducible \
  --output-file "$out" \
  -r "$work/requirements.txt" \
  "$wheel"
cp "$work/requirements.txt" "${out}.requirements.txt"
echo "built $out ($(du -h "$out" | cut -f1)) for $("$python" -c 'import sys; print(sys.version.split()[0])')"
