#!/usr/bin/env bash
# Run the OpenEngine servicer CPU tests without importing the real `vllm`
# package (which pulls torch/CUDA). pytest's rootdir collection would walk up
# the `vllm/` parent __init__ chain, so we copy the test files to a neutral
# temp dir and point the bootstrap at the real source via OE_DIR_OVERRIDE.
#
# Usage: ./run_cpu_tests.sh [/path/to/python]
#   default python: ../../../../.devvenv/bin/python (repo dev venv)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OE_DIR="$(cd "$HERE/.." && pwd)"
PY="${1:-$OE_DIR/../../../../.devvenv/bin/python}"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cp "$HERE"/_bootstrap.py "$HERE"/fakes.py "$HERE"/test_servicer.py "$TMP"/

cd "$TMP"
OE_DIR_OVERRIDE="$OE_DIR" "$PY" -m pytest test_servicer.py -q
