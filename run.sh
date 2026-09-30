#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

if [[ $# -eq 0 ]]; then
    exec "$PYTHON" "$SCRIPT_DIR/main.py" --help
fi

exec "$PYTHON" "$SCRIPT_DIR/main.py" "$@"
