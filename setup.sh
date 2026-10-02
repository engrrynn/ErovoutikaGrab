#!/usr/bin/env bash
# ErovoutikaGrab Setup Shortcut
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$(id -u)" -ne 0 ]; then
    echo "[!] Elevated privileges required. Re-running with sudo..."
    exec sudo "$SCRIPT_DIR/install.sh" "$@"
else
    exec "$SCRIPT_DIR/install.sh" "$@"
fi
