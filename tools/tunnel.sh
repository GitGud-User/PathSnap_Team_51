#!/usr/bin/env bash
# Put the local PathSnap API (and the UI it serves) on a temporary public HTTPS link:
#   https://<random>.trycloudflare.com
#
# Usage:
#   .venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000   # terminal 1
#   tools/tunnel.sh                                                            # terminal 2
#
# Everything (models, data, logs) stays on this machine; Cloudflare only relays requests.
# Stop with Ctrl+C. Every run gives a NEW link, so re-share it after a restart.
set -euo pipefail

PORT=${PORT:-8000}
API=${API:-http://127.0.0.1:$PORT}

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "cloudflared is not installed. Install it with:" >&2
  echo "  brew install cloudflared          # macOS" >&2
  echo "  winget install --id Cloudflare.cloudflared -e   # Windows" >&2
  exit 1
fi

if ! curl -fsS --max-time 5 "$API/api/health" >/dev/null; then
  echo "No API answering at $API" >&2
  echo "Start it first: .venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port $PORT" >&2
  exit 1
fi

mkdir -p logs
echo "Tunnelling $API -> public link below. Share the https://<random>.trycloudflare.com URL."
echo "Log: logs/tunnel.log"
echo

cloudflared tunnel --url "$API" --no-autoupdate 2>&1 | tee -a logs/tunnel.log
