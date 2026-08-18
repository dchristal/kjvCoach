#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
# Free port 8002 if already in use
lsof -ti:8002 | xargs kill -9 2>/dev/null || true
source venv/bin/activate
exec uvicorn app:app --host 0.0.0.0 --port 8002 --reload
