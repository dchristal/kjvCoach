#!/bin/bash
# tidy:ui http://127.0.0.1:8002/
# tidy:nogpu  — UI/proxy only; uses whatever model llama-server has on :8080.
set -euo pipefail
cd "$(dirname "$0")"
# Free port 8002 if already in use
lsof -ti:8002 | xargs kill -9 2>/dev/null || true
source venv/bin/activate

uvicorn app:app --host 0.0.0.0 --port 8002 --reload &
PID=$!

echo -n "Waiting for kjvCoach"
for i in {1..30}; do
  if ! kill -0 "$PID" >/dev/null 2>&1; then
    echo " ❌ kjvCoach exited early."
    exit 1
  fi
  if curl -s http://localhost:8002/health >/dev/null 2>&1; then
    echo " ✅ Ready!"
    break
  fi
  echo -n "."
  sleep 1
done

wait "$PID"
