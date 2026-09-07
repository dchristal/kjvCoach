#!/bin/bash
# tidy:ui http://127.0.0.1:8002/
# tidy:nogpu  — UI/proxy only; uses whatever model llama-server has on :8080.
set -euo pipefail
cd "$(dirname "$0")"

PORT=8002
LOCK="${TMPDIR:-/tmp}/kjvcoach-${PORT}.lock"

source venv/bin/activate

free_port() {
  local port=$1
  local pids
  pids=$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)
  if [[ -n "${pids}" ]]; then
    # shellcheck disable=SC2086
    kill -TERM ${pids} 2>/dev/null || true
    sleep 0.4
    pids=$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)
    if [[ -n "${pids}" ]]; then
      # shellcheck disable=SC2086
      kill -KILL ${pids} 2>/dev/null || true
    fi
  fi
  local i
  for i in $(seq 1 30); do
    pids=$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)
    [[ -z "${pids}" ]] && return 0
    sleep 0.1
  done
  echo " ❌ port ${port} still busy (${pids})" >&2
  return 1
}

# Serialize kill+bind+health so overlapping tidy Restarts cannot race.
exec 9>"$LOCK"
flock 9

free_port "$PORT"

# Loopback only — Tailscale Serve terminates HTTPS on :8002 (0.0.0.0 conflicts).
uvicorn app:app --host 127.0.0.1 --port "$PORT" --reload &
PID=$!

echo -n "Waiting for kjvCoach"
ready=0
for i in {1..30}; do
  if ! kill -0 "$PID" >/dev/null 2>&1; then
    echo " ❌ kjvCoach exited early."
    exit 1
  fi
  if curl -sf --max-time 1 "http://127.0.0.1:${PORT}/health" >/dev/null; then
    echo " ✅ Ready!"
    ready=1
    break
  fi
  echo -n "."
  sleep 1
done

# Release before wait so a later Restart can take the lock and replace us.
flock -u 9

if [[ "$ready" -ne 1 ]]; then
  echo " ❌ kjvCoach health check timed out."
  kill -TERM "$PID" 2>/dev/null || true
  exit 1
fi

wait "$PID"
