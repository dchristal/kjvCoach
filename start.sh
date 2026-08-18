#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
source venv/bin/activate
exec uvicorn app:app --host 0.0.0.0 --port 8002 --reload
