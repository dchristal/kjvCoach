# kjvCoach

FastAPI app (`app.py`) serving a KJV Bible study / coaching tool. UI at
`http://127.0.0.1:8002/ui`. KJV text + cross-reference data in `kjv.db` /
`kjv1769.ccdb` (`ccdb.py` reads the ccdb format); `patterns.json` drives some
matching. `chat.py` / `kjvcode.py` hold the LLM-facing logic.

## Running

- `start.sh` on port **8002**, `# tidy:nogpu` — it's a UI/proxy, not a GPU consumer.
- Uses `venv/` (activated in `start.sh`). Recreate with `uv`, never copy across machines.
- LLM calls go to the shared `llama-server` at `http://localhost:8080/v1` (whatever
  model `tidy switch` has loaded). Previously pointed at LM Studio on :1234 — don't
  reintroduce that.

## Deploy

Has `fly.toml`, `Procfile`, `Dockerfile` — deployable to Fly.io. Local dev is the
default; check with Dave before touching Fly config or deploying.

Read `app.py` for endpoint specifics — this file is orientation only.
