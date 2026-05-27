# ALAS — Local Run Instructions

## Prerequisites

```bash
pip install fastapi uvicorn langchain langgraph langchain-openai \
            chromadb openai pydantic pydantic-settings httpx
```

## Environment setup

```bash
# Agent service (.env in alas/)
cp .env.example .env
# Set: OPENAI_API_KEY=sk-...

# Gateway (.env.example in alas/gateway/)
# Defaults work out of the box for local dev — no changes needed
```

## Start both services

**Terminal 1 — Agent Service (port 8000)**
```bash
cd alas/
uvicorn agent_service.main:app --port 8000 --reload
```

**Terminal 2 — Gateway (port 8001)**
```bash
cd alas/
uvicorn gateway.main:app --port 8001 --reload
```

## Verify

```bash
# Gateway health
curl http://localhost:8001/health

# Agent health (via gateway)
curl http://localhost:8001/health/agent

# List scenarios
curl http://localhost:8001/api/scenarios

# Get a demo token
curl -X POST http://localhost:8001/auth/token \
     -H "Content-Type: application/json" \
     -d '{"user_id": "alice"}'

# Create a session
curl -X POST http://localhost:8001/api/sessions \
     -H "Content-Type: application/json" \
     -d '{"scenario_id": "job_interview_v1"}'
# Returns: {"session_id": "...", "ws_url": "ws://...", "opening_line": "..."}

# Connect via WebSocket (use wscat or websocat)
wscat -c "ws://localhost:8001/ws/<session_id>"
# Then send: {"type": "message", "content": "I have 5 years of backend experience."}
```

## Run validators (no API key needed)

```bash
cd alas/
python3 validate_backend.py       # 100 gateway tests
python3 validate_scenarios.py     # 80 scenario tests  
python3 validate.py               # 92 agent service tests
```

## WebSocket message format

**Client → Gateway**
```json
{"type": "message",   "content": "Your text here"}
{"type": "ping",      "timestamp": 1234567890.0}
{"type": "end"}
{"type": "reconnect", "session_id": "...", "last_turn_index": 3}
```

**Gateway → Client**
```json
{"type": "auth_ok",      "user_id": "alice",        "token_expires_in": 3600}
{"type": "connected",    "session_id": "...",        "opening_line": "...", "phase": "setup"}
{"type": "token",        "content": "Tell",         "turn_index": 2}
{"type": "token",        "content": " me",          "turn_index": 2}
{"type": "turn_done",    "avatar_response": "...",  "emotion": "curious", "phase": "core"}
{"type": "score",        "composite": 0.74,         "rationale": "Good STAR structure."}
{"type": "phase_change", "from_phase": "core",      "to_phase": "escalation"}
{"type": "session_end",  "summary": {...}}
{"type": "error",        "code": "...",             "message": "...", "recoverable": true}
```

## Architecture at a glance

```
Browser / Client
      │  REST: POST /api/sessions, GET /api/scenarios
      │  WS:   ws://localhost:8001/ws/{session_id}
      ▼
 Gateway :8001  (gateway/main.py)
   ├── Auth middleware   (JWT, demo passthrough)
   ├── Session store     (memory or SQLite)
   ├── Routers           (auth, sessions, websocket)
   └── AgentServiceClient ──HTTP──►  Agent Service :8000
                                       (agent_service/main.py)
                                         ├── LangGraph graph
                                         ├── Memory hierarchy
                                         ├── Scenario engine
                                         └── Evaluation engine
```
