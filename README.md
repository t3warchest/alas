# ALAS Agent Service

**Adaptive Learning Agent System** — Production-inspired AI orchestration demo.

LangGraph · Multi-layer Memory · Real-time Evaluation · FastAPI · ChromaDB

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        AGENT SERVICE                            │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              LANGGRAPH ORCHESTRATION GRAPH               │   │
│  │                                                          │   │
│  │  ┌──────────────┐    ┌──────────────┐    ┌───────────┐  │   │
│  │  │ContextAssembly│───▶│PromptBuilder │───▶│  LLMCall  │  │   │
│  │  └──────────────┘    └──────────────┘    └─────┬─────┘  │   │
│  │         │                   │                  │         │   │
│  │    [STM + Episodic   [Scenario config    [streaming      │   │
│  │     + Semantic RAG]   + memory blocks]    tokens]        │   │
│  │                                                 │         │   │
│  │  ┌──────────────┐    ┌──────────────┐    ┌─────▼─────┐  │   │
│  │  │EvaluationStep│◀───│ MemoryUpdate │◀───│ResponseParser│ │   │
│  │  └──────┬───────┘    └──────────────┘    └───────────┘  │   │
│  │         │                                                 │   │
│  │         ▼  PhaseRouter                                    │   │
│  │    ┌─────────────────────────┐                           │   │
│  │    │  continue (END)         │                           │   │
│  │    │  └─▶ await next message │                           │   │
│  │    │  end_session            │                           │   │
│  │    │  └─▶ flush to episodic  │                           │   │
│  │    └─────────────────────────┘                           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  MEMORY HIERARCHY                        │   │
│  │                                                          │   │
│  │  Layer 1 — Short-Term Memory (in-process dict)           │   │
│  │    • recent N turns (sliding window)                     │   │
│  │    • active phase, turn index, live scores               │   │
│  │    • lifetime: session duration only                     │   │
│  │                                                          │   │
│  │  Layer 2 — Episodic Memory (ChromaDB: "episodes")        │   │
│  │    • post-session summaries, behavioral notes            │   │
│  │    • retrieved by user_id metadata filter                │   │
│  │    • influences: memory_block in every system prompt     │   │
│  │                                                          │   │
│  │  Layer 3 — Semantic Memory (ChromaDB: two collections)   │   │
│  │    • scenario_knowledge: chunked domain docs (RAG)       │   │
│  │    • user_profiles: embedded skill trajectory vectors    │   │
│  │    • retrieved by ANN similarity to student utterance    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  EVALUATION ENGINE                       │   │
│  │    • parallel LLM call (gpt-4o-mini) per student turn    │   │
│  │    • rubric dimensions: clarity, empathy, structure,     │   │
│  │      relevance, confidence  (all 0.0–1.0)               │   │
│  │    • per-turn scores → session aggregation → trend       │   │
│  │    • behavioral notes flushed to episodic at end         │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
      ┌───────────────┐              ┌────────────────┐
      │  REST API     │              │  WebSocket     │
      │  FastAPI      │              │  (streaming)   │
      │  /api/v1/...  │              │  /ws/{session} │
      └───────────────┘              └────────────────┘
```

---

## File Layout

```
alas/
├── agent_service/
│   ├── main.py                   FastAPI app + lifespan
│   ├── session.py                SessionOrchestrator (public API)
│   │
│   ├── graph/
│   │   ├── state.py              AgentState TypedDict (shared contract)
│   │   ├── agent_graph.py        LangGraph nodes + compiled graph
│   │   ├── prompt_builder.py     Dynamic system prompt construction
│   │   └── response_parser.py    Spoken text + metadata extraction
│   │
│   ├── memory/
│   │   └── manager.py            STM + EpisodicMemory + SemanticMemory + MemoryManager
│   │
│   ├── evaluation/
│   │   └── engine.py             Per-turn rubric scoring + session aggregation
│   │
│   ├── scenarios/
│   │   └── registry.py           Built-in scenarios + ingest_all_scenarios()
│   │
│   ├── api/
│   │   ├── routes.py             REST endpoints
│   │   ├── websocket.py          WebSocket streaming handler
│   │   └── models.py             Pydantic request/response DTOs
│   │
│   └── utils/
│       ├── config.py             Pydantic Settings (single env source)
│       └── logging.py            Structured JSON logging
│
├── tests/
│   ├── test_response_parser.py   Parser unit tests (no deps)
│   ├── test_evaluation.py        Aggregation unit tests (no deps)
│   ├── test_memory_stm.py        STM unit tests (no deps)
│   ├── test_prompt_builder.py    Prompt builder unit tests (no deps)
│   └── test_session_integration.py  Integration tests (mocked LLM)
│
├── validate.py                   Standalone validation (no pip needed)
├── demo.py                       Interactive terminal demo (rich UI)
├── requirements.txt
├── pyproject.toml
└── .env.example
```

---

## How Memory Visibly Influences Responses

### First session
```
MEMORY CONTEXT
This appears to be the student's first session.
```
Avatar asks broad warm-up questions, no assumptions.

### Second+ session
```
MEMORY CONTEXT
Prior session summaries:
  [1] Session abc | Turns: 12 | composite:0.61
      Avg scores — clarity:0.65 empathy:0.48 structure:0.70

Behavioral notes:
  • Composite: 0.61 (stable). Strength: structure (0.70).
    Growth area: empathy (0.48). Evaluated over 12 turns.

Reference this memory naturally in your responses.
```
Avatar explicitly references prior weakness:
> "Last time we spoke, I noticed you had strong structure in your answers — good STAR framing. This time I want to push on the empathy dimension a bit more. How do you typically handle a team member who disagrees with your technical direction?"

---

## Setup & Run

```bash
# 1. Clone / enter project
cd alas

# 2. Create env
cp .env.example .env
# edit .env → set OPENAI_API_KEY

# 3. Install
pip install -r requirements.txt

# 4. Validate (no API key needed)
python validate.py

# 5. Terminal demo (requires API key)
python demo.py                                     # job interview
python demo.py --scenario difficult_conversation_v1
python demo.py --list                              # list scenarios

# 6. API server
uvicorn agent_service.main:app --reload

# 7. Full test suite
pytest tests/ -v
```

---

## REST API Quick Reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness probe |
| GET | `/scenarios` | List available scenarios |
| POST | `/sessions` | Create session → returns opening line |
| POST | `/sessions/message` | Send message → avatar response + score |
| GET | `/sessions/{id}/scores` | All per-turn scores |
| POST | `/sessions/{id}/end` | End session → evaluation summary |
| WS | `/ws/{session_id}` | Streaming conversation |

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| LangGraph StateGraph | Explicit DAG topology; each node is a testable pure function; conditional edges cleanly encode phase routing |
| AgentState TypedDict | Single data contract threaded through all nodes — no hidden coupling |
| Three memory layers | STM for speed, episodic for cross-session continuity, semantic for RAG — each layer independently replaceable |
| Dynamic prompt assembly | Scenario-agnostic agent; new scenarios need only JSON config + knowledge chunks |
| Separate eval LLM | Evaluation never blocks the conversation; uses cheaper model; isolated failure |
| Response delimiter | Reliable spoken/metadata separation without regex fragility |
| Pydantic Settings | Single env source; typed; testable without mock patches |
| Structured JSON logging | Drop-in for any log aggregator; no parsing required |
