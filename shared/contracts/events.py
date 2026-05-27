"""
shared/contracts/events.py

All WebSocket event type strings used between client ↔ gateway ↔ agent service.
Defining them as string constants (not an enum) keeps JSON serialisation trivial
while preventing typo-driven bugs across the codebase.

Client → Gateway events:
    WS_MSG_*   — messages the client sends to the server

Gateway → Client events:
    WS_EVT_*   — events the server pushes to the client
"""

# ── Client → Gateway ────────────────────────────────────────────────────────

WS_MSG_MESSAGE    = "message"       # student sends a chat message
WS_MSG_PING       = "ping"         # heartbeat from client
WS_MSG_END        = "end"          # client requests session termination
WS_MSG_RECONNECT  = "reconnect"    # client reconnects to an existing session

CLIENT_EVENTS = frozenset({
    WS_MSG_MESSAGE,
    WS_MSG_PING,
    WS_MSG_END,
    WS_MSG_RECONNECT,
})

# ── Gateway → Client ────────────────────────────────────────────────────────

WS_EVT_TOKEN        = "token"          # streaming response token
WS_EVT_TURN_DONE    = "turn_done"      # full turn complete (replaces old "result")
WS_EVT_SCORE        = "score"          # per-turn evaluation score
WS_EVT_PHASE_CHANGE = "phase_change"   # scenario phase transitioned
WS_EVT_SESSION_END  = "session_end"    # session concluded + final summary
WS_EVT_PONG         = "pong"           # heartbeat reply
WS_EVT_ERROR        = "error"          # recoverable error (session stays open)
WS_EVT_FATAL        = "fatal"          # non-recoverable error (session closing)
WS_EVT_AUTH_OK      = "auth_ok"        # connection authenticated
WS_EVT_CONNECTED    = "connected"      # session ready, avatar opening line included

SERVER_EVENTS = frozenset({
    WS_EVT_TOKEN,
    WS_EVT_TURN_DONE,
    WS_EVT_SCORE,
    WS_EVT_PHASE_CHANGE,
    WS_EVT_SESSION_END,
    WS_EVT_PONG,
    WS_EVT_ERROR,
    WS_EVT_FATAL,
    WS_EVT_AUTH_OK,
    WS_EVT_CONNECTED,
})
