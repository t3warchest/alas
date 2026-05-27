"""
shared/contracts/__init__.py — public re-exports from the shared contracts layer.
"""
from shared.contracts.events import (
    WS_MSG_MESSAGE, WS_MSG_PING, WS_MSG_END, WS_MSG_RECONNECT, CLIENT_EVENTS,
    WS_EVT_TOKEN, WS_EVT_TURN_DONE, WS_EVT_SCORE, WS_EVT_PHASE_CHANGE,
    WS_EVT_SESSION_END, WS_EVT_PONG, WS_EVT_ERROR, WS_EVT_FATAL,
    WS_EVT_AUTH_OK, WS_EVT_CONNECTED, SERVER_EVENTS,
)
from shared.contracts.messages import (
    WSMessage, ClientMessage, ClientPing, ClientReconnect,
    AuthOkEvent, ConnectedEvent, TokenEvent, TurnDoneEvent,
    ScoreEvent, PhaseChangeEvent, SessionEndEvent, PongEvent,
    ErrorEvent, FatalEvent, parse_client_frame,
)
from shared.contracts.session import (
    SessionStatus, GatewaySession,
    CreateSessionDTO, SessionCreatedDTO, SessionSummaryDTO,
)
