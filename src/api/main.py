"""
FastAPI server for the AI Consciousness RAG chatbot.

Exposes the QueryBot over HTTP: a stateless one-shot endpoint and a
session-based chat API that keeps conversation history so follow-up questions
stay coherent.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.api.security import (
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
    allowed_origins,
    docs_enabled,
    rate_limit_settings,
)
from src.api.session_store import Message, SessionStore
from src.config import config
from src.rag_pipeline.bot import QueryBot, format_source_doc

logger = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 2000

bot: QueryBot | None = None
sessions = SessionStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads the vector store and LLM once, before the API serves traffic."""
    global bot
    logger.info("Initializing QueryBot for the API...")
    bot = QueryBot(config.rag_application)
    logger.info("QueryBot initialized successfully.")
    yield


app = FastAPI(
    title="AI Consciousness Research Assistant API",
    description="Query a RAG pipeline on academic texts about consciousness.",
    version="0.2.0",
    lifespan=lifespan,
    docs_url="/docs" if docs_enabled() else None,
    redoc_url="/redoc" if docs_enabled() else None,
    openapi_url="/openapi.json" if docs_enabled() else None,
)

app.add_middleware(SecurityHeadersMiddleware)

_max_requests, _window_seconds = rate_limit_settings()
app.add_middleware(
    RateLimitMiddleware, max_requests=_max_requests, window_seconds=_window_seconds
)

# Browsers block cross-origin calls from the frontend dev server, so allow it
# explicitly. Origins are whitelisted rather than wildcarded, and credentials
# stay off because the API uses no cookies or auth headers.
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)


# --- Pydantic Models for API Data Validation ---

class AskRequest(BaseModel):
    """Defines the schema for a request to the /ask endpoint."""
    query: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    length: str = "medium"


class AskResponse(BaseModel):
    """Defines the schema for a response from the /ask endpoint."""
    answer: str
    sources: list[str]
    refused: bool


class MessageRequest(BaseModel):
    """A user message sent to an existing chat session."""
    message: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    length: str = "medium"


class MessageOut(BaseModel):
    """A stored message returned when replaying a conversation."""
    role: str
    content: str
    sources: list[str] = []
    refused: bool = False


class HistoryResponse(BaseModel):
    messages: list[MessageOut]


class SessionResponse(BaseModel):
    session_id: str


class HealthResponse(BaseModel):
    status: str
    bot_ready: bool


# --- Helpers ---

def _require_bot() -> QueryBot:
    if bot is None:
        raise HTTPException(
            status_code=503, detail="Bot is not initialized. Please wait."
        )
    return bot


def _resolve_length(length: str) -> int:
    length_map = config.rag_application.answer_length_map
    if length not in length_map:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid length. Choose from: {', '.join(length_map.keys())}",
        )
    return length_map[length]


def _require_session(session_id: str) -> None:
    if not sessions.exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found.")


# --- API Endpoints ---

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe reporting whether the model is loaded and ready."""
    return HealthResponse(status="ok", bot_ready=bot is not None)


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:
    """Answers a single question without any conversation history."""
    active_bot = _require_bot()
    result = active_bot.ask(
        query=request.query, num_predict_tokens=_resolve_length(request.length)
    )
    return AskResponse(
        answer=result["result"],
        sources=_format_sources(result["source_documents"]),
        refused=result["refused"],
    )


@app.post("/sessions", response_model=SessionResponse, status_code=201)
def create_session() -> SessionResponse:
    """Starts a new chat session."""
    return SessionResponse(session_id=sessions.create())


@app.get("/sessions/{session_id}/messages", response_model=HistoryResponse)
def get_messages(session_id: str) -> HistoryResponse:
    """Replays the conversation, so a reloaded page can restore its history."""
    _require_session(session_id)
    return HistoryResponse(
        messages=[
            MessageOut(
                role=m.role, content=m.content, sources=m.sources, refused=m.refused
            )
            for m in sessions.get_messages(session_id)
        ]
    )


@app.post("/sessions/{session_id}/messages", response_model=AskResponse)
def post_message(session_id: str, request: MessageRequest) -> AskResponse:
    """Answers a message in the context of its session's history."""
    active_bot = _require_bot()
    _require_session(session_id)
    num_tokens = _resolve_length(request.length)

    # Read history before storing the new message so the current question is
    # not duplicated into its own context.
    history = sessions.history_pairs(session_id)
    result = active_bot.ask(
        query=request.message, num_predict_tokens=num_tokens, chat_history=history
    )
    sources = _format_sources(result["source_documents"])

    sessions.append(session_id, Message(role="user", content=request.message))
    sessions.append(
        session_id,
        Message(
            role="assistant",
            content=result["result"],
            sources=sources,
            refused=result["refused"],
        ),
    )

    return AskResponse(
        answer=result["result"], sources=sources, refused=result["refused"]
    )


@app.delete("/sessions/{session_id}", status_code=204)
def delete_session(session_id: str) -> None:
    """Ends a chat session and discards its history."""
    _require_session(session_id)
    sessions.delete(session_id)


def _format_sources(source_documents: list) -> list[str]:
    return sorted({format_source_doc(doc) for doc in source_documents})
