# Multi-Memory Assistant

An interactive CLI chat assistant built with Python, LangChain, Google Gemini (`langchain-google-genai`), SQLAlchemy, and PostgreSQL (`pgvector`), featuring streaming responses, session-based short-term memory, sliding-window summarization, semantic vector memory, and episodic memory.

## Features

- **Streaming CLI chat**: Real-time token streaming via `ChatGoogleGenerativeAI`.
- **Session-based short-term memory**: `ShortTermMemory` manages isolated per-session message histories with empty-ID validation and turn-safe rollback on streaming or summarization errors.
- **Token-budgeted context compression**: Tracks message token counts per turn and summarizes older exchanges into a structured summary while preserving recent exchanges.
- **Structured summarization**: Prioritizes stable user facts, preferences, technical skills, active projects, and decisions over transient details without turning assumptions into facts.
- **Semantic memory storage, search & updates**: Generates document/query embeddings via `EmbeddingService` (`GoogleGenerativeAIEmbeddings`) and stores, updates (regenerating embeddings only when content changes), and queries user-scoped vector memories in PostgreSQL with `pgvector` through `SemanticMemoryRepository` and `SemanticMemoryService`.
- **Episodic memory tracking**: Stores and retrieves session- and user-scoped event/conversation summaries with deterministic pagination through `EpisodicMemoryRepository` and `EpisodicMemoryService`.
- **Database migrations**: SQLAlchemy ORM models (`SemanticMemory`, `EpisodicMemory`) managed with Alembic.

## Project Structure

```text
├── app/
│   ├── main.py                               # CLI entry point (`uv run assistant`) and chat loop
│   ├── core/
│   │   ├── config.py                         # Environment configuration (`Settings`)
│   │   └── database.py                       # SQLAlchemy engine, `SessionLocal`, `Base`, `get_db`
│   ├── agents/
│   │   ├── chat_agent.py                     # Gemini chat model factory
│   │   └── summarize_agent.py                # Conversation summarization agent and prompt
│   ├── memory/
│   │   ├── short_term.py                     # Session-based in-memory chat history
│   │   ├── semantic.py                       # `SemanticMemoryService` (`create_memory`, `search_memories`, `update_memory`)
│   │   └── episodic.py                       # `EpisodicMemoryService` (`create_memory`, `get_memory`, `get_session_memories`, `get_user_memories`)
│   ├── models/
│   │   ├── semantic_memory.py                # `SemanticMemory` ORM model (`pgvector`)
│   │   └── episodic_memory.py                # `EpisodicMemory` ORM model
│   ├── repositories/
│   │   ├── semantic_memory_repository.py     # Semantic memory CRUD, soft deactivation, and cosine similarity search
│   │   └── episodic_memory_repository.py     # Episodic memory creation, ID/session lookups, and paginated user queries
│   ├── services/
│   │   └── embedding_service.py              # Gemini document/query embedding wrapper
│   └── utils/
│       └── chat_services.py                  # Sliding-window history summarization and token validation
├── migrations/                               # Alembic migration environment and revisions
├── scripts/
│   └── manual_semantic_memory_repository.py  # Manual live DB/Gemini repository check
└── tests/
    ├── test_chat_services.py                 # Unit tests for sliding-window summarization
    ├── test_main.py                          # Integration/rollback tests for CLI loop
    ├── test_short_term.py                    # Unit tests for `ShortTermMemory`
    ├── test_semantic_memory_repository.py    # Unit tests for `SemanticMemoryRepository`
    ├── test_semantic_memory_service.py       # Unit tests for `SemanticMemoryService`
    ├── test_episodic_memory_repository.py    # Unit tests for `EpisodicMemoryRepository`
    ├── test_episodic_memory_service.py       # Unit tests for `EpisodicMemoryService`
    └── test_embedding_service.py             # Manual live Gemini embedding check
```

## Setup

1. **Install dependencies** (requires Python `>=3.14` and [`uv`](https://docs.astral.sh/uv/)):

   ```bash
   uv sync
   ```

2. **Configure environment variables**:

   Copy `.env.example` to `.env` and fill in your credentials:

   ```bash
   cp .env.example .env
   ```

   | Variable | Default | Description |
   | --- | --- | --- |
   | `GEMINI_API_KEY` | *(required)* | Google Gemini API key |
   | `CHAT_MODEL` | `gemini-2.5-flash` | Gemini model used for chat and summarization |
   | `SLIDING_WINDOW_SIZE` (or `SLIDING_WIN_SIZE`) | `500` | Token threshold that triggers history compression |
   | `DATABASE_URL` | *(required for DB)* | PostgreSQL connection URL (e.g. `postgresql+psycopg://user:pass@localhost:5432/multi_memory`) |
   | `EMBEDDING_MODEL` | `gemini-embedding-001` | Gemini embedding model used by `EmbeddingService` |
   | `EMBEDDING_DIMENSION` | `1536` | Vector dimension for embeddings and `pgvector` column (validated against the `1536` schema contract) |

3. **Run database migrations** (requires PostgreSQL with the `pgvector` extension enabled):

   ```bash
   uv run alembic upgrade head
   ```

## Usage

Start the interactive assistant:

```bash
uv run assistant
```

Type `exit`, `quit`, or `bye` to end the session.

## Testing

Run the automated unit test suite (mocked, no external API or database calls):

```bash
uv run pytest -v
```

Manual live verification scripts (require valid `GEMINI_API_KEY` and `DATABASE_URL`):

```bash
uv run python -m tests.test_embedding_service
uv run python -m scripts.manual_semantic_memory_repository
```
