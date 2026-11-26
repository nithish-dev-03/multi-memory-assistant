# Multi-Memory Assistant

An interactive CLI chat assistant built with Python, LangChain, and Google Gemini (`langchain-google-genai`), featuring streaming responses and sliding-window conversation summarization.

## Features

- **Streaming CLI chat**: Real-time token streaming via `ChatGoogleGenerativeAI`.
- **Token-budgeted context management**: Tracks message token counts per turn and compresses older exchanges into a structured summary while keeping recent exchanges intact.
- **Structured summarization**: Prioritizes stable user facts, preferences, technical skills, active projects, and decisions over transient small talk without turning assumptions into facts.
- **Turn-safe rollback**: Restores conversation history to its pre-turn state if streaming or summarization fails.

## Project Structure

```text
app/
├── main.py                     # CLI entry point (`uv run assistant`) and conversation loop
├── core/
│   └── config.py               # Environment configuration (`Settings`)
├── agents/
│   ├── chat_agent.py           # Gemini chat model factory
│   └── summarize_agent.py      # Conversation summarization agent and prompt
├── utils/
│   └── chat_services.py        # Sliding-window history summarization and token validation
└── memory/
    ├── short_term.py           # Placeholder for upcoming short-term memory module
    └── semantic.py             # Placeholder for upcoming semantic/vector memory
```

## Setup

1. **Install dependencies** (requires Python `>=3.14` and [`uv`](https://docs.astral.sh/uv/)):

   ```bash
   uv sync
   ```

2. **Configure environment variables**:

   Copy `.env.example` to `.env` and set your Gemini API key:

   ```bash
   cp .env.example .env
   ```

   | Variable | Default | Description |
   | --- | --- | --- |
   | `GEMINI_API_KEY` | *(required)* | Google Gemini API key |
   | `CHAT_MODEL` | `gemini-2.5-flash` | Gemini model used for chat and summarization |
   | `SLIDING_WINDOW_SIZE` (or `SLIDING_WIN_SIZE`) | `500` | Token threshold that triggers history compression |
   | `EMBEDDING_MODEL` | `gemini-embedding-001` | Configured in `Settings`, reserved for upcoming semantic memory (`app/memory/semantic.py`) |

## Usage

Start the interactive assistant:

```bash
uv run assistant
```

Type `exit`, `quit`, or `bye` to end the session.
