# Financial AI Assistant MCP Host and MCP Client

## Overview
Financial AI Assistant MCP Host is a FastAPI service that brokers chat-style requests into an MCP (Model Context Protocol) tool registry. It retrieves a named system prompt, hands the conversation to a LangChain agent backed by OpenAI, and returns both the model’s reply and any MCP tool invocations it performed along the way.

## Features
- FastAPI endpoint for submitting financial analysis queries.
- LangChain agent (GPT-4o-mini by default) with configurable recursion guardrails.
- Automatic discovery of MCP tools and prompts from a remote tool registry.
- Structured response capturing executed tool calls for auditability.

## Project Structure
- `src/api/chat.py` – FastAPI app and `/chat` endpoint.
- `src/host/host.py` – LangChain agent orchestration and MCP integration.
- `src/host/connection_manager.py` – Async connection lifecycle for MCP servers.
- `src/config.py` – Environment variables configuration using `pydantic-settings`.

## Prerequisites
- Python 3.12+.
- An OpenAI API key with access to the chosen chat model.
- URL for an MCP-compatible tool registry exposing the required prompt and tools.
- [`uv`](https://github.com/astral-sh/uv) for dependency management (a `uv.lock` is included).

## Installation

```bash
uv sync
```


## Configuration
Create a `.env` file in the project root:

```dotenv
OPENAI_API_KEY=sk-your-key
TOOL_REGISTRY_URL=https://your-tool-registry.example.com
```

Both values are required, startup will fail if either is missing or blank.

## Running the API

### Using uv
```bash
uv run uvicorn src.api.chat:app --host 0.0.0.0 --port 5001 --reload
```

### Using uvicorn directly
```bash
uvicorn src.api.chat:app --host 0.0.0.0 --port 5001 --reload
```

When the server starts it initializes the MCP connection, lists discovered tools, and becomes ready to accept chat requests.

## API Usage

### Endpoint
`POST /chat`

### Request body
```json
{
  "query": "Hey, I'm researching the company that builds the Model Y but I can't remember its stock symbol. Could you figure out the right ticker, pull a 5-minute intraday snapshot for today, summarize any headline sentiment from the last 24 hours, and grab a short excerpt from the latest 10‑K risk factors (1.A)?"
}
```

### Response body
```json
{
  "reply": "The company that builds the Model Y is **Tesla, Inc.**, and its stock symbol is **TSLA**....",
  "tool_calls": [
    {
      "name": "alphavantage_search_symbol",
      "args": {
        "keywords": "Tesla"
      },
      "result": "..."
    },
    {
      "name": "alphavantage_get_intraday",
      "args": {
        "symbol": "TSLA",
        "interval": "5min"
      },
      "result": "..."
    }
  ]
}
```

### Example
```bash
curl -X POST http://localhost:5001/chat \
     -H "Content-Type: application/json" \
     -d '{"query": "Summarize recent trends in the S&P 500."}'
```

