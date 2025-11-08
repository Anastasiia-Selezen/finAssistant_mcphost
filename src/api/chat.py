import logging
from typing import Any

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from src.host.host import MCPHost

SYSTEM_PROMPT_NAME = "scope_financial_analysis_prompt"


from contextlib import asynccontextmanager

host = MCPHost()
logger = logging.getLogger("chat")
logging.basicConfig(level=logging.INFO)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await host.initialize()
    try:
        yield
    finally:
        await host.cleanup()

app = FastAPI(lifespan=lifespan)

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    reply: str
    tool_calls: list[dict[str, Any]] | None = None

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    logger.info("Requesting system prompt from MCPHost...")
    logger.info(f"User query: {request.query}")
    prompt_args = {"query": request.query}
    prompt = await host.get_system_prompt(SYSTEM_PROMPT_NAME, prompt_args)
    logger.info("Processing query with LLM...")
    result = await host.process_query(prompt_text=prompt.messages[0].content.text)
    return ChatResponse(reply=result.text, tool_calls=result.tool_calls)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=5005)
