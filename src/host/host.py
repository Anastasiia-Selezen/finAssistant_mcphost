import logging
import uuid
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel, create_model

from ..config import settings

from .connection_manager import MCPClient


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_host")

from dataclasses import dataclass


@dataclass
class ChatResult:
    text: str
    tool_calls: list[dict[str, Any]]


class MCPHost:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        openai_api_key = settings.OPENAI_API_KEY
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set. Please set it to your OpenAI API key.")
        self.llm = ChatOpenAI(
            model=self.model,
            temperature=0,
            api_key=openai_api_key,
        )
        self.client = MCPClient()
        self.thread_id = str(uuid.uuid4())
        self._agent = None
        self._recursion_limit = 10

    async def initialize(self):
        await self.client.initialize_all()

    async def get_system_prompt(self, name, args) -> str:
        if not self.client.is_initialized:
            raise RuntimeError("MCP Client is not initialized. Call initialize_all() first.")
        return await self.client.get_prompt(name, args)

    async def process_query(self, prompt_text: str) -> ChatResult:
        if not self.client.is_initialized:
            raise RuntimeError("MCP Client is not initialized. Call initialize_all() first.")

        agent = await self._get_agent()
        messages = [SystemMessage(content=prompt_text)] if prompt_text else []

        try:
            result_state = await agent.ainvoke(
                {"messages": messages},
                config={
                    "recursion_limit": self._recursion_limit,
                    "configurable": {"thread_id": self.thread_id},
                },
            )
            message_history: list[BaseMessage] = []
            if isinstance(result_state, dict):
                message_history = result_state.get("messages", [])

            final_text = self._get_final_message_text(message_history)
            tool_calls = self._collect_tool_calls(message_history)

            logger.info("Final response: %s", final_text)
            return ChatResult(text=final_text, tool_calls=tool_calls)
        except GraphRecursionError:
            logger.error(
                "LangGraph recursion limit (%d) reached without completion; returning fallback response.",
                self._recursion_limit,
            )
            return ChatResult(
                text="I hit the iteration limit while trying to figure this out. You can retry or rephrase the request.",
                tool_calls=[],
            )

    async def _get_agent(self):
        if self._agent is None:
            tools = await self._load_langgraph_tools()
            self._agent = create_agent(self.llm, tools)
        return self._agent

    async def _load_langgraph_tools(self) -> list[StructuredTool]:
        tools_response = await self.client.get_mcp_tools()
        langchain_tools: list[StructuredTool] = []

        for tool in tools_response.tools:
            tool_name = tool.name
            args_model = self._make_args_model(tool)

            async def _call_tool(_name: str = tool_name, **kwargs: Any):
                payload = kwargs.get("payload", kwargs)
                if not isinstance(payload, dict):
                    payload = {"value": payload}
                logger.info("Calling MCP tool '%s' with %s", _name, payload)
                result = await self.client.call_tool(_name, payload)
                logger.info("MCP tool '%s' returned %s", _name, result)
                return result

            langchain_tools.append(
                StructuredTool(
                    name=tool_name,
                    description=tool.description or "",
                    args_schema=args_model,
                    func=self._sync_tool_placeholder,
                    coroutine=_call_tool,
                )
            )

        return langchain_tools

    def _make_args_model(self, tool) -> type[BaseModel]:
        """
        Build a lightweight arguments model that exposes the tool's top-level fields.
        Nested details are ignored to keep the schema compact.
        """
        schema = tool.inputSchema or {}
        properties: dict[str, Any] = schema.get("properties") or {}
        required_fields = set(schema.get("required") or [])

        field_definitions: dict[str, tuple[Any, Any]] = {}
        for field_name, spec in properties.items():
            default_value = spec.get("default", ...) if field_name in required_fields else spec.get("default", None)
            field_definitions[field_name] = (Any, default_value)

        if not field_definitions:
            field_definitions["payload"] = (dict[str, Any], ...)

        model_name = f"{tool.name.title().replace(' ', '').replace('-', '_')}Input"
        return create_model(model_name, **field_definitions)

    def _collect_tool_calls(self, messages: list[BaseMessage]) -> list[dict[str, Any]]:
        pending_calls: dict[str, dict[str, Any]] = {}
        executed_calls: list[dict[str, Any]] = []

        for message in messages:
            if isinstance(message, AIMessage):
                for tool_call in getattr(message, "tool_calls", []) or []:
                    call_id = tool_call.get("id")
                    if call_id:
                        pending_calls[call_id] = {
                            "name": tool_call.get("name"),
                            "args": tool_call.get("args") or {},
                        }
            elif isinstance(message, ToolMessage):
                call_id = getattr(message, "tool_call_id", None)
                payload = message.content
                info = pending_calls.pop(call_id, {}) if call_id else {}
                executed_calls.append(
                    {
                        "name": info.get("name") or getattr(message, "name", None),
                        "args": info.get("args") or {},
                        "result": payload,
                    }
                )

        return executed_calls

    def _get_final_message_text(self, messages: list[BaseMessage]) -> str:
        final_message = next(
            (message for message in reversed(messages) if isinstance(message, AIMessage)),
            None,
        )
        if not final_message:
            return ""

        content = final_message.content
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            return "".join(str(part) for part in content)

        return str(content)

    @staticmethod
    def _sync_tool_placeholder(**_: Any):
        raise RuntimeError("Synchronous execution is not supported for MCP tools.")

    async def get_mcp_tools(self):
        if not self.client.is_initialized:
            raise RuntimeError("MCP Client is not initialized. Call initialize_all() first.")
        return await self.client.get_mcp_tools()

    async def call_tool(self, function_name: str, function_args: dict) -> Any:
        if not self.client.is_initialized:
            raise RuntimeError("MCP Client is not initialized. Call initialize_all() first.")
        return await self.client.call_tool(function_name, function_args)

    async def cleanup(self):
        await self.client.cleanup_all()
