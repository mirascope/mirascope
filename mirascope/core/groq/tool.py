"""The `GroqTool` class for easy tool usage with Groq LLM calls."""

from __future__ import annotations

import jiter
from groq.types.chat import (
    ChatCompletionMessageToolCall,
    ChatCompletionToolParam,
)
from groq.types.shared_params import FunctionDefinition
from pydantic.json_schema import SkipJsonSchema

from ..base import BaseTool


class GroqTool(BaseTool):
    """A class for defining tools for Groq LLM calls."""

    tool_call: SkipJsonSchema[ChatCompletionMessageToolCall]

    @classmethod
    def tool_schema(cls) -> ChatCompletionToolParam:
        """Constructs a JSON Schema tool schema from the `BaseModel` schema defined."""
        fn = FunctionDefinition(name=cls._name(), description=cls._description())
        model_schema = cls.model_tool_schema()
        if model_schema["properties"]:
            fn["parameters"] = model_schema
        return ChatCompletionToolParam(function=fn, type="function")

    @classmethod
    def from_tool_call(cls, tool_call: ChatCompletionMessageToolCall) -> GroqTool:
        """Constructs an `GroqTool` instance from a `tool_call`."""
        model_json = jiter.from_json(tool_call.function.arguments.encode())
        model_json["tool_call"] = tool_call.model_dump()
        return cls.model_validate(model_json)