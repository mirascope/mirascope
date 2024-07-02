"""This module contains the setup_call function, which is used to set up the"""

import inspect
from typing import Any, Awaitable, Callable

from google.generativeai import GenerativeModel  # type: ignore
from google.generativeai.types import (  # type: ignore
    ContentsType,
    GenerateContentResponse,
)

from ...base import BaseTool, _utils
from ..call_params import GeminiCallParams
from ..dynamic_config import GeminiDynamicConfig
from ..tool import GeminiTool


def setup_call(
    *,
    model: str,
    client: object | None,
    fn: Callable[..., GeminiDynamicConfig | Awaitable[GeminiDynamicConfig]],
    fn_args: dict[str, Any],
    dynamic_config: GeminiDynamicConfig,
    tools: list[type[BaseTool] | Callable] | None,
    json_mode: bool,
    call_params: GeminiCallParams,
    extract: bool = False,
) -> tuple[
    Callable[..., GenerateContentResponse],
    str,
    list[dict[str, ContentsType]],
    list[type[GeminiTool]],
    dict[str, Any],
]:
    prompt_template, messages, tool_types, call_kwargs = _utils.setup_call(
        fn, fn_args, dynamic_config, tools, GeminiTool, call_params
    )

    gemini_messages = []
    for message in messages:
        if (role := message["role"]) == "system":
            gemini_messages += [
                {
                    "role": "user",
                    "parts": [message["content"]],
                },
                {
                    "role": "model",
                    "parts": ["Ok! I will adhere to this system message."],
                },
            ]
        else:
            gemini_messages.append({"role": role, "parts": [message["content"]]})

    if json_mode:
        generation_config = call_kwargs.get("generation_config", {})
        generation_config["response_mime_type"] = "application/json"
        call_kwargs["generation_config"] = generation_config
        gemini_messages[-1]["parts"][-1] += _utils.json_mode_content(
            tool_types[0] if tool_types else None
        )
        call_kwargs.pop("tools", None)
    elif extract:
        tool_config = call_kwargs.get("tool_config", {})
        tool_config["function_calling_config"] = {"mode": "auto"}
        call_kwargs["tool_config"] = tool_config

    if client is None:
        client = GenerativeModel(model_name=model)
    create = (
        client.generate_content_async
        if inspect.iscoroutinefunction(fn)
        else client.generate_content
    )
    call_kwargs |= {"contents": gemini_messages}

    return create, prompt_template, gemini_messages, tool_types, call_kwargs