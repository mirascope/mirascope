"""This module contains the `BaseCallResponseChunk` class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, field_serializer

from .tool import BaseTool

_ChunkT = TypeVar("_ChunkT", bound=Any)
_BaseToolT = TypeVar("_BaseToolT", bound=BaseTool)
_UserMessageParamT = TypeVar("_UserMessageParamT", bound=Any)


class BaseCallResponseChunk(
    BaseModel, Generic[_ChunkT, _BaseToolT, _UserMessageParamT], ABC
):
    """A base abstract interface for LLM streaming response chunks.

    Attributes:
        chunk: The original response chunk from whichever model response this wraps.
        tool_types: The tool types sent in the LLM call if any.
        user_message_param: The most recent message if it was a user message. Otherwise
            `None`.
        cost: The cost of the completion in dollars.
    """

    chunk: _ChunkT
    tool_types: list[type[_BaseToolT]] | None = None
    user_message_param: _UserMessageParamT | None = None
    cost: float | None = None

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    @property
    @abstractmethod
    def content(self) -> str:
        """Should return the string content of the response chunk.

        If there are multiple choices in a chunk, this method should select the 0th
        choice and return it's string content.

        If there is no string content (e.g. when using tools), this method must return
        the empty string.
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def model(self) -> str | None:
        """Should return the name of the response model."""
        ...  # pragma: no cover

    @property
    @abstractmethod
    def id(self) -> str | None:
        """Should return the id of the response."""
        ...  # pragma: no cover

    @property
    @abstractmethod
    def finish_reasons(self) -> list[str] | None:
        """Should return the finish reasons of the response.

        If there is no finish reason, this method must return None.
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def usage(self) -> Any:
        """Should return the usage of the response.

        If there is no usage, this method must return None.
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def input_tokens(self) -> int | float | None:
        """Should return the number of input tokens.

        If there is no input_tokens, this method must return None.
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def output_tokens(self) -> int | float | None:
        """Should return the number of output tokens.

        If there is no output_tokens, this method must return None.
        """
        ...  # pragma: no cover

    @field_serializer("tool_types")
    def serialize_tool_types(self, tool_types: list[type[_BaseToolT]], _info):
        return [{"type": "function", "name": tool.__name__} for tool in tool_types]

    def __str__(self) -> str:
        """Returns the string content of the chunk."""
        return self.content