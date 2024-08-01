"""Tests the `cohere._utils.handle_stream` module."""

import pytest
from cohere.types import (
    NonStreamedChatResponse,
    StreamedChatResponse,
    StreamedChatResponse_StreamEnd,
    StreamedChatResponse_TextGeneration,
    StreamedChatResponse_ToolCallsGeneration,
    ToolCall,
)

from mirascope.core.cohere._utils.handle_stream import (
    handle_stream,
    handle_stream_async,
)
from mirascope.core.cohere.tool import CohereTool


class FormatBook(CohereTool):
    """Returns the title and author nicely formatted."""

    title: str
    author: str

    def call(self):
        """Dummy call."""


@pytest.fixture()
def mock_chunks() -> list[StreamedChatResponse]:
    """Returns a list of mock `ChatCompletionChunk` instances."""
    tool_call = ToolCall(
        name="FormatBook",
        parameters={"title": "The Name of the Wind", "author": "Patrick Rothfuss"},
    )
    return [
        StreamedChatResponse_TextGeneration(
            text="json_output",
        ),
        StreamedChatResponse_ToolCallsGeneration(tool_calls=[tool_call]),
        StreamedChatResponse_StreamEnd(
            finish_reason="COMPLETE",
            response=NonStreamedChatResponse(generation_id="id", text="", meta=None),
        ),
    ]


def test_handle_stream(mock_chunks: list[StreamedChatResponse]) -> None:
    """Tests the `handle_stream` function."""

    result = [
        t for t in handle_stream((c for c in mock_chunks), tool_types=[FormatBook])
    ]
    assert len(result) == 3
    assert result[0][1] is None


@pytest.mark.asyncio
async def test_handle_stream_async(mock_chunks: list[StreamedChatResponse]) -> None:
    """Tests the `handle_stream_async` function."""

    async def generator():
        for chunk in mock_chunks:
            yield chunk

    result = []
    async for t in handle_stream_async(generator(), tool_types=[FormatBook]):
        result.append(t)
    assert len(result) == 3
    assert result[0][1] is None