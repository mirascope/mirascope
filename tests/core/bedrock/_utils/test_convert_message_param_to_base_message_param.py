import pytest

from mirascope.core import BaseMessageParam
from mirascope.core.base import DocumentPart, ImagePart, TextPart, ToolCallPart
from mirascope.core.bedrock._utils._convert_message_param_to_base_message_param import (
    convert_message_param_to_base_message_param,
)


def test_single_text():
    message_param = {"role": "assistant", "content": [{"text": "Hello world"}]}
    result = convert_message_param_to_base_message_param(message_param)  # pyright: ignore [reportArgumentType]
    assert isinstance(result, BaseMessageParam)
    assert result.role == "assistant"
    assert result.content == "Hello world"


def test_multiple_text():
    message_param = {
        "role": "assistant",
        "content": [{"text": "Hello"}, {"text": "World"}],
    }
    result = convert_message_param_to_base_message_param(message_param)  # pyright: ignore [reportArgumentType]
    assert isinstance(result, BaseMessageParam)
    assert result.role == "assistant"
    assert len(result.content) == 2
    assert isinstance(result.content[0], TextPart)
    assert result.content[0].text == "Hello"
    assert isinstance(result.content[1], TextPart)
    assert result.content[1].text == "World"


def test_text_not_string():
    message_param = {"role": "assistant", "content": [{"text": 123}]}
    with pytest.raises(ValueError, match="Text content must be a string."):
        convert_message_param_to_base_message_param(message_param)  # pyright: ignore [reportArgumentType]


def test_image_block_valid():
    message_param = {
        "role": "assistant",
        "content": [
            {"image": {"format": "JPEG", "source": {"bytes": b"fake_jpeg_data"}}}
        ],
    }
    result = convert_message_param_to_base_message_param(message_param)  # pyright: ignore [reportArgumentType]
    assert len(result.content) == 1
    assert isinstance(result.content[0], ImagePart)
    assert result.content[0].media_type == "image/jpeg"
    assert result.content[0].image == b"fake_jpeg_data"


def test_image_block_missing_bytes():
    message_param = {
        "role": "assistant",
        "content": [{"image": {"format": "PNG", "source": {}}}],
    }
    with pytest.raises(
        ValueError, match="Image block must have 'source.bytes' as bytes."
    ):
        convert_message_param_to_base_message_param(message_param)  # pyright: ignore [reportArgumentType]


def test_image_block_unsupported_format():
    message_param = {
        "role": "assistant",
        "content": [{"image": {"format": "TIFF", "source": {"bytes": b"something"}}}],
    }
    with pytest.raises(ValueError, match="Unsupported image format: TIFF"):
        convert_message_param_to_base_message_param(message_param)  # pyright: ignore [reportArgumentType]


def test_document_block_valid():
    message_param = {
        "role": "assistant",
        "content": [{"document": {"format": "PDF", "source": {"bytes": b"pdf_data"}}}],
    }
    result = convert_message_param_to_base_message_param(message_param)  # pyright: ignore [reportArgumentType]
    assert len(result.content) == 1
    assert isinstance(result.content[0], DocumentPart)
    assert result.content[0].media_type == "application/pdf"
    assert result.content[0].document == b"pdf_data"


def test_document_block_unsupported_format():
    message_param = {
        "role": "assistant",
        "content": [{"document": {"format": "DOCX", "source": {"bytes": b"doc_data"}}}],
    }
    with pytest.raises(ValueError, match="Unsupported document format: DOCX"):
        convert_message_param_to_base_message_param(message_param)  # pyright: ignore [reportArgumentType]


def test_document_block_missing_bytes():
    message_param = {
        "role": "assistant",
        "content": [{"document": {"format": "PDF", "source": {}}}],
    }
    with pytest.raises(
        ValueError, match="Document block must have 'source.bytes' as bytes."
    ):
        convert_message_param_to_base_message_param(message_param)  # pyright: ignore [reportArgumentType]


def test_tool_use_block():
    message_param = {
        "role": "assistant",
        "content": [
            {
                "toolUse": {
                    "name": "my_tool",
                    "toolUseId": "tool_123",
                    "input": {"arg": "val"},
                }
            }
        ],
    }
    result = convert_message_param_to_base_message_param(message_param)  # pyright: ignore [reportArgumentType]

    assert result.role == "assistant"  # According to code as is.
    assert len(result.content) == 1
    assert isinstance(result.content[0], ToolCallPart)
    assert result.content[0].name == "my_tool"
    assert result.content[0].id == "tool_123"
    assert result.content[0].args == {"arg": "val"}


def test_no_supported_content():
    message_param = {"role": "assistant", "content": [{}]}
    with pytest.raises(
        ValueError, match="Content block does not contain supported content."
    ):
        convert_message_param_to_base_message_param(message_param)  # pyright: ignore [reportArgumentType]


def test_empty_content():
    message_param = {"role": "assistant", "content": []}
    result = convert_message_param_to_base_message_param(message_param)  # pyright: ignore [reportArgumentType]
    assert result.role == "assistant"
    assert result.content == []
