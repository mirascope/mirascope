# File: tests/test_convert_message_param_to_base_message_param.py
from unittest.mock import MagicMock, Mock

import pytest

from mirascope.core import BaseMessageParam
from mirascope.core.base import DocumentPart, ImagePart, TextPart
from mirascope.core.gemini._utils._convert_message_param_to_base_message_param import (
    _is_image_mime,
)
from mirascope.core.vertex._utils._convert_message_param_to_base_message_param import (
    _to_document_part,
    _to_image_part,
    convert_message_param_to_base_message_param,
)


def test_vertex_convert_parts_text_only():
    """
    Test vertex_convert_parts with a Part containing only text.
    """
    Part = MagicMock()  # Mock Part class
    mock_part = Part()
    mock_part.text = "hello world"
    mock_part.inline_data = None
    mock_part.file_data = None
    mock_part.function_call = None

    result = convert_message_param_to_base_message_param(Mock(parts=[mock_part]))
    assert isinstance(result, BaseMessageParam)
    assert result.role == "assistant"
    assert len(result.content) == 1
    assert isinstance(result.content[0], TextPart)
    assert result.content[0].text == "hello world"


def test_vertex_convert_parts_image():
    """
    Test vertex_convert_parts with an image inline_data.
    """
    Part = MagicMock()
    InlineData = MagicMock()
    mock_part = Part()
    mock_part.text = None
    mock_part.file_data = None
    mock_part.function_call = None

    mock_inline_data = InlineData()
    mock_inline_data.mime_type = "image/png"
    mock_inline_data.data = b"\x89PNG\r\n\x1a\n"
    mock_part.inline_data = mock_inline_data

    result = convert_message_param_to_base_message_param(Mock(parts=[mock_part]))
    assert isinstance(result, BaseMessageParam)
    assert len(result.content) == 1
    assert isinstance(result.content[0], ImagePart)
    assert result.content[0].media_type == "image/png"


def test_vertex_convert_parts_document():
    """
    Test vertex_convert_parts with a PDF document file_data.
    """
    Part = MagicMock()
    FileData = MagicMock()
    mock_part = Part()
    mock_part.text = None
    mock_part.inline_data = None
    mock_part.function_call = None

    mock_file_data = FileData()
    mock_file_data.mime_type = "application/pdf"
    mock_file_data.data = b"%PDF-1.4..."
    mock_part.file_data = mock_file_data

    result = convert_message_param_to_base_message_param(Mock(parts=[mock_part]))
    assert isinstance(result, BaseMessageParam)
    assert len(result.content) == 1
    assert isinstance(result.content[0], DocumentPart)
    assert result.content[0].media_type == "application/pdf"


def test_vertex_convert_parts_unsupported_image():
    """
    Test vertex_convert_parts with unsupported image mime type.
    """
    Part = MagicMock()
    InlineData = MagicMock()
    mock_part = Part()
    mock_part.text = None
    mock_part.file_data = None
    mock_part.function_call = None
    mock_inline_data = InlineData()
    mock_inline_data.mime_type = "image/tiff"  # not supported
    mock_inline_data.data = b"fake"
    mock_part.inline_data = mock_inline_data

    with pytest.raises(
        ValueError,
        match="Unsupported inline_data mime type: image/tiff. Cannot convert to BaseMessageParam.",
    ):
        convert_message_param_to_base_message_param(Mock(parts=[mock_part]))


def test_vertex_convert_parts_unsupported_document():
    """
    Test vertex_convert_parts with an unsupported document mime type.
    """
    Part = MagicMock()
    FileData = MagicMock()
    mock_part = Part()
    mock_part.text = None
    mock_part.inline_data = None
    mock_part.function_call = None
    mock_file_data = FileData()
    mock_file_data.mime_type = "application/msword"  # not supported
    mock_file_data.data = b"DOC..."
    mock_part.file_data = mock_file_data

    with pytest.raises(
        ValueError,
        match="Unsupported file_data mime type: application/msword. Cannot convert to BaseMessageParam.",
    ):
        convert_message_param_to_base_message_param(Mock(parts=[mock_part]))


def test_vertex_convert_parts_tool_result():
    """
    Test vertex_convert_parts with a Part containing a function_call.
    """
    Part = MagicMock()  # Mock Part class
    mock_part = Part()
    mock_part.text = None
    mock_part.inline_data = None
    mock_part.file_data = None
    function_call = MagicMock()
    function_call.name = "test"
    function_call.arguments = [("arg1", "value1"), ("arg2", "value2")]
    mock_part.function_call = function_call

    result = convert_message_param_to_base_message_param(Mock(parts=[mock_part]))
    assert isinstance(result, BaseMessageParam)
    assert result.role == "tool"
    assert len(result.content) == 1
    assert result.content[0].type == "tool_call"  # pyright: ignore [reportAttributeAccessIssue]
    assert result.content[0].name == "test"  # pyright: ignore [reportAttributeAccessIssue]


def test_vertex_convert_parts_no_supported_content():
    """
    Test vertex_convert_parts with a Part that does not have text, inline_data, file_data, or function_call.
    """
    Part = MagicMock()  # Mock Part class
    mock_part = Part()
    mock_part.text = None
    mock_part.inline_data = None
    mock_part.file_data = None
    mock_part.function_call = None

    with pytest.raises(
        ValueError,
        match="Part does not contain any supported content \\(text, image, or document\\).",
    ):
        convert_message_param_to_base_message_param(Mock(parts=[mock_part]))


def test_to_image_part_supported():
    """
    Test _to_image_part with a supported image mime type.
    """
    # This should pass without error, ensuring the "if _is_image_mime(mime)" branch is covered.
    part = _to_image_part("image/png", b"fake_image_data")
    assert part.media_type == "image/png"


def test_to_image_part_unsupported():
    """
    Test _to_image_part with an unsupported image mime type.
    """
    # This ensures the ValueError branch inside _to_image_part is covered.
    with pytest.raises(ValueError, match="Unsupported image media type: image/tiff."):
        _to_image_part("image/tiff", b"fake_image_data")


def test_to_document_part_supported():
    """
    Test _to_document_part with a supported document mime type.
    """
    # This should pass without error, ensuring the "if mime_type != 'application/pdf'" branch is covered for success.
    part = _to_document_part("application/pdf", b"fake_pdf_data")
    assert part.media_type == "application/pdf"


def test_to_document_part_unsupported():
    """
    Test _to_document_part with an unsupported document mime type.
    """
    # This ensures the ValueError branch inside _to_document_part is covered.
    with pytest.raises(
        ValueError, match="Unsupported document media type: application/msword."
    ):
        _to_document_part("application/msword", b"fake_pdf_data")


def test_is_image_mime_supported():
    """
    Test _is_image_mime with a supported image type.
    """
    assert _is_image_mime("image/png") is True


def test_is_image_mime_unsupported():
    """
    Test _is_image_mime with an unsupported image type.
    """
    assert _is_image_mime("image/tiff") is False


def test_convert_message_param_to_base_message_param_inline_data_image():
    """
    Test convert_message_param_to_base_message_param with inline_data as image.
    Covers the line:
    if _is_image_mime(mime):
        contents.append(_to_image_part(mime, data))
    """
    Part = MagicMock()
    InlineData = MagicMock()
    mock_part = Part()
    mock_part.text = None
    mock_part.file_data = None
    mock_part.function_call = None

    mock_inline_data = InlineData()
    mock_inline_data.mime_type = "image/png"
    mock_inline_data.data = b"\x89PNG\r\n\x1a\n"
    mock_part.inline_data = mock_inline_data

    result = convert_message_param_to_base_message_param(Mock(parts=[mock_part]))
    assert len(result.content) == 1
    assert isinstance(result.content[0], ImagePart)
    assert result.content[0].media_type == "image/png"


def test_convert_message_param_to_base_message_param_inline_data_pdf():
    """
    Test convert_message_param_to_base_message_param with inline_data as PDF.
    Covers the line:
    elif mime == "application/pdf":
        contents.append(_to_document_part(mime, data))
    """
    Part = MagicMock()
    InlineData = MagicMock()
    mock_part = Part()
    mock_part.text = None
    mock_part.file_data = None
    mock_part.function_call = None

    mock_inline_data = InlineData()
    mock_inline_data.mime_type = "application/pdf"
    mock_inline_data.data = b"%PDF-1.4..."
    mock_part.inline_data = mock_inline_data

    result = convert_message_param_to_base_message_param(Mock(parts=[mock_part]))
    assert len(result.content) == 1
    assert isinstance(result.content[0], DocumentPart)
    assert result.content[0].media_type == "application/pdf"


def test_convert_message_param_to_base_message_param_file_data_image():
    """
    Test convert_message_param_to_base_message_param with file_data as image.
    Covers the line:
    if _is_image_mime(mime):
        contents.append(_to_image_part(mime, data))
    """
    Part = MagicMock()
    FileData = MagicMock()
    mock_part = Part()
    mock_part.inline_data = None
    mock_part.text = None
    mock_part.function_call = None

    mock_file_data = FileData()
    mock_file_data.mime_type = "image/jpeg"
    mock_file_data.data = b"fake_jpeg_data"
    mock_part.file_data = mock_file_data

    result = convert_message_param_to_base_message_param(Mock(parts=[mock_part]))
    assert len(result.content) == 1
    assert isinstance(result.content[0], ImagePart)
    assert result.content[0].media_type == "image/jpeg"
