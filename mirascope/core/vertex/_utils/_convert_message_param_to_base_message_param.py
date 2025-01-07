from google.cloud.aiplatform_v1beta1.types.content import FileData
from vertexai.generative_models import Content

from mirascope.core import BaseMessageParam
from mirascope.core.base import DocumentPart, ImagePart, TextPart
from mirascope.core.base.message_param import ToolCallPart


def _is_image_mime(mime_type: str) -> bool:
    return mime_type in ["image/jpeg", "image/png", "image/gif", "image/webp"]


def _to_image_part(mime_type: str, data: bytes) -> ImagePart:
    if not _is_image_mime(mime_type):
        raise ValueError(
            f"Unsupported image media type: {mime_type}. "
            "Expected one of: image/jpeg, image/png, image/gif, image/webp."
        )
    return ImagePart(type="image", media_type=mime_type, image=data, detail=None)


def _to_document_part(mime_type: str, data: bytes) -> DocumentPart:
    if mime_type != "application/pdf":
        raise ValueError(
            f"Unsupported document media type: {mime_type}. "
            "Only application/pdf is supported."
        )
    return DocumentPart(type="document", media_type=mime_type, document=data)


def convert_message_param_to_base_message_param(
    message: Content,
) -> BaseMessageParam:
    """Converts a Part to a BaseMessageParam."""
    role: str = "assistant"
    contents = []
    has_tool_call = False
    for part in message.parts:
        if part.text:
            contents.append(TextPart(type="text", text=part.text))

        elif part.inline_data:
            blob = part.inline_data
            mime = blob.mime_type
            data = blob.data
            if _is_image_mime(mime):
                contents.append(_to_image_part(mime, data))
            elif mime == "application/pdf":
                contents.append(_to_document_part(mime, data))
            else:
                raise ValueError(
                    f"Unsupported inline_data mime type: {mime}. Cannot convert to BaseMessageParam."
                )

        elif part.file_data:
            file_data: FileData = part.file_data
            mime = file_data.mime_type
            data = file_data.data
            if _is_image_mime(mime):
                contents.append(_to_image_part(mime, data))
            elif mime == "application/pdf":
                contents.append(_to_document_part(mime, data))
            else:
                raise ValueError(
                    f"Unsupported file_data mime type: {mime}. Cannot convert to BaseMessageParam."
                )
        elif part.function_call:
            contents.append(
                ToolCallPart(
                    type="tool_call",
                    name=part.function_call.name,
                    args=dict(part.function_call.arguments),
                )
            )
            has_tool_call = True
        else:
            raise ValueError(
                "Part does not contain any supported content (text, image, or document)."
            )

    if len(contents) == 1 and isinstance(contents[0], TextPart):
        return BaseMessageParam(role=role, content=contents)

    return BaseMessageParam(role="tool" if has_tool_call else role, content=contents)
