"""The Mirascope Azure Module."""

from typing import TypeAlias

from azure.ai.inference.models import ChatRequestMessage

from ..base import BaseMessageParam
from ._call import azure_call
from ._call import azure_call as call
from .call_params import AzureCallParams
from .call_response import AzureCallResponse
from .call_response_chunk import AzureCallResponseChunk
from .dynamic_config import AsyncAzureDynamicConfig, AzureDynamicConfig
from .models import AzureModels
from .stream import AzureStream
from .tool import AzureTool, AzureToolConfig

AzureMessageParam: TypeAlias = ChatRequestMessage | BaseMessageParam

__all__ = [
    "AsyncAzureDynamicConfig",
    "call",
    "AzureDynamicConfig",
    "AzureCallParams",
    "AzureCallResponse",
    "AzureCallResponseChunk",
    "AzureMessageParam",
    "AzureModels",
    "AzureStream",
    "AzureTool",
    "AzureToolConfig",
    "azure_call",
]
