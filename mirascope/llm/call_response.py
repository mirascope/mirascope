from __future__ import annotations

from typing import Any, Generic, TypeVar

from mirascope.core import BaseDynamicConfig
from mirascope.core.base import (
    BaseCallParams,
    BaseCallResponse,
    BaseMessageParam,
    BaseTool,
)
from mirascope.core.base.types import FinishReason
from mirascope.llm._response_metaclass import _ResponseMetaclass
from mirascope.llm.tool import Tool

_ResponseT = TypeVar("_ResponseT")


class CallResponse(
    BaseCallResponse[
        _ResponseT,
        BaseTool,
        Any,
        BaseDynamicConfig[Any, Any, Any],
        BaseMessageParam,
        BaseCallParams,
        BaseMessageParam,
    ],
    Generic[_ResponseT],
    metaclass=_ResponseMetaclass,
):
    """
    A provider-agnostic CallResponse class.

    We rely on _response having `common_` methods or properties for normalization.
    """

    _response: BaseCallResponse[_ResponseT, Any, Any, Any, Any, Any, Any]

    def __init__(
        self,
        response: BaseCallResponse[_ResponseT, Any, Any, Any, Any, Any, Any],
    ) -> None:
        super().__init__(
            **{field: getattr(response, field) for field in response.model_fields}
        )
        object.__setattr__(self, "_response", response)

    def __getattribute__(self, name: str) -> Any:  # noqa: ANN401
        special_names = {
            "_response",
            "__dict__",
            "__class__",
            "model_fields",
            "__annotations__",
            "__pydantic_validator__",
            "__pydantic_fields_set__",
            "__pydantic_extra__",
            "__pydantic_private__",
            "__class_getitem__",
            "__repr__",
            "__str__",
            "_properties",
        } | set(object.__getattribute__(self, "_properties"))

        if name in special_names:
            return object.__getattribute__(self, name)

        try:
            response = object.__getattribute__(self, "_response")
            return getattr(response, name)
        except AttributeError:
            return object.__getattribute__(self, name)

    def __str__(self) -> str:
        return str(self._response)

    @property
    def finish_reasons(self) -> list[FinishReason] | None:  # pyright: ignore [reportIncompatibleMethodOverride]
        return self._response.common_finish_reasons

    @property
    def message_param(self) -> BaseMessageParam:
        return self._response.common_message_param

    @property
    def tools(self) -> list[Tool] | None:  # pyright: ignore [reportIncompatibleMethodOverride]
        return self._response.common_tools

    @property
    def tool(self) -> Tool | None:
        tools = self._response.common_tools
        if tools:
            return tools[0]
        return None
