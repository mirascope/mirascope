"""The `MistralStream` class for convenience around streaming LLM calls."""

from mistralai.models.chat_completion import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    FinishReason,
)
from mistralai.models.common import UsageInfo

from ..base._stream import BaseStream
from ._utils import calculate_cost
from .call_params import MistralCallParams
from .call_response import MistralCallResponse
from .call_response_chunk import MistralCallResponseChunk
from .dynamic_config import MistralDynamicConfig
from .tool import MistralTool


class MistralStream(
    BaseStream[
        MistralCallResponse,
        MistralCallResponseChunk,
        ChatMessage,
        ChatMessage,
        ChatMessage,
        ChatMessage,
        MistralTool,
        MistralDynamicConfig,
        MistralCallParams,
        FinishReason,
    ]
):
    _provider = "mistral"

    @property
    def cost(self) -> float | None:
        """Returns the cost of the call."""
        return calculate_cost(self.input_tokens, self.output_tokens, self.model)

    def _construct_message_param(
        self, tool_calls: list | None = None, content: str | None = None
    ) -> ChatMessage:
        message_param = ChatMessage(
            role="assistant", content=content if content else "", tool_calls=tool_calls
        )
        return message_param

    def construct_call_response(self) -> MistralCallResponse:
        """Constructs the call response from a consumed MistralStream."""
        if not hasattr(self, "message_param"):
            raise ValueError(
                "No stream response, check if the stream has been consumed."
            )
        usage = UsageInfo(
            prompt_tokens=int(self.input_tokens or 0),
            completion_tokens=int(self.output_tokens or 0),
            total_tokens=int(self.input_tokens or 0) + int(self.output_tokens or 0),
        )
        completion = ChatCompletionResponse(
            id=self.id if self.id else "",
            choices=[
                ChatCompletionResponseChoice(
                    finish_reason=self.finish_reasons[0]
                    if self.finish_reasons
                    else None,
                    index=0,
                    message=self.message_param,
                )
            ],
            created=0,
            model=self.model,
            object="",
            usage=usage,
        )
        return MistralCallResponse(
            metadata=self.metadata,
            response=completion,
            tool_types=self.tool_types,
            prompt_template=self.prompt_template,
            fn_args=self.fn_args if self.fn_args else {},
            dynamic_config=self.dynamic_config,
            messages=self.messages,
            call_params=self.call_params,
            call_kwargs=self.call_kwargs,
            user_message_param=self.user_message_param,
            start_time=self.start_time,
            end_time=self.end_time,
        )