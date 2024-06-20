"""This module defines the base class for tools used in LLM calls."""

import inspect
from abc import abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict

from .._internal import utils


class BaseTool(BaseModel):
    """A class for defining tools for LLM calls."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def name(cls) -> str:
        """Returns the name of the tool."""
        return cls.__name__

    @classmethod
    def description(cls) -> str:
        """Returns the description of the tool."""
        return (
            inspect.cleandoc(cls.__doc__)
            if cls.__doc__
            else utils.DEFAULT_TOOL_DOCSTRING
        )

    @abstractmethod
    def call(self) -> Any:
        """The method to call the tool."""
        ...  # pragma: no cover