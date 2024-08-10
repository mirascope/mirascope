# LLM Validation With Retries

This recipe demonstrates how to leverage Large Language Models (LLMs) -- specifically Anthropic's Claude 3.5 Sonnet -- to perform automated validation on any value. We'll cover how to use **LLMs for complex validation tasks**, how to integrate this with Pydantic's validation system, and how to leverage [Tenacity](ADD LINK) to automatically **reinsert validation errors** back into an LLM call to **improve results**.

??? info "Key Concepts"

    - [Calls](ADD LINK)
    - [Response Model](ADD LINK)
    - [Tenacity Integration](ADD LINK)

!!! note "Background"

    While traditional validation tools like type checkers or Pydantic are limited to hardcoded rules (such as variable types or arithmetic), LLMs allow for much more nuanced and complex validation. This approach can be particularly useful for validating natural language inputs or complex data structures where traditional rule-based validation falls short.

## Setup

To set up our environment, first let's install all of the packages we will use:

```shell
pip install "mirascope[anthropic,tenacity]"
```

Make sure to also set your `ANTHROPIC_API_KEY` if you haven't already.

## Basic LLM Validation

!!! note ""

    [basic_validation.py](ADD LINK)

Let's start with a simple example of using an LLM to check for spelling and grammatical errors in a text snippet:

```python
from mirascope.core import anthropic, prompt_template
from pydantic import BaseModel, Field


class SpellingAndGrammarCheck(BaseModel):
    has_errors: bool = Field(
        description="Whether the text has typos or grammatical errors"
    )


@anthropic.call(
    model="claude-3-5-sonnet-20240620",
    response_model=SpellingAndGrammarCheck,
    json_mode=True,
)
@prompt_template(
    """
    Does the following text have any typos or grammatical errors? {text}
    """
)
def check_for_errors(text: str): ...


text = "Yestday I had a gret time!"
response = check_for_errors(text)
assert response.has_errors
```

## Pydantic's AfterValidator

!!! note ""

    [after_validator.py](ADD LINK)

We can use Pydantic's [`AfterValidator`](ADD LINK) to integrate our LLM-based validation directly into a Pydantic model:

```python
from typing import Annotated

from pydantic import BaseModel, AfterValidator, Field, ValidationError


class TextWithoutErrors(BaseModel):
    text: Annotated[
        str,
        AfterValidator(
            lambda t: t
            if not (check_for_errors(t)).has_errors
            else (_ for _ in ()).throw(ValueError("Text contains errors"))
        ),
    ]


valid = TextWithoutErrors(text="This is a perfectly written sentence.")

try:
    invalid = TextWithoutErrors(
        text="I walkd to supermarket and i picked up some fish?"
    )
except ValidationError as e:
    print(f"Validation error: {e}")
    # > Validation error: 1 validation error for TextWithoutErrors
    #   text
    #     Value error, Text contains errors [type=value_error, input_value='I walkd to supermarket a... i picked up some fish?', input_type=str]
    #       For further information visit https://errors.pydantic.dev/2.8/v/value_error
```

## Reinsert Validation Errors For Improved Performance

!!! note ""

    [validation_errors_with_retries.py](ADD LINK)

One powerful technique for enhancing LLM generations is to automatically reinsert validation errors into subsequent calls. This approach allows the LLM to learn from its previous mistakes as few-shot examples and improve it's output in real-time. We can achieve this using Mirascope's integration with Tenacity, which collects `ValidationError` messages for easy insertion into the prompt.

```python
from typing import Annotated

from mirascope.core import anthropic, prompt_template
from mirascope.integrations.tenacity import collect_errors
from pydantic import AfterValidator, BaseModel, Field, ValidationError
from tenacity import retry, stop_after_attempt


class GrammarCheck(BaseModel):
    text: Annotated[
        str,
        AfterValidator(
            lambda t: t
            if not (check_for_errors(t)).has_errors
            else (_ for _ in ()).throw(ValueError("Text still contains errors"))
        ),
    ] = Field(description="The corrected text with proper grammar")
    explanation: str = Field(description="Explanation of the corrections made")


@retry(stop=stop_after_attempt(3), after=collect_errors(ValidationError))
@anthropic.call(
    "claude-3-5-sonnet-20240620", response_model=GrammarCheck, json_mode=True
)
@prompt_template(
    """
    {previous_errors}

    Correct the grammar in the following text.
    If no corrections are needed, return the original text.
    Provide an explanation of any corrections made.

    Text: {text}
    """
)
def correct_grammar(
    text: str, *, errors: list[ValidationError] | None = None
) -> anthropic.AnthropicDynamicConfig:
    previous_errors = (
        f"Previous Errors: {errors}"
        if errors
        else "No previous errors."
    )
    return {"computed_fields": {"previous_errors": previous_errors}}


try:
    text = "I has went to the store yesterday and buyed some milk."
    result = correct_grammar(text)
    print(f"Corrected text: {result.text}")
    print(f"Explanation: {result.explanation}")
except ValidationError:
    print("Failed to correct grammar after 3 attempts")
```

!!! tip "Additional Real-World Examples"

    - Code Review: Validate code snippets for best practices and potential bugs.
    - Data Quality Checks: Validate complex data structures for consistency and completeness.
    - Legal Document Validation: Check legal documents for compliance with specific regulations.
    - Medical Record Validation: Ensure medical records are complete and consistent.
    - Financial Report Validation: Verify financial reports for accuracy and compliance with accounting standards.

When adapting this recipe to your specific use-case, consider the following:

- Tailor the prompts to provide clear instructions and relevant context for your specific validation tasks.
- Balance the trade-off between validation accuracy and performance, especially when implementing retries.
- Implement proper error handling and logging for production use.
- Consider caching validation results for frequently validated items to improve performance.