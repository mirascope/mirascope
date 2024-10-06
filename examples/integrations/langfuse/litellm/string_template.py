from mirascope.core import litellm, prompt_template
from mirascope.integrations.langfuse import with_langfuse


@with_langfuse()
@litellm.call("gpt-4o-mini")
@prompt_template("Recommend a {genre} book")
def recommend_book(genre: str): ...


print(recommend_book("fantasy"))