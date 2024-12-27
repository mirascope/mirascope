from mirascope.core import BaseMessageParam
from mirascope import llm


@llm.call(provider="bedrock", model="anthropic.claude-3-haiku-20240307-v1:0")
def recommend_book(genre: str) -> list[BaseMessageParam]:
    return [BaseMessageParam(role="user", content=f"Recommend a {genre} book")]


response = recommend_book("fantasy")
print(response.content)