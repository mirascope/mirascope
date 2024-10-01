from mirascope.core import groq


def parse_recommendation(response: groq.GroqCallResponse) -> tuple[str, str]:
    title, author = response.content.split(" by ")
    return (title, author)


@groq.call("llama-3.1-8b-instant", output_parser=parse_recommendation)
def recommend_book(genre: str) -> str:
    return f"Recommend a {genre} book. Output only Title by Author"


print(recommend_book("fantasy"))
# Output: ('"The Name of the Wind"', 'Patrick Rothfuss')
