from groq import GroqError
from mirascope.core import groq


@groq.call(model="llama-3.1-8b-instant", stream=True)
def recommend_book(genre: str) -> str:
    return f"Recommend a {genre} book"


try:
    for chunk, _ in recommend_book("fantasy"):
        print(chunk.content, end="", flush=True)
except GroqError as e:
    print(f"Streaming Error: {str(e)}")
