from mirascope.core import gemini


@gemini.call(model="gemini-1.5-flash", stream=True)
def recommend_book(genre: str) -> str:
    return f"Recommend a {genre} book"


try:
    for chunk, _ in recommend_book("fantasy"):
        print(chunk.content, end="", flush=True)
except Exception as e:
    print(f"Streaming Error: {str(e)}")