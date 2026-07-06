from app.llm.client import client

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain RAG in one paragraph."
)

print(response.text)
