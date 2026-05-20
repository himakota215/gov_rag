from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# Local Ollama model
llm = ChatOllama(
    model="phi3"
)

print("Local Government Scheme RAG Chatbot Ready!")

while True:

    query = input("\nAsk your question: ")

    if query.lower() == "exit":
        break

    category = input(
        "Enter category (student/business/general) or press Enter to skip: "
    )

    # Retrieval
    if category.strip() != "":

        docs = vectorstore.similarity_search(
            query,
            k=5,
            filter={"category": category}
        )

    else:

        docs = vectorstore.similarity_search(
            query,
            k=5
        )

    # Build context
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    # Prompt
    prompt = f"""
You are a government schemes assistant.

Use ONLY the provided context.

If information is missing, say:
'I could not find this information in the documents.'

Context:
{context}

Question:
{query}
"""

    # Generate response
    response = llm.invoke(prompt)

    print("\nAnswer:\n")
    print(response.content)

    print("\nSources:\n")

    for doc in docs:

        print(doc.metadata)

        print("-" * 40)