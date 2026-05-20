from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
# Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS vector database
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# Load Groq LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"
)

print("Government Scheme RAG Chatbot Ready!")

while True:

    query = input("\nAsk your question: ")

    if query.lower() == "exit":
        break

    # Retrieve relevant chunks
    category = input("Enter category (student/business/general): ")

    docs = vectorstore.similarity_search(
    query,
    k=3,
    filter={"category": category}
    )

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a government schemes assistant.

Use ONLY the information provided in the context below.

If the answer is not present in the context, reply:
"I could not find this information in the provided documents."

Do NOT use outside knowledge.
Do NOT make up schemes or details.

Context:
{context}

Question:
{query}
"""

    # Generate response
    response = llm.invoke(prompt)

    answer = response.content

    print("\nAnswer:\n")
    print(answer)
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        category = doc.metadata.get("category", "N/A")
        print(f"Source: {source}")
        print(f"Page: {page}")
        print(f"Category: {category}")
        print("-" * 40)