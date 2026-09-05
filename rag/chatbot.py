from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from retrieval import retrieve_documents, VALID_CATEGORIES
import os

# -----------------------------
# Load Environment Variables
# -----------------------------
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

# -----------------------------
# Load Embedding Model
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# Load FAISS Vector Store
# -----------------------------
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# -----------------------------
# Load LLM
# -----------------------------
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"
)

print("===========================================")
print(" Government Scheme RAG Assistant is Ready ")
print(" Type 'exit' to quit.")
print("===========================================")

# -----------------------------
# Chat Loop
# -----------------------------
while True:

    query = input("\nAsk your question: ")

    if query.lower() == "exit":
        break

    # Category list now matches ingest.py's detect_category() output
    # (education sub-categories), imported from retrieval.py so both
    # files can't drift out of sync.
    category = input(
        f"Enter category ({'/'.join(VALID_CATEGORIES)}): "
    ).strip()

    if category not in VALID_CATEGORIES:
        print(f"\nUnrecognized category '{category}', defaulting to 'all'.")
        category = "all"

    # -----------------------------
    # Retrieve Documents
    # -----------------------------
    docs = retrieve_documents(
        query=query,
        category=category,
        k=5
    )

    # -----------------------------
    # No Relevant Documents Found
    # -----------------------------
    if len(docs) == 0:

        print("\nI could not find this information in the provided documents.")
        continue

    # -----------------------------
    # Build Context
    # -----------------------------
    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    # -----------------------------
    # Prompt
    # -----------------------------
    prompt = f"""
You are a government schemes assistant.

Use ONLY the information provided in the context below.

If the answer is not present in the context, reply exactly:

"I could not find this information in the provided documents."

Do NOT use outside knowledge.
Do NOT answer unrelated questions.
Do NOT solve mathematical problems.
Do NOT make up schemes or details.

Context:
{context}

Question:
{query}
"""

    # -----------------------------
    # Generate Response
    # -----------------------------
    response = llm.invoke(prompt)

    print("\nAnswer:\n")
    print(response.content)

    # -----------------------------
    # Sources
    # -----------------------------
    print("\nSources:\n")

    for doc in docs:

        print("Source     :", doc.metadata.get("source"))

        print("Page       :", doc.metadata.get("page"))

        print("Category   :", doc.metadata.get("category"))

        score = doc.metadata.get("score")
        print("Score      :", score if score is not None else "N/A (BM25 match)")

        print("Match Type :", doc.metadata.get("match_type", "unknown"))

        print("-" * 60)