from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS index
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

print("FAISS index loaded successfully!")

while True:
    query = input("\nAsk a question: ")

    if query.lower() == "exit":
        break

    # Retrieve similar chunks
    docs = vectorstore.similarity_search(query, k=3)

    print("\nTop Retrieved Chunks:\n")

    for i, doc in enumerate(docs, 1):
        print(f"Result {i}:\n")
        print(doc.page_content)
        print("\n" + "-"*50)