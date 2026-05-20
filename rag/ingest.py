from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

documents = []

data_path = "../data"

# Load PDFs
for file in os.listdir(data_path):

    if file.endswith(".pdf"):

        loader = PyPDFLoader(os.path.join(data_path, file))

        docs = loader.load()

        # Add metadata BEFORE chunking
        for doc in docs:

            source = doc.metadata.get("source", "").lower()

            if "scholarship" in source:
                doc.metadata["category"] = "student"

            elif "mudra" in source:
                doc.metadata["category"] = "business"

            else:
                doc.metadata["category"] = "general"

        documents.extend(docs)

print("Documents loaded:", len(documents))

# Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print("Chunks created:", len(chunks))

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create FAISS index
vectorstore = FAISS.from_documents(chunks, embeddings)

vectorstore.save_local("faiss_index")

print("FAISS index saved successfully!")