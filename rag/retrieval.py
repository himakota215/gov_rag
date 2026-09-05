import os
import json

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from rank_bm25 import BM25Okapi


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")
CHUNKS_PATH = os.path.join(BASE_DIR, "chunks.json")


# =========================================================
# SETTINGS
# =========================================================

FETCH_K = 10
TOP_K = 5
RRF_K = 60


# =========================================================
# LOAD EMBEDDINGS
# =========================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =========================================================
# LOAD FAISS
# =========================================================

vectorstore = FAISS.load_local(
    INDEX_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)


# =========================================================
# LOAD CHUNKS
# =========================================================

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks_data = json.load(f)


# =========================================================
# SIMPLE TOKENIZER
# =========================================================

def tokenize(text):
    return text.lower().split()


# =========================================================
# PREPARE BM25
# =========================================================

bm25_documents = []

for item in chunks_data:
    text = item.get("page_content", "")
    bm25_documents.append(tokenize(text))

bm25 = BM25Okapi(bm25_documents)


# =========================================================
# DOCUMENT CLASS
# =========================================================

class RetrievedDocument:

    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


# =========================================================
# CREATE UNIQUE DOCUMENT ID
# =========================================================

def get_document_id(doc):
    filename = doc.metadata.get("filename", "")
    page = doc.metadata.get("page", "")

    chunk_id = doc.metadata.get(
        "chunk_id",
        doc.metadata.get("id", "")
    )

    if not chunk_id:
        content = getattr(doc, "page_content", "")
        chunk_id = hash(content)

    return (filename, page, chunk_id)


# =========================================================
# BM25 RETRIEVAL
# =========================================================

def get_bm25_documents(query, k=FETCH_K):

    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    results = []

    for index in ranked_indices[:k]:

        item = chunks_data[index]

        metadata = dict(
            item.get("metadata", {})
        )

        # chunks.json stores id separately
        metadata["chunk_id"] = item.get(
            "id",
            index
        )

        results.append(
            RetrievedDocument(
                page_content=item.get(
                    "page_content",
                    ""
                ),
                metadata=metadata
            )
        )

    return results


# =========================================================
# CATEGORY FILTER
# =========================================================

def filter_by_category(documents, category):

    if not category or category == "all":
        return documents

    filtered = []

    for doc in documents:

        doc_category = doc.metadata.get(
            "category",
            "general"
        )

        if doc_category == category:
            filtered.append(doc)

    return filtered


# =========================================================
# FAISS RETRIEVAL
# =========================================================

def faiss_search(
    query,
    k=FETCH_K,
    category="all"
):

    results = vectorstore.similarity_search(
        query,
        k=k
    )

    # Add chunk IDs to FAISS metadata by matching
    # filename + page + content with chunks.json.
    for doc in results:

        if "chunk_id" not in doc.metadata:

            filename = doc.metadata.get(
                "filename",
                ""
            )

            page = doc.metadata.get(
                "page",
                ""
            )

            content = doc.page_content

            for index, item in enumerate(chunks_data):

                item_metadata = item.get(
                    "metadata",
                    {}
                )

                if (
                    item_metadata.get(
                        "filename",
                        ""
                    ) == filename
                    and
                    item_metadata.get(
                        "page",
                        ""
                    ) == page
                    and
                    item.get(
                        "page_content",
                        ""
                    ) == content
                ):

                    doc.metadata["chunk_id"] = item.get(
                        "id",
                        index
                    )

                    break

    results = filter_by_category(
        results,
        category
    )

    return results


# =========================================================
# BM25 SEARCH
# =========================================================

def bm25_search(
    query,
    k=FETCH_K,
    category="all"
):

    # Retrieve the full BM25 ranking first so that
    # category filtering does not shrink the candidate pool.
    raw_results = get_bm25_documents(
        query,
        k=len(chunks_data)
    )

    documents = filter_by_category(
        raw_results,
        category
    )

    return documents[:k]


# =========================================================
# RECIPROCAL RANK FUSION
# =========================================================

def reciprocal_rank_fusion(
    result_lists,
    rrf_k=RRF_K
):
    """
    RRF(d) = sum of 1 / (rrf_k + rank)
    where rank starts at 1.
    """

    scores = {}
    documents = {}

    for results in result_lists:

        for rank, doc in enumerate(
            results,
            start=1
        ):

            doc_id = get_document_id(doc)

            score = 1 / (rrf_k + rank)

            scores[doc_id] = (
                scores.get(doc_id, 0) + score
            )

            documents[doc_id] = doc

    ranked_ids = sorted(
        scores.keys(),
        key=lambda x: scores[x],
        reverse=True
    )

    return [
        documents[doc_id]
        for doc_id in ranked_ids
    ]


# =========================================================
# HYBRID RRF SEARCH
# =========================================================

def hybrid_search(
    query,
    top_k=TOP_K,
    category="all"
):

    faiss_results = faiss_search(
        query=query,
        k=FETCH_K,
        category=category
    )

    bm25_results = bm25_search(
        query=query,
        k=FETCH_K,
        category=category
    )

    fused_results = reciprocal_rank_fusion(
        [
            faiss_results,
            bm25_results
        ],
        rrf_k=RRF_K
    )

    return fused_results[:top_k]


# =========================================================
# MAIN RETRIEVAL FUNCTION
# =========================================================

def retrieve(
    query,
    top_k=TOP_K,
    category="all"
):

    return hybrid_search(
        query=query,
        top_k=top_k,
        category=category
    )


# =========================================================
# COMPATIBILITY FUNCTION
# =========================================================

def retrieve_documents(
    query,
    k=TOP_K,
    category="all"
):
    """
    Compatibility wrapper for the existing evaluate.py.

    evaluate.py calls:

        retrieve_documents(
            question,
            category=query_category,
            k=top_k
        )

    Internally this uses:

        FAISS + BM25 + RRF
    """

    return retrieve(
        query=query,
        top_k=k,
        category=category
    )


# =========================================================
# DEBUG FUNCTION
# =========================================================

def print_results(
    query,
    top_k=TOP_K,
    category="all"
):

    results = retrieve_documents(
        query=query,
        k=top_k,
        category=category
    )

    print("\n" + "=" * 70)

    print("QUERY:")
    print(query)

    print("\nCATEGORY:", category)

    print("\nRRF RESULTS:")

    for rank, doc in enumerate(
        results,
        start=1
    ):

        print(f"\nRank {rank}")

        print(
            "File:",
            doc.metadata.get(
                "filename",
                "Unknown"
            )
        )

        print(
            "Page:",
            doc.metadata.get(
                "page",
                "Unknown"
            )
        )

        print(
            "Category:",
            doc.metadata.get(
                "category",
                "Unknown"
            )
        )

        print(
            "Text:",
            doc.page_content[:300].replace(
                "\n",
                " "
            )
        )

    print("\n" + "=" * 70)

def debug_retrieval(query, category="all", k=10):

    print("\n" + "=" * 80)
    print("DEBUG RETRIEVAL")
    print("=" * 80)

    print("\nQUERY:")
    print(query)

    print("\nCATEGORY:")
    print(category)

    # -----------------------------------------------------
    # FAISS
    # -----------------------------------------------------

    faiss_results = faiss_search(
        query=query,
        k=k,
        category=category
    )

    print("\n" + "-" * 80)
    print("FAISS RESULTS")
    print("-" * 80)

    for rank, doc in enumerate(
        faiss_results,
        start=1
    ):

        print(
            f"{rank}. "
            f"{doc.metadata.get('filename', 'Unknown')} "
            f"| page={doc.metadata.get('page', 'Unknown')}"
        )

        print(
            "   ",
            doc.page_content[:200]
            .replace("\n", " ")
        )

    # -----------------------------------------------------
    # BM25
    # -----------------------------------------------------

    bm25_results = bm25_search(
        query=query,
        k=k,
        category=category
    )

    print("\n" + "-" * 80)
    print("BM25 RESULTS")
    print("-" * 80)

    for rank, doc in enumerate(
        bm25_results,
        start=1
    ):

        print(
            f"{rank}. "
            f"{doc.metadata.get('filename', 'Unknown')} "
            f"| page={doc.metadata.get('page', 'Unknown')}"
        )

        print(
            "   ",
            doc.page_content[:200]
            .replace("\n", " ")
        )

    # -----------------------------------------------------
    # RRF
    # -----------------------------------------------------

    fused_results = reciprocal_rank_fusion(
        [
            faiss_results,
            bm25_results
        ]
    )

    print("\n" + "-" * 80)
    print("RRF RESULTS")
    print("-" * 80)

    for rank, doc in enumerate(
        fused_results[:k],
        start=1
    ):

        print(
            f"{rank}. "
            f"{doc.metadata.get('filename', 'Unknown')} "
            f"| page={doc.metadata.get('page', 'Unknown')}"
        )

        print(
            "   ",
            doc.page_content[:200]
            .replace("\n", " ")
        )

    print("\n" + "=" * 80)

# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    debug_retrieval(
        "PM-USP Central Sector Scheme scholarship college university students income",
        category="all",
        k=10
    )