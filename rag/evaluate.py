"""
Retrieval evaluation for the gov_rag project.

Location: place this file in rag/, alongside retrieval.py, chatbot.py,
chunks.json, and faiss_index/ -- it imports retrieve_documents directly
from retrieval.py, so it must live in the same folder to avoid import
path issues.

Run from inside rag/:
    python evaluate.py

What this does:
Builds a small hand-labeled test set (question -> expected source PDF
filename) and checks whether retrieve_documents() actually surfaces a
chunk from that file in its results. Reports Precision@1 and Recall@k.

Update TEST_SET below to match the actual PDFs you have in data/ --
the filenames must match exactly what shows up in doc.metadata["filename"]
after ingestion (check chunks.json if unsure).
"""

from retrieval import retrieve_documents

# ---------------------------------------------------------------------
# TEST SET
# ---------------------------------------------------------------------
# Each entry: (question, expected_filename, expected_category)
# expected_category should match what detect_category() in ingest.py
# assigns for that document -- update this list to match your real PDFs.
#
# NOTE: adjust these filenames/categories to match your actual data/
# folder exactly (see the filenames in your chunks.json).

TEST_SET = [
    ("what is the income limit for NMMSS scholarship",
     "NMMSS Department of School Education & LiteracyGuidelines.pdf", "general_merit"),

    ("scholarship for class 9 students economically weaker",
     "NMMSS Department of School Education & LiteracyGuidelines.pdf", "general_merit"),

    ("income limit for post matric scholarship for college students",
     "NSP_SCHEME OF SCHOLARSHIP FOR COLLEGE ANDguidelines.pdf", "general_merit"),

    ("frequently asked questions central sector scholarship scheme",
     "CSSS_FAQS.pdf", "general_merit"),

    ("scholarship eligibility for students with disability",
     "DEPDFAQ.pdf", "disability"),

    ("documents required for disability scholarship",
     "DEPDFAQ.pdf", "disability"),

    ("post matric scholarship for scheduled caste students",
     "PMS_for_SCs_Scheme_Guidelines.pdf", "sc_st_obc"),

    ("income limit for SC students scholarship",
     "PMS_for_SCs_Scheme_Guidelines.pdf", "sc_st_obc"),
]


def evaluate(top_k=5, category_mode="all"):
    """
    category_mode:
      "all"     -> run every query with category="all" (tests raw retrieval)
      "correct" -> run every query with its own expected category
                   (tests whether category filtering helps or hurts)
    """
    hits_at_1 = 0
    hits_at_k = 0
    total = len(TEST_SET)

    print(f"Running evaluation | top_k={top_k} | category_mode={category_mode}\n")

    for question, expected_filename, expected_category in TEST_SET:

        query_category = expected_category if category_mode == "correct" else "all"

        docs = retrieve_documents(question, category=query_category, k=top_k)

        retrieved_filenames = [d.metadata.get("filename", "") for d in docs]

        top1_correct = (
            len(retrieved_filenames) > 0
            and retrieved_filenames[0] == expected_filename
        )
        topk_correct = expected_filename in retrieved_filenames

        hits_at_1 += int(top1_correct)
        hits_at_k += int(topk_correct)

        status = "PASS" if topk_correct else "FAIL"
        print(f"[{status}] {question}")
        print(f"        expected     : {expected_filename}")
        print(f"        top-1        : {retrieved_filenames[0] if retrieved_filenames else 'NONE'}")
        print(f"        in top-{top_k}     : {retrieved_filenames}")
        print()

    print("=" * 60)
    print(f"Precision@1        : {hits_at_1}/{total} = {hits_at_1/total:.0%}")
    print(f"Recall@{top_k}           : {hits_at_k}/{total} = {hits_at_k/total:.0%}")


if __name__ == "__main__":
    print("### Run 1: no category filter (category='all') ###\n")
    evaluate(category_mode="all")

    print("\n\n### Run 2: using the correct category filter per question ###\n")
    evaluate(category_mode="correct")