import os
import json
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

import fitz
import pytesseract

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data")
CHUNKS_PATH = os.path.join(BASE_DIR, "chunks.json")
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")

tesseract_path = (
    os.getenv("TESSERACT_PATH")
    or shutil.which("tesseract")
    or r"C:\Tesseract-OCR\tesseract.exe"
)

if os.path.exists(tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    raise RuntimeError(
        "Tesseract not found. Install it or set TESSERACT_PATH to the binary location."
    )


# =====================================================
# LOAD EMBEDDINGS
# =====================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =====================================================
# CATEGORY DETECTION  (updated: education sub-categories)
# =====================================================
# All current docs are education schemes, so a flat "student" category for
# everything gives you no real filtering power. Instead this detects WHICH
# kind of education scheme each page is about, so you can filter queries
# like "SC scholarships" vs "disability scholarships" vs "general merit".
#
# Order matters: more specific categories are checked before the generic
# fallback. A page can only match one category here (first match wins) --
# if you find pages matching multiple groups incorrectly, that's a sign the
# keyword lists need tightening, worth checking during your eval.

def detect_category(text):

    text = text.lower()

    if any(word in text for word in [
        "scheduled caste", "sc student", "sc/st", " sc ", "scheduled tribe",
        "st student", "obc", "other backward class"
    ]):
        return "sc_st_obc"

    elif any(word in text for word in [
        "disability", "disabled", "divyang", "pwd", "differently abled"
    ]):
        return "disability"

    elif any(word in text for word in [
        "girl", "women", "female", "empowerment"
    ]):
        return "girls_women"

    elif any(word in text for word in [
        "minority", "muslim", "christian", "sikh", "buddhist", "parsi", "jain"
    ]):
        return "minority"

    elif any(word in text for word in [
        "scholarship", "student", "education", "college", "school", "nsp",
        "merit", "means-cum-merit"
    ]):
        return "general_merit"

    return "general"


# =====================================================
# DATA QUALITY CHECK
# =====================================================
# Government PDFs/pages are sometimes near-empty (scanned images with no
# extractable text, boilerplate cover pages, blank tables). Silently
# ingesting these adds noise to your vector store. This flags them loudly
# during ingestion instead of you discovering it later while debugging a
# bad retrieval result.

MIN_CHARS_PER_PAGE = 40

def is_low_quality(text):
    stripped = text.strip()
    return len(stripped) < MIN_CHARS_PER_PAGE
def ocr_page(pdf_path, page_number):
    """
    OCR a single PDF page when normal PDF text extraction fails.
    page_number is zero-based.
    """

    pdf = fitz.open(pdf_path)

    page = pdf[page_number]

    # Render page as image
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))

    image = pix.tobytes("png")

    from PIL import Image
    from io import BytesIO

    img = Image.open(BytesIO(image))

    text = pytesseract.image_to_string(img)

    pdf.close()

    return text


# =====================================================
# LOAD DOCUMENTS
# =====================================================

documents = []
skipped_pages = []

print("\nLoading PDFs...\n")

for file in os.listdir(DATA_PATH):

    if not file.endswith(".pdf"):
        continue

    print("Loading :", file)

    loader = PyPDFLoader(
        os.path.join(DATA_PATH, file)
    )

    pages = loader.load()
    for page_index, page in enumerate(pages):
        text = page.page_content
        page_number = page_index + 1
        page.metadata["filename"] = file
        page.metadata["page"] = page_number

        # ---------------------------------------------
        # NORMAL TEXT EXTRACTION
        # ---------------------------------------------

        if not is_low_quality(text):

            page.metadata["category"] = detect_category(text)

            documents.append(page)

        # ---------------------------------------------
        # OCR FALLBACK
        # ---------------------------------------------

        else:

            print(
                f"Low text detected: {file} | page {page_number}"
            )

            print("Running OCR...")

            pdf_path = os.path.join(DATA_PATH, file)

            ocr_text = ocr_page(
                pdf_path,
                page_index
            )

            # -----------------------------------------
            # Check OCR result
            # -----------------------------------------

            if is_low_quality(ocr_text):

                skipped_pages.append(
                    (file, page_number)
                )

                print(
                    f"OCR also failed: {file} | page {page_number}"
                )

                continue

            # -----------------------------------------
            # Replace empty PDF text with OCR text
            # -----------------------------------------

            page.page_content = ocr_text

            page.metadata["filename"] = file
            page.metadata["page"] = page_number

            page.metadata["category"] = detect_category(
                ocr_text
            )

            documents.append(page)

        print(
            f"OCR successful: {file} | page {page_number}"
        )

print("\nDocuments Loaded :", len(documents))

if skipped_pages:
    print("\nSkipped", len(skipped_pages), "low-content page(s):")
    for filename, page_num in skipped_pages:
        print("  -", filename, "| page", page_num)


# =====================================================
# CHUNKING
# =====================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

print("\nChunks Created :", len(chunks))


# =====================================================
# CATEGORY SUMMARY  (new)
# =====================================================
# Quick sanity check printed at ingest time -- lets you see at a glance
# whether categories are distributed as expected (e.g. catch a whole PDF
# falling into "general" when it shouldn't).

category_counts = {}
for chunk in chunks:
    cat = chunk.metadata.get("category", "unknown")
    category_counts[cat] = category_counts.get(cat, 0) + 1

print("\nCategory Breakdown:")
for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
    print(f"  {cat:15s} : {count} chunks")


# =====================================================
# SAVE CHUNKS
# =====================================================

chunk_list = []

for i, chunk in enumerate(chunks):

    chunk_list.append(

        {
            "id": i,
            "page_content": chunk.page_content,
            "metadata": chunk.metadata
        }

    )

with open(CHUNKS_PATH, "w", encoding="utf-8") as f:

    json.dump(
        chunk_list,
        f,
        ensure_ascii=False,
        indent=4
    )

print("\nchunks.json saved.")


# =====================================================
# CREATE FAISS
# =====================================================

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

vectorstore.save_local(INDEX_PATH)

print("FAISS Index Saved Successfully.")

print("\nIngestion Completed.")