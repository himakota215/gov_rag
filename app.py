import streamlit as st
from langchain_ollama import ChatOllama
from rag.retrieval import retrieve_documents, VALID_CATEGORIES

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Government Scheme Assistant",
    page_icon="🏛️",
    layout="wide"
)

st.title("🏛️ Government Scheme RAG Assistant")

st.markdown(
    """
Ask questions about:

- Government Schemes
- Scholarships
- Women Empowerment
- AI Mission
- Startup Schemes
"""
)

# ==========================================================
# SESSION STATE
# ==========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================================
# LOAD LLM
# ==========================================================

llm = ChatOllama(
    model="phi3"
)

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("Filters")

category = st.sidebar.selectbox(
    "Choose Category",
    VALID_CATEGORIES
)

# ==========================================================
# DISPLAY CHAT HISTORY
# ==========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================================================
# USER INPUT
# ==========================================================

query = st.chat_input("Ask your question...")

if query:

    # ----------------------------
    # Store User Message
    # ----------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    # ======================================================
    # RETRIEVAL
    # ======================================================

    docs = retrieve_documents(
        query=query,
        category=category
    )

    # ======================================================
    # CHECK IF NO DOCUMENTS FOUND
    # ======================================================

    if len(docs) == 0:

        answer = "I could not find this information in the documents."

    else:

        # ==================================================
        # BUILD CONTEXT
        # ==================================================

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        # ==================================================
        # PROMPT
        # ==================================================

        prompt = f"""
You are an AI Government Scheme Assistant.

Answer ONLY from the given context.

If the answer is not present, reply exactly:

"I could not find this information in the documents."

Do NOT use outside knowledge.

Context:
{context}

Question:
{query}

Answer:
"""

        # ==================================================
        # LLM RESPONSE
        # ==================================================

        response = llm.invoke(prompt)

        answer = response.content

    # ======================================================
    # STORE RESPONSE
    # ======================================================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    # ======================================================
    # DISPLAY RESPONSE
    # ======================================================

    with st.chat_message("assistant"):

        st.markdown(answer)

        if len(docs) > 0:

            st.markdown("---")
            st.subheader("📄 Sources")

            for i, doc in enumerate(docs, start=1):

                filename = doc.metadata.get("filename", "Unknown")

                page = doc.metadata.get("page", "N/A")

                category_name = doc.metadata.get(
                    "category",
                    "N/A"
                )

                score = doc.metadata.get(
                    "score",
                    "BM25"
                )

                with st.expander(f"Source {i}"):

                    st.write(f"**File:** {filename}")

                    st.write(f"**Page:** {page}")

                    st.write(f"**Category:** {category_name}")

                    st.write(f"**Score:** {score}")

                    st.write("**Preview:**")

                    st.write(doc.page_content[:500] + "...")