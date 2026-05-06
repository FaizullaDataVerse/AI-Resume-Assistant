import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.vectorstores import Chroma

# ------------------ CONFIG ------------------
st.set_page_config(page_title="AI Resume Assistant (RAG)", page_icon="🤖", layout="wide")

# ------------------ LOAD API ------------------
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    st.error("❌ Add MISTRAL_API_KEY in .env file")
    st.stop()

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("Upload your resume and start analyzing 🚀")

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat = []

# ------------------ TITLE ------------------
st.markdown("<h1 style='text-align:center;'>🤖 AI Resume Assistant (RAG)</h1>", unsafe_allow_html=True)
st.markdown("---")

# ------------------ SESSION ------------------
if "chat" not in st.session_state:
    st.session_state.chat = []

if "db" not in st.session_state:
    st.session_state.db = None

if "ats_score" not in st.session_state:
    st.session_state.ats_score = None

# ------------------ FILE UPLOAD ------------------
col1, col2 = st.columns([2,1])

with col1:
    file = st.file_uploader("📤 Upload Resume (PDF)", type=["pdf"])

with col2:
    if st.session_state.ats_score:
        st.metric("📊 ATS Score", f"{st.session_state.ats_score}/100")

# ------------------ PROCESS FILE ------------------
if file:
    with open("resume.pdf", "wb") as f:
        f.write(file.read())

    data = PyPDFLoader("resume.pdf").load()

    docs = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=50
    ).split_documents(data)

    embedding = MistralAIEmbeddings()
    db = Chroma.from_documents(docs, embedding)

    st.session_state.db = db
    st.success("✅ Resume processed successfully!")

# ------------------ LLM ------------------
llm = ChatMistralAI(
    model="mistral-small-2506",
    max_tokens=250
)

# ------------------ CHAT ------------------
if st.session_state.db:

    query = st.chat_input("💬 Ask about your resume...")

    if query:
        retriever = st.session_state.db.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(query)
        context = "\n\n".join([d.page_content for d in docs])

        # First vs next response
        if len(st.session_state.chat) == 0:
            instruction = "Analyze resume: give ATS score, strengths, weaknesses, improvements, and ask 3 questions.,if the question is not related to the resume," \
            "answer the question directly as you are a helpful assistant."
        else:
            instruction = "Improve resume and answer briefly.,if the question is not related to the resume," \
            "answer the question directly as you are a helpful assistant."

        prompt = f"{instruction}\n\nResume:\n{context}\n\nUser:\n{query}"

        response = llm.invoke(prompt).content

        # 👉 Extract ATS score (simple logic)
        if "ATS" in response and st.session_state.ats_score is None:
            import re
            match = re.search(r"\b(\d{2,3})\b", response)
            if match:
                st.session_state.ats_score = match.group(1)

        st.session_state.chat.append((query, response))

# ------------------ CHAT DISPLAY ------------------
for q, a in st.session_state.chat:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center;'>Built with ❤️ using LangChain + Mistralai</p>",
    unsafe_allow_html=True
)