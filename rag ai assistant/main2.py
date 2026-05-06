import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.vectorstores import Chroma

# -------- Load API Key --------
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    raise ValueError("❌ API key not found")

# -------- Load Resume --------
pdf_path = r"C:\Users\princ\OneDrive\Desktop\Faijulla_Shabbir_Alas_Resume.pdf"
data = PyPDFLoader(pdf_path).load()

# -------- Split Text --------
docs = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
).split_documents(data)

# -------- Create Vector DB --------
embedding = MistralAIEmbeddings(api_key=api_key)
vectorstore = Chroma.from_documents(docs, embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# -------- LLM --------
llm = ChatMistralAI(
    model="mistral-small-2506",
    api_key=api_key,
    max_tokens=150,
    temperature=0.3
)

# -------- Memory --------
chat_history = []

# -------- Main Function --------
def get_response(query):
    docs = retriever.invoke(query)
    context = "\n\n".join([d.page_content for d in docs])

    history_text = ""
    for q, a in chat_history:
        history_text += f"User: {q}\nAssistant: {a}\n"

    # 👉 First response = analysis
    if len(chat_history) == 0:
        instruction = """
You are an ATS Resume Analyzer.

- Analyze the resume
"""
    else:
        instruction = """
You are a Resume Assistant.

- Use the resume to answer user questions."""

    prompt = f"""
    {instruction}

    History:
    {history_text}

    Resume:
    {context}

    User:
    {query}"""

    response = llm.invoke(prompt).content
    chat_history.append((query, response))

    return response

# -------- Chat Loop --------
print("🤖 Resume AI (type 'exit' to quit)\n")

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        print("👋 Bye!")
        break

    answer = get_response(user_input)
    print("Bot:", answer, "\n")