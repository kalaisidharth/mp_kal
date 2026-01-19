
# pip installs you may need (run once):
# pip install -U langchain langchain-community langchain-huggingface langchain-text-splitters sentence-transformers faiss-cpu

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# --- Use Hugging Face embeddings & chat model ---
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint

import os, getpass

# --- Set Hugging Face token (required for HuggingFaceEndpoint) ---
# Create token: https://huggingface.co/settings/tokens  (scope: "read")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass.getpass("Hugging Face token: ")

# ───────── LLM (Chat) via Hugging Face Inference API ─────────
# You can change repo_id to another instruct/chat model, e.g.:
#  - "mistralai/Mistral-7B-Instruct-v0.2"
#  - "HuggingFaceH4/zephyr-7b-beta"
#  - "tiiuae/falcon-7b-instruct"
llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-20b"
    )
)

# ───────── Embeddings ─────────
# If you prefer to keep OpenAI embeddings, you can, but here we switch to HF end-to-end.
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

texts = [
    "LangChain helps developers build LLM applications.",
    "FAISS is used for vector similarity search.",
    "Chat history must be manually maintained in LangChain 1.1.",
    "Retrievers are used in RAG pipelines.",
    "OpenAI embeddings create vector representations."
]

# Build FAISS vector store & retriever (unchanged)
db = FAISS.from_texts(texts, embeddings)
retriever = db.as_retriever()

# ───────── Multi-session chat history store ─────────
store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# ───────── Prompt (with history placeholder) ─────────
rag_prompt_with_history = ChatPromptTemplate.from_messages([
    ("system", "Use the retrieved context to answer the user. "
               "If the answer is not in the context, say so briefly before answering from general knowledge."),
    MessagesPlaceholder("history"),  # history injected here by RunnableWithMessageHistory
    ("human", "{question}\n\nContext:\n{context}")
])

# ───────── Retrieve context at runtime and assign into chain input ─────────
def get_context_from_retriever(question_dict):
    # The input to this function will be like: {'question': '...'}
    docs = retriever.invoke(question_dict["question"])
    return "\n".join([d.page_content for d in docs])

runnable = RunnablePassthrough.assign(context=get_context_from_retriever)

# Final chain: input -> add context -> render prompt -> call HF LLM
rag_chain_with_context = runnable | rag_prompt_with_history | llm

# Wrap with message history manager
conversational_rag_chain_with_history = RunnableWithMessageHistory(
    rag_chain_with_context,
    get_session_history,
    input_messages_key="question",  # which key in input is the user message
    history_messages_key="history", # which key in prompt accepts chat history
) m

def ask_with_managed_history(question: str, session_id: str = "default_session"):
    response = conversational_rag_chain_with_history.invoke(
        {"question": question},
        config={"configurable": {"session_id": session_id}}
    )
    return response.content

# ───────── Demo ─────────
store.clear()

print("User (session1): What is FAISS?")
print("AI (session1):", ask_with_managed_history("What is FAISS?", session_id="session1"))

print("\nUser (session1): What did I ask earlier?")
print("AI (session1):", ask_with_managed_history("What did I ask earlier?", session_id="session1"))

print("\nUser (session2): How does LangChain handle memory?")
print("AI (session2):", ask_with_managed_history("How does LangChain handle memory?", session_id="session2"))

print("\nUser (session1): And what about LangChain's memory?")
print("AI (session1):", ask_with_managed_history("And what about LangChain's memory?", session_id="session1"))

