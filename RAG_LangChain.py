

# #without FAISS vector store, in-memory embeddings and retrieval
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.messages import HumanMessage, AIMessage
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from langchain_huggingface import HuggingFaceEmbeddings

# import numpy as np
# import os, getpass
# from typing import List, Tuple

# # --- Set Hugging Face token (required for HuggingFaceEndpoint) ---
# # Create a token at https://huggingface.co/settings/tokens (scope: "read")
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass.getpass("Hugging Face token: ")

# # --- Use a compact, accurate sentence-transformer for embeddings ---
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # --- Sample corpus ---
# texts = [
#     "LangChain helps developers build LLM applications.",
#     "FAISS is used for vector similarity search.",
#     "Chat history must be manually maintained in LangChain 1.1.",
#     "Retrievers are used in RAG pipelines.",
#     "OpenAI embeddings create vector representations."
# ]

# # --- Precompute in-memory embeddings for the corpus (no FAISS) ---
# doc_embeddings = embeddings.embed_documents(texts)  # List[List[float]]
# doc_embeddings = np.array(doc_embeddings, dtype=np.float32)  # shape: (N, D)

# def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
#     """
#     Compute cosine similarity between each row in a and each row in b.
#     a: shape (n, d), b: shape (m, d)
#     returns: shape (n, m)
#     """
#     # Normalize rows to unit length
#     a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
#     b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
#     return a_norm @ b_norm.T

# def retrieve(query: str, k: int = 3) -> List[Tuple[str, float]]:
#     """
#     Simple in-memory retriever using cosine similarity over precomputed embeddings.
#     Returns top-k (text, score).
#     """
#     q_emb = np.array([embeddings.embed_query(query)], dtype=np.float32)  # shape: (1, D)
#     sims = cosine_sim_matrix(q_emb, doc_embeddings)  # shape: (1, N)
#     sims = sims.flatten()
#     top_idx = np.argsort(-sims)[:k]
#     return [(texts[i], float(sims[i])) for i in top_idx]

# # --- Use a Hugging Face instruct/chat model via the Inference API ---
# llm = ChatHuggingFace(
#     llm=HuggingFaceEndpoint(
#         repo_id="openai/gpt-oss-20b"
#     )
# )

# # --- Prompt with chat history placeholder ---
# prompt = ChatPromptTemplate.from_messages([
#     ("system",
#      "You are a helpful assistant. Use the retrieved context to answer the user. "
#      "If the answer isn't in the context, say so briefly, then answer from general knowledge."),
#     MessagesPlaceholder(variable_name="messages"),
#     ("human", "{question}\n\nContext:\n{context}")
# ])

# # --- Chain ---
# chain = prompt | llm

# # --- Manual history (HumanMessage + AIMessage), same pattern as your code ---
# history: List = []

# def ask(question: str) -> str:
#     global history

#     # Retrieve top-k context passages (no FAISS)
#     retrieved = retrieve(question, k=3)
#     context = "\n".join([t for t, _ in retrieved])

#     # Add human message to history
#     history.append(HumanMessage(content=question))

#     # Run the RAG chain
#     response = chain.invoke({
#         "context": context,
#         "messages": history,   # actually used via MessagesPlaceholder
#         "question": question
#     })

#     # Add AI reply to history
#     history.append(AIMessage(content=response.content))

#     return response.content

# # --- Demo ---
# print("User: What is FAISS?")
# print("AI:", ask("What is FAISS?"))

# print("\nUser: What did I ask earlier?")
# print("AI:", ask("What did I ask earlier?"))

# print("\nUser: How does LangChain handle memory?")
# print("AI:", ask("How does LangChain handle memory?"))


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
import os, getpass
from typing import List, Tuple

os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass.getpass("Hugging Face token: ")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


texts = [
    "LangChain helps developers build LLM applications.",
    "FAISS is used for vector similarity search.",
    "Chat history must be manually maintained in LangChain 1.1.",
    "Retrievers are used in RAG pipelines.",
    "OpenAI embeddings create vector representations."
]
user_query="What is LangChain?"
doc_embeddings = embeddings.embed_documents(texts)  # List[List[float]]
doc_embeddings = np.array(doc_embeddings, dtype=np.float32)  # (N, D)
print("Document embeddings shape:", doc_embeddings.shape)
doc_vectors=embeddings.embed_documents(texts)
query_vector=embeddings.embed_query(user_query)

scores=[np.dot(query_vector,doc_vec) for doc_vec in doc_vectors]
print("Similarity scores:", scores)
best_index=np.argmax(scores)
print("Best matching document index:", best_index)
context=texts[best_index]
print("Context for the query:", context)
  