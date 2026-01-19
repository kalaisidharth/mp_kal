
# # pip installs you may need (uncomment and run once):
# # !pip install -U langchain langchain-community langchain-huggingface sentence-transformers faiss-cpu

# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.messages import HumanMessage, AIMessage
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# import os, getpass

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

# # --- Build the vector store and retriever ---
# db = FAISS.from_texts(texts, embeddings)
# retriever = db.as_retriever()

# # --- Use a Hugging Face instruct model via the Inference API ---
# llm = ChatHuggingFace(
#     llm=HuggingFaceEndpoint(
#         repo_id="openai/gpt-oss-20b"
       
#     )
# )

# # --- Prompt with chat history placeholder ---
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are a helpful assistant. Use the retrieved context to answer the user. "
#                "If the answer isn't in the context, say so briefly, then answer from general knowledge."),
#     MessagesPlaceholder(variable_name="messages"),
#     ("human", "{question}\n\nContext:\n{context}")
# ])

# # --- Chain ---
# chain = prompt | llm

# # --- Manual history (HumanMessage + AIMessage), as in your original code ---
# history = []

# def ask(question: str) -> str:
#     global history

#     # Retrieve context documents
#     docs = retriever.invoke(question)
#     context = "\n".join([d.page_content for d in docs])

#     # Add human message to history
#     history.append(HumanMessage(content=question))

#     # Run the RAG chain
#     response = chain.invoke({
#         "context": context,
#         "messages": history,   # now actually used via MessagesPlaceholder
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




# pip installs you may need (uncomment and run once):
# !pip install -U langchain langchain-community langchain-huggingface sentence-transformers numpy