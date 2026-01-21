from sentence_transformers import CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os,getpass

os.environ["HF_TOKEN"] = getpass.getpass("Enter your Hugging Face API Token: ")
embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore=FAISS.from_texts(["Cross encoders score query-document pairs","FAISS is a fast vector search library","LangChain changed retriever imports after 1.0",
                              "Compression is different from reranking"],embedding=embeddings)

retriever=vectorstore.as_retriever(search_kwargs={"k":10})
retriever.invoke("How does cross encoding work?")
reranker=CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def retrieve_and_rerank(query,top_n=5):
    docs=retriever.invoke(query)
    pairs=[(query,d.page_content) for d in docs]
    scores=reranker.predict(pairs)
    ranked=sorted(zip(scores,docs),key=lambda x:x[0],reverse=True)
    return [doc for _,doc in ranked[:top_n]]
docs=retrieve_and_rerank("How does cross encoding work?")
for d in docs:
    print(d.page_content)

