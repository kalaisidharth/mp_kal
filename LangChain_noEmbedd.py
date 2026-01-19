from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os, getpass
from langchain_core.documents import Document


os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass.getpass()
llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-20b"
    )
)

documents = [
    "A Christmas Carol is a novella by Charles Dickens, first published in London on 19 December 1843.",
    "The story tells of sour and stingy Ebenezer Scrooge's ideological, ethical, and emotional transformation after \n"
    "the supernatural visits of Jacob Marley and the Ghosts of Christmas Past, Present, and Yet to Come.",
    "The novella met with instant success and critical acclaim. It is regarded as one of the greatest Christmas stories ever written."
]

context="\n\n".join(documents)
print("Document to process:", context)

prompt=ChatPromptTemplate.from_template("""You are an assistant that answers questions strictly using the provided context.
                                        Context: {context}
                                        Question: {question}
                                        If the answer is not in the context, say:'I don't know based on the provided context.'"""
)
chain=prompt | llm

response=chain.invoke({"context":context, "question":"What is the significance of Christmas Eve in A Christmas Carol?"})
print("Response:", response)

document_for_splitter=[Document(page_content=doc)for doc in documents]
text_splitter=RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10,separators=["\n\n", "\n", " ", ""])
chunks=text_splitter.split_documents(document_for_splitter)

print(chunks)

context="\n\n".join(chunk.page_content for chunk in chunks)
print(context)
response=chain.invoke({
    "context":context,
    "question":"What is A Christmas Carol?"
})
print(response)