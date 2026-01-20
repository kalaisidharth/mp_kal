from huggingface_hub import InferenceClient
import base64
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import streamlit as st

load_dotenv()
api_key = os.getenv("HF_TOKEN")

# Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VLM_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

def encode_image(uploaded_file):
    """Encodes uploaded file object to base64."""
    bytes_data = uploaded_file.getvalue()
    encoded_string = base64.b64encode(bytes_data)
    return encoded_string.decode('utf-8')

def get_image_description(client, img_base64):
    """
    Step 1 & 2: Sends image to LLM to get a detailed description.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                {"type": "text", "text": "Describe this image in high detail. Include the subject, background, colors, and any notable features."}
            ]
        }
    ]
    
    completion = client.chat_completion(model=VLM_MODEL, messages=messages, max_tokens=1000)
    description = completion.choices[0].message.content
    return description

def create_rag_index(description_text):
    """
    Step 3: Converts description to embeddings and stores in FAISS.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    doc = Document(page_content=description_text, metadata={"source": "image_description"})
    vector_store = FAISS.from_documents([doc], embeddings)
    return vector_store

def answer_question_from_rag(client, vector_store, question):
    """
    Step 4: Retrieves context and answers the user's question.
    """
    retrieved_docs = vector_store.similarity_search(question, k=1)
    context = retrieved_docs[0].page_content

    prompt = f"""Use the following context to answer the question.
    Context:
    {context}
    Question: 
    {question}
    Answer:"""

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    completion = client.chat_completion(model=VLM_MODEL, messages=messages, max_tokens=300)
    return completion.choices[0].message.content

def main():
    st.set_page_config(page_title="Multimodal RAG", page_icon="🖼️")
    st.title("Multimodal RAG Analysis")
    st.write("Upload an image to generate a description, index it with FAISS, and ask questions!")

    client = InferenceClient(api_key=api_key)

    # File Uploader
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        # Display image
        st.image(uploaded_file, caption="Uploaded Image", width=500)
        
        # Initialize session state for this image if not present or changed
        if "last_uploaded_file" not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
             st.session_state.vector_store = None
             st.session_state.description = None
             st.session_state.last_uploaded_file = uploaded_file.name

        # Button to process
        if st.session_state.vector_store is None:
            if st.button("Analyze Image"):
                with st.spinner("Generating description and building RAG index..."):
                    # Encode
                    img_base64 = encode_image(uploaded_file)
                    
                    # Describe
                    description = get_image_description(client, img_base64)
                    st.session_state.description = description
                    
                    # Index
                    vector_store = create_rag_index(description)
                    st.session_state.vector_store = vector_store
                    st.success("Analysis Complete! RAG Index Created.")
        
        # If processed, show QA interface
        if st.session_state.vector_store:
            with st.expander("View Generated Description (Context)"):
                st.write(st.session_state.description)
                
            question = st.text_input("Ask a question about the image:")
            if question:
                with st.spinner("Retrieving answer from context..."):
                    answer = answer_question_from_rag(client, st.session_state.vector_store, question)
                    st.markdown("### Answer")
                    st.write(answer)
                    
if __name__ == "__main__":
    main()

