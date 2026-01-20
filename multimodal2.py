from huggingface_hub import InferenceClient
import base64
import os
import io
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import streamlit as st
from PIL import Image

# -------------------- ENV --------------------
load_dotenv()
api_key = os.getenv("HF_TOKEN")

# -------------------- MODELS --------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VLM_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
T2I_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

# -------------------- UTILS --------------------
def encode_uploaded_image(uploaded_file):
    return base64.b64encode(uploaded_file.getvalue()).decode("utf-8")

def encode_pil_image(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# -------------------- IMAGE GENERATION --------------------
def generate_image_from_text(client, prompt):
    result = client.text_to_image(
        model=T2I_MODEL,
        prompt=prompt
    )

    # HF may return PIL Image OR bytes
    if isinstance(result, Image.Image):
        return result
    else:
        return Image.open(io.BytesIO(result))


# -------------------- IMAGE DESCRIPTION --------------------
def get_image_description(client, img_base64):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}"
                    }
                },
                {
                    "type": "text",
                    "text": "Describe this image in high detail including objects, colors, background, and notable details."
                }
            ]
        }
    ]

    completion = client.chat_completion(
        model=VLM_MODEL,
        messages=messages,
        max_tokens=800
    )

    return completion.choices[0].message.content

# -------------------- RAG --------------------
def create_rag_index(description_text):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    doc = Document(
        page_content=description_text,
        metadata={"source": "image_description"}
    )
    return FAISS.from_documents([doc], embeddings)

def answer_question_from_rag(client, vector_store, question):
    retrieved_docs = vector_store.similarity_search(question, k=1)
    context = retrieved_docs[0].page_content

    prompt = f"""
Use the context below to answer the question accurately.

Context:
{context}

Question:
{question}

Answer:
"""

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }
    ]

    completion = client.chat_completion(
        model=VLM_MODEL,
        messages=messages,
        max_tokens=300
    )

    return completion.choices[0].message.content

# -------------------- STREAMLIT APP --------------------
def main():
    st.set_page_config(page_title="Multimodal RAG", page_icon="🖼️")
    st.title("🖼️ Multimodal RAG (Text ↔ Image ↔ Q&A)")
    st.write("Generate or upload an image, build a RAG index, and ask questions.")

    client = InferenceClient(api_key=api_key)

    mode = st.radio(
        "Choose Input Mode",
        ["Upload Image", "Generate Image from Text"]
    )

    # Reset state
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
        st.session_state.description = None

    # -------------------- IMAGE UPLOAD --------------------
    if mode == "Upload Image":
        uploaded_file = st.file_uploader(
            "Upload an Image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file and st.button("Analyze Image"):
            st.image(uploaded_file, caption="Uploaded Image", width=500)

            with st.spinner("Analyzing image..."):
                img_base64 = encode_uploaded_image(uploaded_file)
                description = get_image_description(client, img_base64)

                st.session_state.description = description
                st.session_state.vector_store = create_rag_index(description)

            st.success("Image analyzed and indexed!")

    # -------------------- TEXT TO IMAGE --------------------
    if mode == "Generate Image from Text":
        text_prompt = st.text_input("Enter a text prompt to generate an image")

        if text_prompt and st.button("Generate & Analyze"):
            with st.spinner("Generating image..."):
                image = generate_image_from_text(client, text_prompt)
                st.image(image, caption="Generated Image", width=500)

            with st.spinner("Analyzing image..."):
                img_base64 = encode_pil_image(image)
                description = get_image_description(client, img_base64)

                st.session_state.description = description
                st.session_state.vector_store = create_rag_index(description)

            st.success("Image generated, analyzed, and indexed!")

    # -------------------- QA --------------------
    if st.session_state.vector_store:
        with st.expander("📄 View Image Description (Context)"):
            st.write(st.session_state.description)

        question = st.text_input("Ask a question about the image")

        if question:
            with st.spinner("Answering..."):
                answer = answer_question_from_rag(
                    client,
                    st.session_state.vector_store,
                    question
                )

            st.markdown("### ✅ Answer")
            st.write(answer)

# -------------------- RUN --------------------
if __name__ == "__main__":
    main()
