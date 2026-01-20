import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image
import io
import os
from dotenv import load_dotenv

# -------------------- SETUP --------------------
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

client = InferenceClient(api_key=HF_TOKEN)

# -------------------- IMAGE GENERATION --------------------
def generate_image(prompt):
    result = client.text_to_image(
        model=MODEL_ID,
        prompt=prompt
    )

    # HF may return PIL Image OR bytes
    if isinstance(result, Image.Image):
        return result
    return Image.open(io.BytesIO(result))

# -------------------- STREAMLIT APP --------------------
st.set_page_config(page_title="Text to Image Generator", page_icon="🎨")
st.title("🎨 Text → Image Generator")

prompt = st.text_input("Enter a prompt to generate an image")

if prompt and st.button("Generate Image"):
    with st.spinner("Generating image..."):
        image = generate_image(prompt)

    st.image(image, caption="Generated Image", use_container_width=True)
