import streamlit as st
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration
import torch
import asyncio

# Fix event loop issue
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Load BLIP model and processor
@st.cache_resource()
def load_model():
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.eval()
    return processor, model

processor, model = load_model()

def generate_caption(image):
    text_prompt = "a photography of"
    inputs = processor(image, text_prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

# Streamlit UI
st.title("🖼️ Image Captioning App")
st.write("Upload an image, and the AI will generate a caption for it!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")
    
    if st.button("Generate Caption"):
        caption = generate_caption(image)
        st.write("### Generated Caption:")
        st.success(caption)
