import streamlit as st
from captioning.image_captioner import generate_caption
from advisor.llm_suggester import suggest_decor
import tempfile

st.title("🛋️ Interior Style Advisor")

uploaded = st.file_uploader("Upload a photo of a bedroom", type=["jpg", "png", "jpeg"])
if uploaded:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded.read())
        image_path = tmp.name

    st.image(image_path, caption="Uploaded Bedroom")
    caption = generate_caption(image_path)
    st.markdown(f"**Caption**: {caption}")

    decor_tips = suggest_decor(caption)
    st.markdown("### 📝 Decor Suggestions")
    st.write(decor_tips)