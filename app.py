import streamlit as st
from data.load_dataset import load_reddit_caption_dataset
from llm.suggestion_engine import get_decor_suggestions
from clip.clip_utils import compute_clip_similarity
from PIL import Image
import random

st.title("🛏️ Bedroom Decor Advisor with Caption-Based Suggestions")
ds = load_reddit_caption_dataset()
sample = random.choice(ds)

image = sample["images"][0]
st.image(image, caption="Uploaded Bedroom Photo")

caption_choice = st.radio("Choose a caption source:", ["Gemma", "Qwen"])
caption = sample["caption_gemma"] if caption_choice == "Gemma" else sample["caption_qwen"]

st.markdown(f"**Selected Caption ({caption_choice})**: {caption}")

if st.button("Generate Decor Suggestions"):
    suggestions = get_decor_suggestions(caption)
    st.markdown("### 📝 Decor Suggestions")
    st.write(suggestions)

    similarity_score = compute_clip_similarity(image, caption)
    st.markdown(f"**CLIP Similarity**: {similarity_score:.2f}")