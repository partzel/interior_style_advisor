import torch
from PIL import Image
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
tokenizer = open_clip.get_tokenizer("ViT-B-32")

def compute_clip_similarity(image_path: str, text: str):
    image = preprocess(Image.open(image_path)).unsqueeze(0)
    text = tokenizer([text])

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        similarity = torch.nn.functional.cosine_similarity(image_features, text_features)

    return similarity.item()