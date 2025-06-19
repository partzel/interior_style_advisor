from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import PlainTextResponse
from typing import Optional
from PIL import Image
import io

from model.predictor import predict

router = APIRouter()

@router.post("/caption", response_class=PlainTextResponse)
async def caption_image(
    image: UploadFile = File(...),
    prompt: Optional[str] = Form(None)
):
    system_prompt = (
        "You are an expert interior designer assistant. Your task is to suggest suitable decor "
        "based on the inner design of a room described in a caption. The caption may include details "
        "about the room's layout, furniture, lighting, materials, colors, and style. Based on this information, "
        "provide thoughtful and visually harmonious decor suggestions that enhance the room's existing features. "
        "Be creative but practical, ensuring your suggestions match the overall theme and feel of the space."
    )

    user_prompt = (
        "Based on this image, suggest appropriate decoration items that would fit the space. "
        "Only output the suggestion — do not include any introductions, explanations, or extra text.\n\n"
        "Room image provided below.\n"
        f"{prompt or ''}\n"
        "Decoration suggestion:"
    )

    try:
        pil_image = Image.open(io.BytesIO(await image.read())).convert("RGB")
    except Exception as e:
        return PlainTextResponse(f"Error opening image: {e}", status_code=400)

    return predict(pil_image, system_prompt, user_prompt)