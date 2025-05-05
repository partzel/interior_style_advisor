from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import PlainTextResponse
from typing import Optional

app = FastAPI()

@app.post("/caption", response_class=PlainTextResponse)
async def caption_image(
    image: UploadFile = File(...),
    prompt: Optional[str] = Form(None)
):
    # For debugging/logging purposes
    print(f"Received image: {image.filename}")
    if prompt:
        print(f"Prompt: {prompt}")

    # TODO: Add your actual image captioning logic here
    # For now, return a dummy caption
    return f"Dummy caption for {image.filename} with prompt: {prompt or 'None'}"


@app.get("/hello", response_class=PlainTextResponse)
async def hello():
    return "Hello world from the server!"