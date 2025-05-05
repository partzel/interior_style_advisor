from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import PlainTextResponse
from typing import Optional
from PIL import Image 
import requests 
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor 

app = FastAPI()

@app.post("/caption", response_class=PlainTextResponse)
async def caption_image(
    image: UploadFile = File(...),
    prompt: Optional[str] = Form(None)
):


    model_id = "microsoft/Phi-3-vision-128k-instruct" 

    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", trust_remote_code=True, torch_dtype="auto", _attn_implementation='flash_attention_2') # use _attn_implementation='eager' to disable flash attention

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True) 

    messages = [ 
        {"role": "system", "content": "<|image_1|>\nGive design ideas about the room picture."},
        {"role": "user", "content": prompt} 
    ]
    
    # Read file contents into memory
    image_bytes = await image.read()
    
    # Open it as a PIL image
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return PlainTextResponse(f"Error opening image: {e}", status_code=400)

    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(prompt, [pil_image], return_tensors="pt").to("cuda:0") 

    generation_args = { 
        "max_new_tokens": 500, 
        "temperature": 0.0, 
        "do_sample": False, 
    } 

    generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args) 

    # remove input tokens 
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 



    # For debugging/logging purposes
    print(f"Received image: {image.filename}")
    if prompt:
        print(f"Prompt: {prompt}")
        
    return response


@app.get("/hello", response_class=PlainTextResponse)
async def hello():
    return "Hello world from the server!"