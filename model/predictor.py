import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.clip_utils import compute_clip_embedding
from model.projector import load_projector

device = "cuda" if torch.cuda.is_available() else "cpu"

llm_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(llm_name)
tokenizer.pad_token = tokenizer.eos_token
llm = AutoModelForCausalLM.from_pretrained(llm_name, device_map="auto").eval().to(device)

projector = load_projector()

def predict(image, system_prompt, user_prompt_template):
    image_embedding = compute_clip_embedding(image)
    projected = projector(image_embedding)

    user_prompt = user_prompt_template.format(caption="<fill>")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    tokenized = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        input_embeddings = llm.model.embed_tokens(tokenized.input_ids)

    full_input = torch.cat([projected, input_embeddings], dim=1)

    output_ids = llm.generate(
        inputs_embeds=full_input,
        max_new_tokens=500,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)