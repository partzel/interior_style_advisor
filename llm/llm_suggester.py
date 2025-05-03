from transformers import pipeline

def suggest_decor(caption):
    prompt = f"The image shows: {caption}\n\nAs a home design expert, suggest simple improvements or decor tips."
    generator = pipeline("text-generation", model="meta-llama/Llama-3.2-1B-Instruct")
    output = generator(prompt, max_new_tokens=100)[0]["generated_text"]
    return output