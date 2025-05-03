from transformers import pipeline

def suggest_decor(caption):
    prompt = f"The image shows: {caption}\n\nAs a home design expert, suggest simple improvements or decor tips."
    generator = pipeline("text-generation", model="microsoft/phi-2")
    output = generator(prompt, max_new_tokens=100)[0]["generated_text"]
    return output