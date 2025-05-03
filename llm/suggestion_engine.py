from transformers import pipeline

def get_decor_suggestions(caption: str, model_name: str = "microsoft/phi-2"):
    prompt = f"The bedroom is described as: '{caption}'.\n\nAs a home interior design assistant, provide some tasteful decor suggestions."
    generator = pipeline("text-generation", model=model_name)
    response = generator(prompt, max_new_tokens=100)[0]['generated_text']
    return response.strip()