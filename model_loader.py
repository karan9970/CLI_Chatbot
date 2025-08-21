%%writefile model_loader.py

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

def get_textgen_pipeline(model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    textgen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )
    return textgen_pipeline

def generate_response(pipe, prompt: str, max_new_tokens: int = 50):
    response = pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        pad_token_id=pipe.tokenizer.eos_token_id,
        temperature=0.5
    )
    response_text = response[0]["generated_text"]
    # Return only the new part after the prompt for cleaner output
    return response_text[len(prompt):].strip()

if __name__ == "__main__":
    pipe = get_textgen_pipeline("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    user_input = "what is the capital of france?"
    response = generate_response(pipe, user_input)
    print(response)
