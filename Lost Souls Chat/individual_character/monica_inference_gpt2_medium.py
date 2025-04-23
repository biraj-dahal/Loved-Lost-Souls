# inference.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ---- CONFIG ----
MODEL_DIR = "style-gpt2-monica"

# ---- LOAD MODEL ----
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to("cuda" if torch.cuda.is_available() else "cpu")

# ---- GENERATE FUNCTION ----
def generate_response(prompt, max_tokens=50):
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(model.device)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        top_k=50
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# ---- INTERACT ----
while True:
    prompt = input("\nYou: ")
    if prompt.lower() in ["quit", "exit"]:
        break
    response = generate_response(prompt + "\nMonica Geller:")
    print(response)
