# character_bot.py

# import json
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# class FriendsCharacterBot:
#     def __init__(self, character_name):
#         self.character = character_name
#         self.dialogue_data = self._load_data()
#         self.contexts = ["\n".join([f"{utt['speaker']}: {utt['text']}" for utt in d["context"]]) for d in self.dialogue_data]
#         self.responses = [d["response"] for d in self.dialogue_data]

#         # TF-IDF vectorizer
#         self.vectorizer = TfidfVectorizer()
#         self.X = self.vectorizer.fit_transform(self.contexts)
#     def _load_data(self):
#         file_path = f"friends_data/{self.character.replace(' ', '_')}_dialogues.json"
#         with open(file_path, "r", encoding="utf-8") as f:
#             return json.load(f)

#     def chat(self):
#         print(f"\nYou are now chatting with {self.character.split()[0]}!")
#         context_input = input("You: ")

#         while context_input.lower() not in ["bye", "exit", "quit"]:
#             context_vec = self.vectorizer.transform([context_input])
#             sims = cosine_similarity(context_vec, self.X)
#             best_match_idx = sims.argmax()
#             response = self.responses[best_match_idx]
#             print(f"{self.character.split()[0]}:", response)

#             context_input = input("You: ")

# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# from peft import PeftModel
# import os

# # BASE_MODEL = "TinyLlama"  # Or whatever base model you used
# MODEL_DIR = "trained"

# def load_character_model(character_name):
#     print(f"🔄 Loading model for {character_name}...")
    
#     # base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
#     # tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
#     # tokenizer.pad_token = tokenizer.eos_token

#     base_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
#     tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
#     print("✅ Model and tokenizer loaded successfully!")

#     # Load LoRA adapter
#     adapter_path = os.path.join(MODEL_DIR, character_name)
#     model = PeftModel.from_pretrained(base_model, adapter_path)
#     model.eval()
#     # pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
#     # pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")
#     pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, torch_dtype=torch.bfloat16, device_map="auto")

#     return pipe

# def chat(pipe, character_name):
#     print(f"\n🤖 Chatting with {character_name}! Type 'exit' to stop.")
#     conversation = ""
#     while True:
#         user_input = input("You: ")
#         if user_input.lower() in ["exit", "quit"]:
#             break
#         conversation += f"You: {user_input}\n{character_name}:"
#         response = pipe(conversation, max_new_tokens=100, do_sample=True, temperature=0.7, top_p=0.9)[0]['generated_text']
#         generated = response.split(f"{character_name}:")[-1].strip()
#         print(f"{character_name}: {generated}")
#         conversation += f" {generated}\n"

# if __name__ == "__main__":
#     characters = [d for d in os.listdir(MODEL_DIR) if os.path.isdir(os.path.join(MODEL_DIR, d))]
#     print("🧑 Choose a character to chat with:")
#     for i, c in enumerate(characters):
#         print(f"{i+1}. {c}")
#     choice = int(input("Enter number: ")) - 1
#     selected_character = characters[choice]

#     pipe = load_character_model(selected_character)
#     chat(pipe, selected_character)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import os

BASE_MODEL = "distilgpt2"  # Use the model you trained with
MODEL_DIR = "trained2"  # Directory where LoRA models are saved

def load_character_model(character_name):
    print(f"🔄 Loading model for {character_name}...")
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token is set

    # Load LoRA adapter
    adapter_path = os.path.join(MODEL_DIR, character_name)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    # Use the LoRA fine-tuned model for text generation
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
    return pipe

def chat(pipe, character_name):
    print(f"\n🤖 Chatting with {character_name}! Type 'exit' to stop.")
    conversation = ""
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        conversation += f"You: {user_input}\n{character_name}:"
        response = pipe(conversation, max_new_tokens=100, do_sample=True, temperature=0.7, top_p=0.9)[0]['generated_text']
        generated = response.split(f"{character_name}:")[-1].strip()
        print(f"{character_name}: {generated}")
        conversation += f" {generated}\n"

if __name__ == "__main__":
    characters = [d for d in os.listdir(MODEL_DIR) if os.path.isdir(os.path.join(MODEL_DIR, d))]
    print("🧑 Choose a character to chat with:")
    for i, c in enumerate(characters):
        print(f"{i+1}. {c}")
    choice = int(input("Enter number: ")) - 1
    selected_character = characters[choice]
    pipe = load_character_model(selected_character)
    chat(pipe, selected_character)
