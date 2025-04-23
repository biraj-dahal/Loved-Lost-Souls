from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
import json
import torch
import gc
from pathlib import Path

# ---- CONFIG ----
MODEL_NAME = "gpt2"  
DATA_PATH = "indiv_data/Monica_Geller.jsonl"
OUTPUT_DIR = "style-gpt2-monica"
MAX_LENGTH = 64  

# ---- TOKENIZER ----tus

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# ---- LOAD MODEL ----
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("cuda" if torch.cuda.is_available() else "cpu")

# ---- LOAD DATA ----
def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

conversation = load_jsonl(DATA_PATH)
dataset = Dataset.from_list(conversation)
print(dataset)
# ---- FORMAT + TOKENIZE ----
def format_and_tokenize(example):
    if isinstance(example["prompt"], list):  # Batched
        texts = [
            p + " " + r + tokenizer.eos_token
            for p, r in zip(example["prompt"], example["response"])
        ]
        encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=MAX_LENGTH)
    else:  # Single example
        text = example["prompt"] + " " + example["response"] + tokenizer.eos_token
        encodings = tokenizer(text, truncation=True, padding="max_length", max_length=MAX_LENGTH)

    # Add labels = input_ids
    encodings["labels"] = encodings["input_ids"].copy()
    return encodings

tokenized_data = dataset.map(format_and_tokenize, batched=True, remove_columns=["prompt", "response"])

# ---- TRAINING ----
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    learning_rate=2e-5,
    fp16=torch.cuda.is_available(),
    logging_steps=50,
    save_steps=200,
    save_total_limit=1,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data,
)

trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
