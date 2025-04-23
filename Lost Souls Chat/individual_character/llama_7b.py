
# from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, pipeline
# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# from datasets import load_dataset
# import json 

# model_id = "meta-llama/Llama-2-7b-chat-hf"
# dataset = load_dataset('json', data_files='indiv_data/Chandler_Bing.jsonl')
# tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token="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")
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     load_in_4bit=True,
#     device_map="auto",
#     use_auth_token=True
# )

# model = prepare_model_for_kbit_training(model)
# lora_config = LoraConfig(
#     r=8,
#     lora_alpha=32,
#     target_modules=["q_proj", "v_proj"],
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM",
# )

# model = get_peft_model(model, lora_config)
# def tokenize(example):
#     tokens = tokenizer(
#         example["prompt"],
#         truncation=True,
#         padding="max_length",
#         max_length=512,
#     )
#     tokens["labels"] = tokens["input_ids"].copy()
#     return tokens
# tokenized_dataset = dataset.map(tokenize, batched=True)

# training_args = TrainingArguments(
#     output_dir="./chandler_bot",
#     per_device_train_batch_size=2,
#     gradient_accumulation_steps=4,
#     learning_rate=2e-5,
#     num_train_epochs=3,
#     logging_dir="./logs",
#     fp16=True,
#     save_total_limit=1,
#     save_steps=100,
#     logging_steps=10,
#     report_to="none",
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_dataset["train"],
# )

# trainer.train()


# print("Saving model...")
# trainer.save_model("./chandler-bot")
# tokenizer.save_pretrained("./chandler-bot")

# # Inference
# print("Testing chatbot...")
# chat = pipeline("text-generation", model="./chandler-bot", tokenizer="./chandler-bot", device=0)

# prompt = "Could you BE any more sarcastic?"
# output = chat(f"### Human: {prompt}\n### Assistant:", max_new_tokens=50)[0]["generated_text"]
# print("Chandler Bot says:\n", output)

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from datasets import load_dataset
import torch

# Model ID
model_id = "meta-llama/Llama-2-7b-chat-hf"

# Load dataset (make sure it's JSONL format with keys: prompt, response)
dataset = load_dataset("json", data_files={"train": "indiv_data/Chandler_Bing.jsonl"})

# Format messages for chat template
def format_chat(example):
    return {
        "messages": [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["response"]}
        ]
    }

dataset["train"] = dataset["train"].map(format_chat)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Tokenize using the chat template
def tokenize(example):
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    tokens = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=512
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset["train"].map(tokenize, batched=True)

# Load model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_4bit=True,
    device_map="auto",
    use_auth_token=True
)

# Prepare model for LoRA
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Training args
training_args = TrainingArguments(
    output_dir="./chandler_bot_llama",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=3,
    logging_dir="./logs",
    fp16=True,
    save_total_limit=1,
    save_steps=100,
    logging_steps=10,
    report_to="none"
)

# Trainera
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

# Train
trainer.train()

# Save model and tokenizer
trainer.save_model("./chandler_bot_llama")
tokenizer.save_pretrained("./chandler_bot_llama")
