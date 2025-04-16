import os
import json
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType

# Base model you want to fine-tune
BASE_MODEL = "distilgpt2"  # Or "TinyLlama/TinyLlama-1.1B" or similar
DATA_DIR = "jsonl_data"
OUTPUT_DIR = "trained"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(jsonl_path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
    return Dataset.from_list(lines)

def tokenize(example, tokenizer):
    full_prompt = example['prompt'] + example['completion']
    return tokenizer(full_prompt, truncation=True, padding='max_length', max_length=512)

def train_model(character_name, jsonl_path):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    data = load_data(jsonl_path)
    tokenized_data = data.map(lambda ex: tokenize(ex, tokenizer), batched=False)

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["c_attn", "c_proj", "q_attn", "v_attn"]
    )

    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
    output_dir=f"{OUTPUT_DIR}/{character_name}",
    per_device_train_batch_size=1,        # Lower batch size
    gradient_accumulation_steps=2,        # Accumulate gradients
    num_train_epochs=2,
    save_total_limit=1,
    fp16=False,                            # Use 16-bit floating point (if GPU supports)
    logging_dir=f"{OUTPUT_DIR}/{character_name}/logs",
    report_to="none")


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    trainer.train()
    model.save_pretrained(f"{OUTPUT_DIR}/{character_name}")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/{character_name}")
    print(f"✅ Model saved for {character_name}")

if __name__ == "__main__":
    for file in os.listdir(DATA_DIR):
        if file.endswith(".jsonl"):
            character_name = file.replace(".jsonl", "")
            file_path = os.path.join(DATA_DIR, file)
            print(f"🚀 Training model for {character_name}...")
            train_model(character_name, file_path)
