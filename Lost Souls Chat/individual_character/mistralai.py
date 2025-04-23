import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType
from transformers import BitsAndBytesConfig
model_id = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
dataset = load_dataset("csv", data_files={"train": "/home/oem/Downloads/Loved-Lost-Souls/Lost Souls Chat/individual_character/Chandler_Bing.csv"}, split="train")
def tokenize(batch):
    return tokenizer(batch["prompt"], padding="max_length", truncation=True, max_length=512)

dataset = dataset.map(tokenize, batched=True)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

# 4. Prepare model for QLoRA
model = prepare_model_for_kbit_training(model)

# 5. Set LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # These may vary for different models
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

# 6. Training args
training_args = TrainingArguments(
    output_dir="./qlora-style-checkpoint",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    report_to="none"
)

# 7. Trainer
trainer = Trainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()
model.save_pretrained("./qlora-style-model")
tokenizer.save_pretrained("./qlora-style-model")
