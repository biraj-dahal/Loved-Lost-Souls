from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset, Dataset
import json
import torch

# model_name = "EleutherAI/gpt-j-6B"  # Pretrained GPT-J model
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer.pad_token = tokenizer.eos_token
# model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B").to("cuda" if torch.cuda.is_available() else "cpu")
conversation = []
file_path = "indiv_data/Chandler_Bing.jsonl"
with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        conversation.append(json.loads(line))
dataset = Dataset.from_list(conversation)



def format_example(example):
    # We include the prompt and response, ensuring the model sees the prompt and then generates the response.
    text = example["prompt"] + " " + example["response"] + tokenizer.eos_token  # end each example with EOS
    return {"text": text}

dataset = dataset.map(format_example)
train_data = dataset.shuffle(seed=42)  # shuffle data

# Tokenize the dataset
def tokenize_batch(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)
tokenized_data = train_data.map(tokenize_batch, batched=True, remove_columns=["text"])

# Set up training arguments
training_args = TrainingArguments(
    output_dir="style-gptj-chandler",  # where to save
    per_device_train_batch_size=1,
    num_train_epochs=3,
    learning_rate=2e-5,
    fp16=True,  # use mixed precision
    logging_steps=50,
    save_steps=200,
    save_total_limit=1
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data,
)
trainer.train()


test_prompt = "I just cleaned the apartment.\nCHANDLER:"
input_ids = tokenizer(test_prompt, return_tensors='pt').input_ids.to(model.device)
output_ids = model.generate(input_ids, max_new_tokens=50, do_sample=True, temperature=0.8)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))