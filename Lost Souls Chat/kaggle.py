from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig
from datasets import load_dataset, Dataset
from transformers import default_data_collator, get_linear_schedule_with_warmup
import pandas as pd
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling

# want to fine-tune on the dialogue of main characters
# everyone else is kind of irrelevant, honestly
main_chars = ['Ross', 'Monica', 'Rachel', 'Chandler', 'Phoebe', 'Joey']
# sometimes the scripts have different casing for characters
main_chars = [m.lower() for m in main_chars]

def is_valid_line(line, main_chars=main_chars):
    """
    Check if a line is complete, dialogue and part of the main characters.

    Parameters:
    - line (str): The line to be checked.
    """
    if len(line)>0:
        if line[0].isalpha():
            name = line.split(':')[0].lower()
            if name in main_chars:
                return True
    return False

lines = open('Friends_Transcript.txt', 'r').read().split('\n')
# setting up a dictionary with our different corpus sets and the model name
from collections import defaultdict
data_dict = defaultdict(list)
diag_block = []
for l in lines:
    if is_valid_line(l):
        # lines with speaker information removed
        data_dict['llama_friends'].append(l.split(':')[1].strip())
        # lines with speaker information included
        data_dict['llama_friends_speaker'].append(l)
        # collecting subsequent dialog blocks
        diag_block.append(l.strip())
    else:
        if diag_block != []:
            data_dict['llama_friends_block'].append('\n'.join(diag_block))
            diag_block = []

for d in data_dict:
    print(d, '\n', data_dict[d][0])


from huggingface_hub import notebook_login

notebook_login()
model_name = 'meta-llama/Llama-2-7b-chat-hf'

tokenizer = AutoTokenizer.from_pretrained(model_name)

# we'll be loading in 8bit to save some space
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto"
)

# simple prompt - continued with something reasonable
prompt = "I am a"
inputs = tokenizer(prompt, return_tensors="pt")
inputs.to('cuda')
# you can manipulate these to some effect, but this is pretty basic
generated_ids = model.generate(**inputs, max_new_tokens=50)
tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False, # we're training, not doing inference!
    # some basic parameters used in some tutorials I saw!
    r=8, lora_alpha=32, lora_dropout=0.1,
    base_model_name_or_path='meta-llama/Llama-2-7b-hf',
    # interesting - these are the layers we're targeting with our low rank decomposition
    # lots of options here, but these work well
    target_modules = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
  ]
)

# set up model for training
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# creating a dataset
# specify which set you want to use
data_version = 'llama_friends_block'
# new version of Dataset (not Kaggle's default) has from_list, but this just works
lines_dataset = Dataset.from_pandas(pd.DataFrame(data_dict['llama_friends_block']))
def tokenize_function(examples):
  # utility function to map to dataset
  return tokenizer(examples["0"])

tokenized_datasets = lines_dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=['0'])
block_size = 128

def group_texts(examples):
    # Concatenate all texts, chop up into block size (defined above)
    # adapted from https://huggingface.co/docs/transformers/tasks/language_modeling
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
)

# take a look inside!
print(tokenizer.decode(lm_datasets['input_ids'][0]))
# setting up training
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
training_args = TrainingArguments(
    data_version, # name the directory after our data version
    num_train_epochs=3, # tends to work well
    learning_rate=1e-3,  # these parameters led to a bit better style transfer
    weight_decay=0.01,
    report_to = [] # otherwise will try to report to wnb, which is weird default behavior
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets,
    data_collator=data_collator
)
# run training - will take a few hours on the free GPU
trainer.train()
prompt = "I am a"
inputs = tokenizer(prompt, return_tensors="pt")
inputs.to('cuda')
generated_ids = model.generate(**inputs, max_new_tokens=50)
tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig

#peft_model_id = "bpben/llama_friends_block"
peft_model_id = "bpben/llama_friends"
model_name = 'meta-llama/Llama-2-7b-chat-hf'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             load_in_8bit=True,
                                            device_map="auto")

model.load_adapter(peft_model_id)
# back to our simple prompt
prompt = "I am a"
inputs = tokenizer(prompt, return_tensors="pt")
_ = inputs.to('cuda')
model.disable_adapters()
generated_ids = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
