import json
import os

input_dir = "friends_data"
output_dir = "jsonl_data"

os.makedirs(output_dir, exist_ok=True)

def make_jsonl(character_file):
    input_path = os.path.join(input_dir, character_file)
    character_name = character_file.replace("_dialogues.json", "").replace("_", " ")
    output_path = os.path.join(output_dir, f"{character_file.replace('_dialogues.json', '')}.jsonl")

    with open(input_path, "r", encoding="utf-8") as f:
        dialogues = json.load(f)

    with open(output_path, "w", encoding="utf-8") as f_out:
        for entry in dialogues:
            context_lines = [f"{utt['speaker']}: {utt['text']}" for utt in entry["context"]]
            prompt = "\n".join(context_lines) + f"\n{character_name}:"
            completion = " " + entry["response"].strip()

            json_line = {
                "prompt": prompt,
                "completion": completion
            }

            f_out.write(json.dumps(json_line) + "\n")

    print(f"✅ Created: {output_path}")

if __name__ == "__main__":
    for file in os.listdir(input_dir):
        if file.endswith("_dialogues.json"):
            make_jsonl(file)
