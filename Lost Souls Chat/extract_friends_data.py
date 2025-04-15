
from convokit import Corpus, download
import pandas as pd
import json
import os
from collections import Counter, defaultdict


corpus = Corpus(filename=download("friends-corpus"))

def extract_friends_dialogue():

    

    main_characters = [
        "Rachel Green", 
        "Ross Geller", 
        "Chandler Bing", 
        "Monica Geller", 
        "Joey Tribbiani", 
        "Phoebe Buffay"
    ]
    

    os.makedirs("friends_data", exist_ok=True)
    character_counts = Counter()
    for utt in corpus.iter_utterances():
        speaker = utt.speaker.id
        if speaker in main_characters:
            character_counts[speaker] += 1
    
    print("Character utterance counts:")
    for character, count in character_counts.most_common():
        print(f"- {character}: {count} utterances")
    
    print("\nExtracting conversations...")
    episodes = defaultdict(list)
    
    for utt in corpus.iter_utterances():
        conv_id = utt.conversation_id
        episodes[conv_id].append({
            'speaker': utt.speaker.id,
            'text': utt.text,
            'timestamp': utt.timestamp,
            'reply_to': utt.reply_to,
            'id': utt.id
        })
    
    for conv_id, utterances in episodes.items():
        episodes[conv_id] = sorted(utterances,key=lambda x: x['timestamp'] if x['timestamp'] is not None else 0)

    
    
    with open("friends_data/friends_conversations.txt", "w", encoding="utf-8") as f:
        for conv_id, utterances in episodes.items():
            f.write(f"Episode: {conv_id}\n")
            f.write("-" * 50 + "\n")
            for utt in utterances:
                speaker = utt['speaker']
                text = utt['text'].replace("\n", " ")
                f.write(f"{speaker}: {text}\n")
            f.write("\n\n")
    
    for character in main_characters:
        with open(f"friends_data/{character.replace(' ', '_')}.txt", "w", encoding="utf-8") as f:
            for conv_id, utterances in episodes.items():
                conversation_context = []
                for utt in utterances:
                    speaker = utt['speaker']
                    text = utt['text'].replace("\n", " ")
                    
                    conversation_context.append(f"{speaker}: {text}")
                    
                    if speaker == character and len(conversation_context) > 1:
                        f.write("CONTEXT:\n")
                        for i, context_line in enumerate(conversation_context[:-1]):
                            f.write(f"{context_line}\n")
                        
                        f.write("\nRESPONSE:\n")
                        f.write(f"{conversation_context[-1]}\n\n")
                        f.write("-" * 50 + "\n\n")
    
    character_dialogues = {character: [] for character in main_characters}
    
    for conv_id, utterances in episodes.items():
        for i, utt in enumerate(utterances):
            speaker = utt['speaker']
            if speaker in main_characters:
                context = []
                for prev_utt in utterances[:i]:
                    context.append({
                        "speaker": prev_utt['speaker'],
                        "text": prev_utt['text']
                    })
                
                character_dialogues[speaker].append({
                    "episode": conv_id,
                    "context": context,
                    "response": utt['text'],
                    "timestamp": utt['timestamp']
                })
    
    for character, dialogues in character_dialogues.items():
        with open(f"friends_data/{character.replace(' ', '_')}_dialogues.json", "w", encoding="utf-8") as f:
            json.dump(dialogues, f, indent=2)
    
    with open("friends_data/friends_gpt_training.jsonl", "w", encoding="utf-8") as f:
        for conv_id, utterances in episodes.items():
            for i, utt in enumerate(utterances):
                speaker = utt['speaker']
                if speaker in main_characters and i > 0:
                    prev_dialogue = ""
                    for prev_utt in utterances[:i]:
                        prev_speaker = prev_utt['speaker']
                        prev_text = prev_utt['text'].replace("\n", " ")
                        prev_dialogue += f"{prev_speaker}: {prev_text}\n"
                    
                    entry = {
                        "prompt": f"{prev_dialogue}\n{speaker}:",
                        "completion": f" {utt['text']}",
                        "character": speaker
                    }
                    f.write(json.dumps(entry) + "\n")
    
    print(f"\nExtraction complete! Files saved to the 'friends_data' directory.")
    print("The following files were created:")
    print("- friends_conversations.txt: All conversations in chronological order")
    print("- [Character_Name].txt: Character-specific files with context and responses")
    print("- [Character_Name]_dialogues.json: Structured JSON with context and responses")
    print("- friends_gpt_training.jsonl: JSONL format ready for GPT fine-tuning")

if __name__ == "__main__":
    extract_friends_dialogue()