
from convokit import Corpus, download
import pandas as pd
import json
import os
from collections import Counter, defaultdict
import pprint

main_characters = [
        "Rachel Green", 
        "Ross Geller", 
        "Chandler Bing", 
        "Monica Geller", 
        "Joey Tribbiani", 
        "Phoebe Buffay"
    ]
corpus = Corpus(filename=download("friends-corpus"))
# corpus.print_summary_stats()
# print(len(corpus.speakers))
# pprint.pp(corpus.speakers["Chandler Bing"])
training_data_dir = "indiv_data"
os.makedirs(training_data_dir, exist_ok = True)

utterance_hashmap = {} 
for cur_speaker in main_characters:
    # Open one JSONL file per character
    filename = os.path.join(training_data_dir, f"{cur_speaker.replace(' ', '_')}.jsonl")
    with open(filename, "w", encoding="utf-8") as f:
        # Get all conversations the speaker is involved in
        speaker = corpus.get_speaker(cur_speaker)
        for conv in speaker.iter_conversations():
            for uter in conv.iter_utterances():
                if uter.speaker.id == cur_speaker:
                    if uter.reply_to:
                        prompt_utt = corpus.get_utterance(uter.reply_to)
                        prompt = prompt_utt.text.strip().replace("\n", " ")
                        response = uter.text.strip().replace("\n", " ")

                        if prompt and response:
                            json_obj = {
                                "prompt": prompt,
                                "response": f"{cur_speaker}: " + response
                            }
                            f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")

#Phoebe = 7539
#Monica = 8498
#Ross = 9161
#Chandler = 8568
#Joey = 8215
#Rachel = 9331



# i = 0
# for uter in corpus.iter_utterances():
#     if uter.reply_to:
#        print(corpus.get_utterance(uter.reply_to).speaker, end = ": ")
#        print(corpus.get_utterance(uter.reply_to).text)
#     print(uter.speaker, end =": ")
#     print(uter.text)
#     print("\n")
#     if i == 1:
#         break
#     i+=1
