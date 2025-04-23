
from convokit import Corpus, download
import pandas as pd
import json
import os
from collections import Counter, defaultdict
import pprint


corpus = Corpus(filename=download("friends-corpus"))
# corpus.print_summary_stats()
# print(len(corpus.speakers))
# pprint.pp(corpus.speakers["Chandler Bing"])

utter_ct = 1
cur_speaker = "Rachel Green"
speaker = corpus.get_speaker(cur_speaker)
for conv in speaker.iter_conversations():
    for uter in conv.iter_utterances():
        if uter.speaker.id == cur_speaker:
            print("\nUtter Number: {}".format(utter_ct))
            if uter.reply_to:
                print("Question: {}".format(corpus.get_utterance(uter.reply_to).speaker.id), end = ": ")
                print(corpus.get_utterance(uter.reply_to).text)
            else:
                print("Question: Starting of Convo")
            print("Reply: {}".format(uter.speaker.id), end = ": ")
            print(uter.text)
            utter_ct += 1

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
