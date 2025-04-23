# character_bot.py

import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class FriendsCharacterBot:
    def __init__(self, character_name):
        self.character = character_name
        self.dialogue_data = self._load_data()
        self.contexts = ["\n".join([f"{utt['speaker']}: {utt['text']}" for utt in d["context"]]) for d in self.dialogue_data]
        self.responses = [d["response"] for d in self.dialogue_data]

        # TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer()
        self.X = self.vectorizer.fit_transform(self.contexts)
    def _load_data(self):
        file_path = f"friends_data/{self.character.replace(' ', '_')}_dialogues.json"
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def chat(self):
        print(f"\nYou are now chatting with {self.character.split()[0]}!")
        context_input = input("You: ")

        while context_input.lower() not in ["bye", "exit", "quit"]:
            context_vec = self.vectorizer.transform([context_input])
            sims = cosine_similarity(context_vec, self.X)
            best_match_idx = sims.argmax()
            response = self.responses[best_match_idx]
            print(f"{self.character.split()[0]}:", response)

            context_input = input("You: ")
