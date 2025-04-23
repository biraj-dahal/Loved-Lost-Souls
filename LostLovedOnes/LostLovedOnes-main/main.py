# main.py

from character_bot import FriendsCharacterBot

def choose_character():
    characters = [
        "Rachel Green", 
        "Ross Geller", 
        "Chandler Bing", 
        "Monica Geller", 
        "Joey Tribbiani", 
        "Phoebe Buffay"
    ]
    print("Who would you like to chat with?")
    for idx, name in enumerate(characters, 1):
        print(f"{idx}. {name}")
    
    while True:
        try:
            choice = int(input("Enter the number of your choice: "))
            if 1 <= choice <= len(characters):
                return characters[choice - 1]
        except ValueError:
            pass
        print("Invalid choice. Try again.")

if __name__ == "__main__":
    chosen_character = choose_character()
    bot = FriendsCharacterBot(chosen_character)
    bot.chat()
