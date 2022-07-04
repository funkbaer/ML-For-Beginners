from textblob import TextBlob
from textblob.np_extractors import ConllExtractor
# import and create a Conll extractor to use later 
extractor = ConllExtractor()


# later when you need a noun phrase extractor:

lst_prior = []

def user_input():

    user_input = input("> ")
    user_input_blob = TextBlob(user_input, np_extractor=extractor)  # note non-default extractor specified
    np = user_input_blob.noun_phrases

    if user_input == "exit":
        return False
    if user_input_blob.polarity <= -0.5:
        response = "Oh dear, that sounds bad. "
    elif user_input_blob.polarity <= 0:
        response = "Hmm, that's not great. "
    elif user_input_blob.polarity <= 0.5:
        response = "Well, that sounds positive. "
    elif user_input_blob.polarity <= 1:
        response = "Wow, that sounds great. "

    for item in np:
        if item in lst_prior:
            print(f"Oh you already told me about {item} that's nice to hear" )
            return True
    
    if len(np) == 0:
        print( response + "Can you tell me more?")
    else:
        print( response + f"Can you tell me more about {np[0].pluralize()} ?")

    for item in np:
        lst_prior.append(item)

    return True


bot_running = True

print("Hello, my name is simple bot.")
print("You can end this conversation by simple typing: exit")
print("After typing each answer, press 'enter'")
print("How are you today?")

while bot_running:
    bot_running = user_input()

