import random

random_responses = ["That is quite interesting, please tell me more.",
                    "I see. Do go on.",
                    "Why do you say that?",
                    "Funny weather we've been having, isn't it?",
                    "Let's change the subject.",
                    "Did you catch the game last night?"]

print("Hello, my name is simple bot.")
print("You can end this conversation by simple typing: exit")
print("After typing each answer, press 'enter'")
print("How are you today?")

bot_running = True

# prevent repititions:
last_sentence = 0
res = 1

while bot_running:
    in_txt = input()
    if in_txt == "exit":
        bot_running = False
    else:
        res = random.choice(random_responses)
        if res == last_sentence:
            while res == last_sentence:
                res = random.choice(random_responses)
        last_sentence = res
        print(res)
        
