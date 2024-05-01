import random
import json

import torch

from model import NeuralNetwork
from nltk_utils import bag_of_words, tokenize

# Check if CUDA (GPU) is available, else use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNetwork(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Bot"

def get_response(msg):
    sentence = tokenize(msg)  # Tokenize the input message
    X = bag_of_words(sentence, all_words) # Convert tokenized sentence to bag of words representation
    X = X.reshape(1, X.shape[0]) # get first index (fit)
    X = torch.from_numpy(X).to(device)  # Convert X to PyTorch tensor and move it to the appropriate device (GPU/CPU)


    output = model(X)
    _, predicted = torch.max(output, dim=1) # index of the maximum value

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses']), prob.item()
    
    return "I do not understand...", prob.item()

if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp, prob = get_response(sentence)
        print("Bot:", resp)
        print("Confidence:", f"{prob*100:.2f}%")