import json
import os
import pickle
import spacy
import re

import joblib

# Path to the model file
model_filename = 'chatbot_modelNB.pkl'

# Load the model using joblib
try:
    model = joblib.load(model_filename)
    print("Model loaded successfully!")
except Exception as e:
    print("Failed to load the model:", e)


# Load the spaCy model for NLP tasks
nlp = spacy.load("en_core_web_sm")

# Function to load or create user data
def load_or_create_user_data(name):
    filename = f"{name.lower()}_data.json"
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    else:
        return {
            "name": name,
            "likes": [],
            "dislikes": [],
            "personal_info": {}
        }

# Save user data to a file
def save_user_data(user_data):
    filename = f"{user_data['name'].lower()}_data.json"
    with open(filename, 'w') as f:
        json.dump(user_data, f, indent=4)

# Extract and update user data from chat using spaCy and regex
def extract_and_update_user_data(text, user_data):
    response = ""
    # Use spaCy for extracting personal information
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "AGE":
            user_data['personal_info']['age'] = ent.text
        elif ent.label_ == "GPE":
            user_data['personal_info']['location'] = ent.text
        elif ent.label_ == "ORG" or ent.label_ == "JOB":
            user_data['personal_info']['profession'] = ent.text

    # Extract likes and dislikes using regex
    likes = re.findall(r"i like (\w+)", text, re.I)
    dislikes = re.findall(r"i dislike (\w+)", text, re.I)
    for like in likes:
        if like not in user_data['likes']:
            user_data['likes'].append(like)
            response += f"It's great that you like {like}. "
        else:
            response += f"You still like {like}, nice! "
    for dislike in dislikes:
        if dislike not in user_data['dislikes']:
            user_data['dislikes'].append(dislike)
            response += f"Noted, you dislike {dislike}. "
        else:
            response += f"Still not fond of {dislike}, I remember! "

    if not response:  # If no likes/dislikes were updated, provide a general update about personal info
        response = "Thanks for sharing more about yourself!"

    return response

# Generate response using the model or a fallback
def generate_response(input_text, name):
    try:
        response = model.generate_response(input_text)
        if not response.strip():
            raise ValueError("Empty response generated.")
        return response
    except Exception as e:
        print(f"Error: {str(e)}")
        return f"Sorry, {name}, I'm not sure how to respond to that. Can you tell me more?"

# Main chat function
def chat():
    print("Hi, I'm your Friendly ChatBot. What's your name?")
    name = input("You: ")
    user_data = load_or_create_user_data(name)
    print(f"Welcome back, {name}!" if os.path.exists(f"{name.lower()}_data.json") else f"Nice to meet you, {name}!")

    while True:
        user_input = input("You: ")
        personal_response = extract_and_update_user_data(user_input, user_data)
        if personal_response:
            print(f"Bot: {personal_response}")

        # Generate a model-based response if necessary
        if not personal_response.strip():
            generated_response = generate_response(user_input, user_data['name'])
            print(f"Bot: {generated_response}")

        save_user_data(user_data)

if __name__ == "__main__":
    chat()
