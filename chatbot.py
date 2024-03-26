import os
import json
import pickle
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import spacy
import requests
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

# Load Spacy's English model for NER
nlp = spacy.load("en_core_web_sm")

# Paths
user_models_dir = "user_models"
knowledge_base_path = "knowledge_base.pkl"

# Ensure directories exist
os.makedirs(user_models_dir, exist_ok=True)

# Load or create the knowledge base
if os.path.exists(knowledge_base_path):
    with open(knowledge_base_path, 'rb') as f:
        knowledge_base = pickle.load(f)
else:
    knowledge_base = {}

def preprocess_text(text):
    tokens = [word for word in nltk.word_tokenize(text) if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if not word in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

all_sentences = [preprocess_text(sentence) for term_sentences in knowledge_base.values() for sentence in term_sentences]

vectorizer = TfidfVectorizer()
knowledge_base_vectors = vectorizer.fit_transform(all_sentences)

def perform_web_lookup(query):
    query = query.replace(' ', '+')
    url = f"https://www.google.com/search?q={query}"
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            snippet = soup.find('div', class_='BNeawe s3v9rd AP7Wnd').text
            return snippet
        else:
            return "I couldn't fetch the information due to a network error."
    except Exception as e:
        return f"An error occurred: {e}"

def hardcoded_responses(user_input):
    responses = {
        'hi': "Hello there! How can I assist you today?",
        'hello': "Hello there! How can I assist you today?",
        'bye': "Goodbye! Looking forward to our next conversation.",
        'goodbye': "Goodbye! Looking forward to our next conversation."
    }
    return responses.get(user_input.lower(), "")


def generate_response(user_input, user_model):
    response = hardcoded_responses(user_input)
    if response:
        response = ' '.join(sent_tokenize(response)[:2])

    else:
        user_input_preprocessed = preprocess_text(user_input)
        user_input_vector = vectorizer.transform([user_input_preprocessed])
        similarities = cosine_similarity(user_input_vector, knowledge_base_vectors)
        most_similar_doc_index = similarities.argmax()
        similarity_score = similarities[0, most_similar_doc_index]

        if similarity_score < 0.8:
            response = perform_web_lookup(user_input)
        else:
            response = all_sentences[most_similar_doc_index]
            if user_model['likes']:
                additional_response = " Also, I remember you like " + ", ".join(user_model['likes']) + "."
                full_response = response + additional_response
                response_words = full_response.split()
                # Ensure we include the additional response sensibly within the 30-word limit
                response = ' '.join(response_words[:60]) if len(response_words) > 60 else full_response
            else:
                response_words = response.split()
                response = ' '.join(response_words[:60])

    # Split the response into words and limit to the first 30 words
    response_words = response.split()
    limited_response = ' '.join(response_words[:30])

    return limited_response


def load_or_create_user_model(user_id):
    user_model_path = os.path.join(user_models_dir, f"{user_id}.json")
    if os.path.exists(user_model_path):
        with open(user_model_path, 'r') as file:
            return json.load(file)
    else:
        return {
            "name": "", 
            "personal_info": {}, 
            "likes": [], 
            "dislikes": [], 
            "interactions": [],
            "feedback": [] 
        }

def update_user_model(user_id, user_model):
    user_model_path = os.path.join(user_models_dir, f"{user_id}.json")
    with open(user_model_path, 'w') as file:
        json.dump(user_model, file, indent=4)

def extract_personal_info(text, user_model):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE", "LOC"]:
            user_model["personal_info"].setdefault(ent.label_, []).append(ent.text)

def chatbot_main_loop(user_id):
    user_model = load_or_create_user_model(user_id)
    print("Hello! I'm your chatbot. Ask me anything or type 'quit' to exit.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            print("Chatbot: It was nice talking to you. Goodbye!")
            break

        extract_personal_info(user_input, user_model)
        response = generate_response(user_input, user_model)
        print(f"Chatbot: {response}")
        
        user_model['interactions'].append({"query": user_input, "response": response})
        update_user_model(user_id, user_model)

if __name__ == "__main__":
    user_id = input("Please enter your user ID to start: ").strip()
    chatbot_main_loop(user_id)
