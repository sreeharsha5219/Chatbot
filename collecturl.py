import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Ensure necessary NLTK datasets are downloaded
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def collect_urls(start_urls, depth=2, max_urls=25):
    urls = set(start_urls)
    collected_urls = set()
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:88.0) Gecko/20100101 Firefox/88.0'}

    while depth > 0 and len(collected_urls) < max_urls:
        new_urls = set()
        for url in urls:
            try:
                response = requests.get(url, headers=headers)
                soup = BeautifulSoup(response.content, 'html.parser')
                for a_tag in soup.find_all('a', href=True):
                    link = urljoin(url, a_tag['href'])
                    if link not in collected_urls:
                        new_urls.add(link)
                        if len(collected_urls) + len(new_urls) >= max_urls:
                            break
            except requests.exceptions.RequestException as e:
                print(f"Failed to access {url}: {e}")
        collected_urls.update(urls)
        urls = new_urls - collected_urls
        depth -= 1
    return list(collected_urls)[:max_urls]

def scrape_and_store(urls):
    headers = {'User-Agent': 'Mozilla/5.0'}
    valid_urls = 0
    for i, url in enumerate(urls):
        try:
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text()
            file_path = f"scraped_text_{i}.txt"
            with open(file_path, "w", encoding='utf-8') as file:
                file.write(text)
            valid_urls += 1
        except requests.exceptions.RequestException as e:
            print(f"Failed to scrape {url}: {e}")
    return valid_urls

def clean_text_files(file_count):
    for i in range(file_count):
        file_path = f"scraped_text_{i}.txt"
        try:
            with open(file_path, "r", encoding='utf-8') as file:
                text = file.read()
        except IOError as e:
            print(f"Failed to read {file_path}: {e}")
            continue
            
        text = text.lower()
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Tokenize sentences (stop words included)
        sentences = nltk.sent_tokenize(text)
        cleaned_sentences = sentences  # Directly use sentences without filtering stop words
        
        cleaned_text = '. '.join(cleaned_sentences)
        cleaned_file_path = f"cleaned_text_{i}.txt"
        try:
            with open(cleaned_file_path, "w", encoding='utf-8') as outfile:
                outfile.write(cleaned_text)
        except IOError as e:
            print(f"Failed to write to {cleaned_file_path}: {e}")

def extract_important_terms(file_count, top_n=25):
    vectorizer = TfidfVectorizer(max_features=1000)
    texts = []
    for i in range(file_count):
        cleaned_file_path = f"cleaned_text_{i}.txt"
        with open(cleaned_file_path, "r", encoding='utf-8') as file:
            texts.append(file.read())
    tfidf_matrix = vectorizer.fit_transform(texts)
    scores = np.sum(tfidf_matrix.toarray(), axis=0)
    terms = vectorizer.get_feature_names_out()
    sorted_indices = np.argsort(scores)[::-1]
    top_terms = terms[sorted_indices][:top_n]
    return top_terms, texts

def build_knowledge_base(texts, important_terms):
    knowledge_base = {term: [] for term in important_terms}
    for i, cleaned_file in enumerate(texts):
        sentences = cleaned_file.split('.')
        for sentence in sentences:
            for term in important_terms:
                if term in sentence:
                    knowledge_base[term].append(sentence.strip())
    return knowledge_base

# Example usage
start_urls = ['https://www.racefans.net/lewis-hamilton/', 'https://us.motorsport.com/driver/lewis-hamilton/463153/', 'https://en.wikipedia.org/wiki/Lewis_Hamilton']
urls = collect_urls(start_urls, depth=2, max_urls=25)
valid_files_count = scrape_and_store(urls)
clean_text_files(valid_files_count)
top_terms, texts = extract_important_terms(valid_files_count, top_n=60)
print(top_terms)

important_terms = ['formula', 'hamilton', 'one', 'driver', 'f1', 'season', 'mercedes', 'championship','drivers' 'season','prix','circuits' ,'racefans', 'sport']
knowledge_base = build_knowledge_base(texts, important_terms)

#print(knowledge_base)
#to see the knowledge_base. remove above comments and run it.

# Serialize the knowledge base for persistence
with open('knowledge_base.pkl', 'wb') as f:
    pickle.dump(knowledge_base, f)

# Function to search the knowledge base
def search_knowledge_base(term):
    with open('knowledge_base.pkl', 'rb') as f:
        knowledge_base = pickle.load(f)
    return knowledge_base.get(term, [])

# Example search
# facts_about_hamilton = search_knowledge_base('hamilton')
# for fact in facts_about_hamilton:
#     print(fact)
