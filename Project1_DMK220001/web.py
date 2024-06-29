import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re
import pickle
from pathlib import Path
import string

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

starter_url = "https://www.w3schools.com/dsa/dsa_intro.php"

# Creating directories in the current directory
project_directory = Path("./Files")
raw_directory = project_directory / "raw"
clean_directory = project_directory / "clean"

project_directory.mkdir(parents=True, exist_ok=True)
raw_directory.mkdir(exist_ok=True)
clean_directory.mkdir(exist_ok=True)

def scrape_and_save(urls):
    for i, url in enumerate(urls):
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            text = ' '.join(para.get_text() for para in paragraphs)
            sentences = sent_tokenize(text)
            filename = raw_directory / f"raw_content_{i}.txt"
            with open(filename, 'w', encoding='utf-8') as file:
                file.write('\n'.join(sentences))
            print(f"Content saved to {filename}")
        except requests.RequestException as e:
            print(f"Failed to fetch {url}: {e}")

def clean_text_files():
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    for filepath in raw_directory.glob("*.txt"):
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read().lower()
            text = ''.join(char for char in text if char not in string.punctuation)
            words = word_tokenize(text)
            words = [word for word in words if word not in stop_words]
            words = [lemmatizer.lemmatize(word) for word in words]
            cleaned_text = ' '.join(words)
        cleaned_filepath = clean_directory / filepath.name.replace('raw_', 'clean_')
        with open(cleaned_filepath, 'w', encoding='utf-8') as cleaned_file:
            cleaned_file.write(cleaned_text)
        print(f"Cleaned text saved to {cleaned_filepath}")

def extract_top_terms_from_all_files(top_n=25):
    texts = [file.read_text(encoding='utf-8') for file in clean_directory.glob("*.txt")]
    if not texts:
        print("No cleaned text files found.")
        return
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    try:
        feature_names = vectorizer.get_feature_names_out()  # Try to use the new method
    except AttributeError:
        feature_names = vectorizer.get_feature_names()  # Fallback to the old method if the new one isn't available
    aggregated_scores = np.sum(tfidf_matrix, axis=0)
    scores = np.squeeze(np.asarray(aggregated_scores))
    top_indices = scores.argsort()[-top_n:][::-1]
    top_terms = [feature_names[index] for index in top_indices]
    print(f"Top {top_n} terms across all documents are: {top_terms}")
    return top_terms


def find_sentences_with_terms(terms):
    sentences_dict = {term: [] for term in terms}
    for filepath in raw_directory.glob("*.txt"):
        text = filepath.read_text(encoding='utf-8')
        sentences = sent_tokenize(text)
        for sentence in sentences:
            for term in terms:
                if re.search(rf'\b{re.escape(term)}\b', sentence, re.IGNORECASE):
                    sentences_dict[term].append(sentence)
    return sentences_dict

def main():
    links = BeautifulSoup(requests.get(starter_url).text, 'html.parser').select('div#bodyContent a[href^="/wiki/"]:not([href*=":"])')
    urls = [urljoin('https://en.wikipedia.org', link['href']) for link in links][:100]
    scrape_and_save(urls)
    clean_text_files()
    top_terms = extract_top_terms_from_all_files(top_n=40)
    sentences_dict = find_sentences_with_terms(top_terms)
    with open(project_directory / 'knowledge_base.pkl', 'wb') as file:
        pickle.dump(sentences_dict, file)
    print("Knowledge base has been pickled and saved.")

if __name__ == "__main__":
    main()
