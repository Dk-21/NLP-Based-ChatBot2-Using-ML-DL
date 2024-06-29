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

starter_url = "https://www.programiz.com/dsa"

# Creating directories in the current directory
project_directory = Path("./Files")
raw_directory = project_directory / "raw"
clean_directory = project_directory / "clean"

project_directory.mkdir(parents=True, exist_ok=True)
raw_directory.mkdir(exist_ok=True)
clean_directory.mkdir(exist_ok=True)

visited_urls = set()
url_queue = [starter_url]

def scrape_and_save():
    while url_queue and len(visited_urls) < 500:
        current_url = url_queue.pop(0)
        if current_url in visited_urls:
            continue
        try:
            response = requests.get(current_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            text = ' '.join(para.get_text() for para in paragraphs)
            sentences = sent_tokenize(text)
            filename = raw_directory / f"raw_content_{len(visited_urls)}.txt"
            with open(filename, 'w', encoding='utf-8') as file:
                file.write('\n'.join(sentences))
            print(f"Content saved to {filename}")
            visited_urls.add(current_url)
            
            # Enqueue new URLs from the same domain
            links = soup.select('a[href]')
            for link in links:
                new_url = urljoin(current_url, link['href'])
                if new_url not in visited_urls:
                    url_queue.append(new_url)
        except requests.RequestException as e:
            print(f"Failed to fetch {current_url}: {e}")

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

def main():
    scrape_and_save()
    clean_text_files()

if __name__ == "__main__":
    main()
