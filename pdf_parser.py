from bs4 import BeautifulSoup
import requests
import fitz
import logging
import re
from functools import lru_cache
# streamlit run main.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
import transformers
from transformers import AutoTokenizer, AutoModel
from functools import lru_cache
import torch
import numpy as np
import streamlit as st
import pickle
from google.cloud import storage
from google.oauth2 import service_account
import nltk
nltk.download('punkt')
# nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
# import swifter
# from tqdm import tqdm

def parse_pdf_from_url(pdf_url):
    try:
        response = requests.get(pdf_url)
        if response.status_code == 200:
            pdf_bytes = response.content
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            # Extract text from each page
            pdf_text = ""
            for page_number in range(pdf_document.page_count):
                page = pdf_document[page_number]
                pdf_text += page.get_text()
            pdf_document.close()
        else:
            pdf_text = ''
    except Exception:
        logging.error(f"{pdf_url}")
        pdf_text = ''
    return pdf_text

#preprocessing parsed_text
def lowercase(x):
    try:
        return x.lower()
    except Exception:
        return None
    

phone_pattern = re.compile(r'([0-9\-\.]+)')
email_pattern = re.compile("([a-zA-Z0-9]+[@][a-zA-Z0-9]+[\.]{1}[a-zA-Z]{2,})")
remove_special_character_pattern = re.compile(r'[^a-zA-Z0-9\s\.\/,]')

def preprocess_text(lines):
    # Define the set of special characters and numbers to be removed from both ends
    special_chars = ' ,;.!#1234567890'

    # Define a regular expression pattern to match the specified characters at both ends
    updated_lines = []
    for l in lines:
        if 'copyright' in l or 'phone' in l or 'email' in l:
            updated_lines.append('')
        else:
            # Use re.sub to remove the specified characters from both ends
             # Define a pattern to match common website formats
            website_pattern = r'(www\.[a-zA-Z0-9-]+(\.[a-zA-Z]{2,})+)|(http[s]?://[a-zA-Z0-9-]+(\.[a-zA-Z]{2,})+)'
        
            # Combine the patterns to match both special characters at ends and websites
            combined_pattern = f'^[{re.escape(special_chars)}]+|[{re.escape(special_chars)}]+$|{website_pattern}'
        
            # Use re.sub to remove the specified characters and websites from both ends
            processed_text = re.sub(combined_pattern, '', l)
            processed_text = processed_text.strip(' ').strip('.').strip(':').strip('•').strip('°').strip('©').strip('-')

            # removing some stop words observed
            processed_text = processed_text.replace('ltd', '')
            processed_text = processed_text.replace('features', '')
            processed_text = processed_text.replace('co.', '')
            processed_text = processed_text.replace('•', '')
            
            processed_text = re.sub(phone_pattern, '', processed_text)
            processed_text = re.sub(email_pattern, '', processed_text)

            processed_text = re.sub(remove_special_character_pattern, '', processed_text)
            processed_text = processed_text.strip(' ')
            
            # remove articles in the end if any
            processed_text = processed_text.strip(' a')
            if processed_text.startswith('the ') or processed_text.endswith(' the'):
                processed_text = processed_text.strip('the')
            if processed_text.startswith('a ') or processed_text.endswith(' a'):
                processed_text = processed_text.strip('a ')
            
            processed_text = processed_text.strip()
            processed_text = lowercase(processed_text)
            
            updated_lines.append(processed_text)

    return updated_lines

def remove_special_characters(text):
    # Define a regex pattern to match non-alphanumeric and non-whitespace characters
    pattern = re.compile(r'[^a-zA-Z0-9\s\.\/,]')
    
    # Use the pattern to replace special characters with an empty string
    cleaned_text = re.sub(pattern, '', text)
    
    return cleaned_text

def remove_empty(x):
    x = [i for i in x if i != '']
    return x

light_stop_words = ['led', 'bulb', 'bulbs', 'xenon', 'filament', 'light', 'lights', 'backlight', 'daylight', 'lumen', 'lamp', 'lamps', 'lumens', 'leds','nightlight',
'lamping', 'neon', 'incandescent', 'fluorescents', 'downlights','downlight']

pattern = "(" + light_stop_words[0]
# create a substring pattern

for l in light_stop_words[1:]:
    pattern += "|" + l

pattern += ")"

light_substr_pattern = re.compile(pattern)

    

def process_url(url):
    if 'www.' in url or '.pdf' in url:
        parsed_text = parse_pdf_from_url(url)
    else:
        parsed_text = url
    
    parsed_lines  = preprocess_text(parsed_text.split('\n'))
    parsed_lines = remove_empty(parsed_lines)
    parsed_lines = list(set(parsed_lines))
    return parsed_lines

fixture_words = ['socket', 'troffer', 'holder', 'fixture', 'sensor', 'controller', 'transformer']
fixture_words_plural = []

fixture_lookaheads = ['of ', 'for ', 'without', 'with '] # usual represents association with a lighting product and not a lighting product itself

for f in fixture_words:
    fixture_words_plural.append(f+'s')

fixture_words = fixture_words_plural+fixture_words


def check_substr(substr, text):
    if substr in text:
        return True, text
    return False, text

def check_for_negative_lookaheads(light_phrase_index, predict_lighting, parsed_lines):
    try:
        light_phrase_index = int(light_phrase_index + 1)
        for lindex, l in enumerate(parsed_lines[:light_phrase_index]):
                # tokens = nltk.word_tokenize(l)
                tokens = re.split(r'[ ,.,\n]+', l)
                fixture_word_match = list(set(fixture_words) & set(tokens))
                if fixture_word_match:
                    return lindex, False
    except Exception as e:
        logging.exception("traceback as follows")
        
    return light_phrase_index, predict_lighting
            

def check_lighting(parsed_lines):
    previous_line_ends_with_word = None
    
    for lindex, l in enumerate(parsed_lines):
        tokens = nltk.word_tokenize(l)
        light_words = list(set(light_stop_words) & set(tokens))
        
        # print(f"1st rule based check - {light_words}, {tokens=}, {l=}")
        
        for lw in light_words:
                li = l.find(lw)
                
                fixture_lookaheads_index = next((l.index(word) for word in fixture_lookaheads if word in l), -1)
                
                if lindex>0:
                    previous_line_ends_with_word = any(parsed_lines[lindex-1].endswith(word) for word in fixture_lookaheads)
                    # print(f"{previous_line_ends_with_word=}")

                
                # print(f"{fixture_lookaheads_index=}, {lw=}, {l=}")
                
                if  (fixture_lookaheads_index != -1 and fixture_lookaheads_index < li) or (previous_line_ends_with_word is True):
                    light_words = None 
                    # if with or without precedes light related synonym in a line of text it describes 
                    # an association with a lighting product and not denote lighting product itself
        if light_words:
            pos_tags = nltk.pos_tag(tokens)
            for tk, tag in pos_tags:
                if tk in light_words and tag in ['NN', 'NNS', 'JJ']:
                    return lindex, l, True
    return None, '', False



# Load data
train_df = pd.read_pickle("data/train_data.pkl")
test_df = pd.read_pickle("data/test_data.pkl")

# Combine parsed lines into a single string
train_texts = train_df["parsed_lines"].apply(lambda x: " ".join(x))
test_texts = test_df["parsed_lines"].apply(lambda x: " ".join(x))

# # Load and fit TF-IDF vectorizer
# vectorizer = TfidfVectorizer(max_features=1000)
# train_features = vectorizer.fit_transform(train_texts)
# test_features = vectorizer.transform(test_texts)

try:
    with open("models/light_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
except Exception:
    # Load and fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=1000)
    train_features = vectorizer.fit_transform(train_texts)
    test_features = vectorizer.transform(test_texts)
    with open("models/light_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)


le = LabelEncoder()
train_labels_encoded = le.fit_transform(train_df['is_lighting'])
test_labels_encoded = le.transform(test_df['is_lighting'])



# with open("models/bert_tokenizer.pkl", "rb") as f:
#     tokenizer = pickle.load(f)

# bert_model = AutoModel.from_pretrained("bert-base-uncased")

# # Load pre-trained models and scalers
try:
    with open("models/bert_tokenizer.pkl", "rb") as f:
        bert_tokenizer = pickle.load(f)
except Exception:
    # Load BERT tokenizer and model
    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    with open("models/bert_tokenizer.pkl", "wb") as f:
         pickle.dump(bert_tokenizer, f)

try:
    with open("models/bert_model.pkl", "rb") as f:
        bert_model = pickle.load(f)
except Exception:
    bert_model = AutoModel.from_pretrained("bert-base-uncased")
    with open("models/bert_model.pkl", "wb") as f:
        pickle.dump(bert_model, f)
    

# with open("models/light_standard_scaler.pkl", "rb") as f:
#     scaler = pickle.load(f)

# with open("models/light_vectorizer.pkl", "rb") as f:
#     vectorizer = pickle.load(f)

# with open("models/lr_light_model.pkl", "rb") as f:
#     model = pickle.load(f)

# Tokenize each sentence in the parsed lines list
def tokenize_lines(parsed_lines):
    tokenized_lines = []
    for line in parsed_lines:
        tokens = bert_tokenizer(line, truncation=True, padding="max_length", return_tensors="pt")
        tokenized_lines.append(tokens)
    return tokenized_lines

# Extract BERT embeddings
def extract_bert_embeddings(tokenized_lines):
    bert_embeddings = []
    for tokens in tokenized_lines:
        outputs = bert_model(**tokens)
        last_hidden_state = outputs.last_hidden_state
        mean_embedding = torch.mean(last_hidden_state, dim=1)
        bert_embeddings.append(mean_embedding.detach().numpy())
    return bert_embeddings

try:
    with open("models/lr_light_model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception:
    # Apply tokenization
    train_bert_features = tokenize_lines(train_texts)
    test_bert_features = tokenize_lines(test_texts)
    
    # Apply BERT embedding extraction
    train_bert_embeddings = extract_bert_embeddings(train_bert_features)
    test_bert_embeddings = extract_bert_embeddings(test_bert_features)
    
    # Flatten the last two dimensions (words and embedding dimensions)
    train_bert_embeddings_flat = [embedding.flatten() for embedding in train_bert_embeddings]
    test_bert_embeddings_flat = [embedding.flatten() for embedding in test_bert_embeddings]
    
    train_combined_features = np.concatenate((train_features.toarray(), train_bert_embeddings_flat), axis=1)
    test_combined_features = np.concatenate((test_features.toarray(), test_bert_embeddings_flat), axis=1)

    try:
        with open("models/light_standard_scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    except Exception:
        # Standardize features
        scaler = StandardScaler()
    train_combined_features_scaled = scaler.fit_transform(train_combined_features)
    test_combined_features_scaled = scaler.transform(test_combined_features)
    with open("models/light_standard_scaler.pkl", "wb") as f:
             pickle.dump(scaler, f)
    
    # Train the model with scaled features
    model = LogisticRegression(max_iter=1000)
    model.fit(train_combined_features_scaled, train_labels_encoded)
    
    with open("models/lr_light_model.pkl", "wb") as f:
        pickle.dump(model, f)

with open("models/light_standard_scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

# Predict if lighting
@lru_cache
def predict_if_lighting(x):
    test_texts = [x]
    test_features = vectorizer.transform(test_texts)
    test_bert_features = tokenize_lines(test_texts)
    test_bert_embeddings = extract_bert_embeddings(test_bert_features)
    test_bert_embeddings_flat = [embedding.flatten() for embedding in test_bert_embeddings]
    test_combined_features = np.concatenate((test_features.toarray(), test_bert_embeddings_flat), axis=1)
    test_combined_features_scaled = scaler.transform(test_combined_features)
    test_predictions = model.predict(test_combined_features_scaled)
    test_probabilities = model.predict_proba(test_combined_features_scaled)
    confidence_score = test_probabilities[0, 1]
    return True if test_predictions[0] == 1 else False, confidence_score




def predict_luminosity_from_url(text, bulk_process: bool = False):
    
    if bulk_process is False:
        processed_lines = process_url(text)
    else:
        processed_lines = text 
    
    light_phrase_index, light_phrase, rule_based_prediction = check_lighting(processed_lines)
    
    # if first level of rule based prediction results in True, additional check to reject if it denotes light fixtures/add-ons
    if light_phrase:
        light_phrase_index, rule_based_prediction = check_for_negative_lookaheads(light_phrase_index, rule_based_prediction, processed_lines)
    
    if rule_based_prediction is True:
        # additional check to confirm positives using trained bert model with training data
        bert_if_lighting, confidence_score = predict_if_lighting(light_phrase)
    else:
        bert_if_lighting = False
        confidence_score = 1
    
    # bert_if_lighting, confidence_score = predict_if_lighting(light_phrase)
    return bert_if_lighting and rule_based_prediction, processed_lines, light_phrase, confidence_score
