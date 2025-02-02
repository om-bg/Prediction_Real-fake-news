import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

def load_data(file_path):
    
    data = pd.read_csv(file_path)
    # Prétraiter le jeu de données
    data = preprocess_data(data)
    return data

def preprocess_data(data):
    # Supprimer les colonnes inutiles
    data = data.drop(columns=["Unnamed: 0"])
    
    # Nettoyer les textes
    data['text'] = data['text'].apply(wordopt)
    
    # Supprimer les mots vides (stop words)
    stop = stopwords.words('english')
    data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop]))
    
    return data

def wordopt(text):
    # Nettoyage du texte
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\\W", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text
