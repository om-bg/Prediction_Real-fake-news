from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

def encode_labels(data):
    # Encodage des labels
    label_encoder = LabelEncoder()
    data["label"] = label_encoder.fit_transform(data["label"])
    return data

def vectorize_text(data):
    # Conversion des textes en vecteurs numériques avec Tfidf
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['text'])
    return X, vectorizer
