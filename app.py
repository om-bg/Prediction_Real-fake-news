import streamlit as st
import pandas as pd
import os
import pickle
from final import predict_news

# Fonction pour charger un modèle enregistré
def load_saved_model(model_name):
    model_path = f"{model_name}.pkl"
    vectorizer_path = "vectorizer.pkl"

    if not os.path.exists(model_path):
        st.error(f"Le fichier {model_path} est introuvable. Veuillez d'abord entraîner le modèle.")
        return None, None

    if not os.path.exists(vectorizer_path):
        st.error("Le fichier vectorizer.pkl est introuvable. Veuillez d'abord entraîner un modèle.")
        return None, None

    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)

    with open(vectorizer_path, "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    return model, vectorizer

# Interface Streamlit
def main():
    st.title("📰 Fake News Detection")

    # Sélectionner le modèle
    model_choice = st.radio("Choisissez un modèle :", ["Logistic Regression", "Random Forest"])

    # Correspondance entre le choix et le fichier du modèle
    model_name = "logistic_model" if model_choice == "Logistic Regression" else "random_forest"

    # Charger le modèle et le vectoriseur
    model, vectorizer = load_saved_model(model_name)

    # Vérifier si le modèle est bien chargé
    if model is not None and vectorizer is not None:
        # Entrée du texte de l'utilisateur
        news_text = st.text_area("✍️ Entrez l'article de presse :")

        # Prédiction avec le modèle sélectionné
        if st.button("🔍 Prédire"):
            if news_text:
                prediction = predict_news(model, vectorizer, news_text)
                st.success(f"🧐 **Résultat** : {prediction}")
            else:
                st.warning("⚠️ Veuillez entrer un texte.")
    else:
        st.warning("📌 Veuillez d'abord entraîner et enregistrer un modèle.")

if __name__ == "__main__":
    main()
