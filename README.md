# Prediction Real-fake-news
Ce projet vise a classifier les articles d’actualites comme reels ou faux en
utilisant **Random Forest** et **Linear Regression**.

## Resultat 
![Interface](images/interface.png)

## Fonctionnalités

- **Prétraitement des données** : Nettoyage du texte, suppression des stop words, et vectorisation avec TF-IDF.
- **Visualisations** : Nuages de mots et distribution des labels.
- **Modèles de Machine Learning** : 
  - Régression logistique.
  - Forêt aléatoire (Random Forest).
- **Interface Utilisateur** : Une application Streamlit pour entraîner les modèles et prédire si une news est "Fake" ou "Real".

  
---
## Installation

1. Clonez ce dépôt :
   
   ```bash
   git clone https://github.com/om-bg/Prediction_Real-fake-news.git
   cd Prediction_Real-fake-news
3. Installez les dépendances :
   
   ```bash
   pip install -r requirements.txt
5. Téléchargez les ressources nécessaires pour NLTK (stop words)
   
   ```bash
   python -c "import nltk; nltk.download('stopwords')"

---
## Utilisation
1. Lancer l'application Streamlit
   
   ```bash
   streamlit run app.py

2. Sélectionnez un modèle (Régression logistique ou Forêt aléatoire).
 
3. Faire une prédiction
- Entrez un texte de news dans la zone de texte.
- Cliquez sur "Prédire" pour obtenir le résultat ("Fake News" ou "Real News").







