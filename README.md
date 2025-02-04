# Prediction Real-fake-news
Ce projet vise a classifier les articles d’actualites comme reels ou faux en
utilisant **Random Forest** et **Linear Regression**.

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
2. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
3. Téléchargez les ressources nécessaires pour NLTK (stop words) 
   ```bash
   python -c "import nltk; nltk.download('stopwords')"

---


