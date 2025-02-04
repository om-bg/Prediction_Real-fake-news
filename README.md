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

## Bibliothèques Nécessaires

Voici la liste des bibliothèques Python utilisées dans ce projet, ainsi que leur rôle :

| Bibliothèque          | Version   | Rôle                                                                 |
|-----------------------|-----------|----------------------------------------------------------------------|
| **streamlit**         | 1.26.0    | Création de l'interface utilisateur pour interagir avec le modèle.   |
| **pandas**            | 2.0.3     | Manipulation et analyse des données (chargement, prétraitement).     |
| **scikit-learn**      | 1.3.0     | Machine Learning (entraînement, évaluation, vectorisation TF-IDF).   |
| **nltk**              | 3.8.1     | Traitement du langage naturel (stop words, tokenisation).            |
| **matplotlib**        | 3.7.2     | Visualisation des données (graphiques, matrices de confusion).       |
| **seaborn**           | 0.12.2    | Visualisation avancée (matrices de confusion, graphiques).           |
| **wordcloud**         | 1.9.2     | Génération de nuages de mots pour visualiser les mots fréquents.     |
| **joblib**            | 1.3.2     | Sauvegarde et chargement des modèles entraînés.                      |

---

