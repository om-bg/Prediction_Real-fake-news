import pickle
from sklearn.model_selection import train_test_split
from datapreprocessing import load_data, preprocess_data
from encodage import encode_labels, vectorize_text
from train import train_model, evaluate_model

# Charger et prétraiter les données
data = load_data("news_datasets.csv")
data = encode_labels(data)
X, vectorizer = vectorize_text(data)
data = preprocess_data(data)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, data['label'], test_size=0.2, random_state=42)

# Entraîner Logistic Regression
logistic_model = train_model(X_train, y_train, model_type='logistic_regression')
logistic_accuracy, logistic_report, logistic_cm = evaluate_model(logistic_model, X_test, y_test)

# Entraîner Random Forest
rf_model = train_model(X_train, y_train, model_type='random_forest')
rf_accuracy, rf_report, rf_cm = evaluate_model(rf_model, X_test, y_test)

# Afficher les performances
print(f"Logistic Regression Accuracy: {logistic_accuracy}")
print("Classification Report:\n", logistic_report)

print(f"Random Forest Accuracy: {rf_accuracy}")
print("Classification Report:\n", rf_report)

# Sauvegarder les modèles et le vectorizer
with open("logistic_model.pkl", "wb") as logistic_file:
    pickle.dump(logistic_model, logistic_file)

with open("random_forest.pkl", "wb") as rf_file:
    pickle.dump(rf_model, rf_file)

with open("vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Les modèles et le vectorizer ont été sauvegardés avec succès!")
