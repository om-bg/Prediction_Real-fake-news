from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train_model(X_train, y_train, model_type='logistic_regression'):
    # Entraîner le modèle choisi
    if model_type == 'logistic_regression':
        model = LogisticRegression()
    elif model_type == 'random_forest':
        model = RandomForestClassifier(random_state=0)
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    # Prédiction et évaluation du modèle
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    return accuracy, report, cm

def print_confusion_matrix(cm):
    # Afficher la matrice de confusion
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Prediction')
    plt.ylabel('Truth')
    plt.show()
