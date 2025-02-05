def predict_news(model, vectorizer, news_text):
    # Prédiction pour un article donné
    news_vectorized = vectorizer.transform([news_text])
    prediction = model.predict(news_vectorized)
    return 'Fake News' if prediction == 0 else 'Real News'
