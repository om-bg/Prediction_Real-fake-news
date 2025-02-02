from tracemalloc import stop
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def plot_label_distribution(data):
    # Distribution des labels (Fake vs Real)
    sns.countplot(x='label', data=data)
    plt.title('Distribution of News Labels')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.show()

def generate_wordcloud(text, title, color="black"):
    # Fonction de génération de nuages de mots
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color=color, 
        stopwords=stop, 
        max_words=200, 
        colormap="viridis"
    ).generate(text)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.show()

def generate_text_wordclouds(data):
    # Nuage de mots pour Fake, Real, et l'ensemble des données
    fake_text = ' '.join(data[data['label'] == 'FAKE']['text'])
    real_text = ' '.join(data[data['label'] == 'REAL']['text'])
    all_text = ' '.join(data['text'])
    
    generate_wordcloud(fake_text, "Word Cloud - Fake News", color="red")
    generate_wordcloud(real_text, "Word Cloud - Real News", color="blue")
    generate_wordcloud(all_text, "Word Cloud - All News", color="white")
