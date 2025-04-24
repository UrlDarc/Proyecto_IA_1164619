import pandas as pd
import re
import math
import pickle
from collections import defaultdict

# --- NLTK configuración y recursos ---
import nltk

# Ruta personalizada para recursos NLTK
ruta_nltk = r"C:\Users\david.rodriguez\Documents\DD\IA\Proyecto_IA_1164619\nltk_data"
nltk.data.path.clear()
nltk.data.path.append(ruta_nltk)

nltk.download('punkt', download_dir=ruta_nltk)
nltk.download('stopwords', download_dir=ruta_nltk)
nltk.download('wordnet', download_dir=ruta_nltk)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- Cargar dataset ---
df = pd.read_csv('dataset/Tweets.csv')

# --- Limpieza de texto ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|@\w+|#\w+|[^a-zA-Z\s]", '', text)
    return text

df['cleaned_text'] = df['text'].apply(clean_text)

# --- Preprocesamiento ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words]
    return ' '.join(tokens)

df['lemmatized_text'] = df['cleaned_text'].apply(preprocess)

# --- Preparar datos ---
data = list(zip(df['lemmatized_text'], df['sentiment']))

# --- División en entrenamiento y prueba ---
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# --- Entrenamiento Naive Bayes ---
class_counts = defaultdict(int)
word_counts = {}
vocab = set()

for text, label in train_data:
    class_counts[label] += 1
    if label not in word_counts:
        word_counts[label] = {}
    for word in text.split():
        word_counts[label][word] = word_counts[label].get(word, 0) + 1
        vocab.add(word)

total_docs = sum(class_counts.values())
log_probs = {}
class_priors = {}

for label in class_counts:
    total_words = sum(word_counts[label].values())
    class_priors[label] = math.log(class_counts[label] / total_docs)
    log_probs[label] = {
        word: math.log((word_counts[label].get(word, 0) + 1) / (total_words + len(vocab)))
        for word in vocab
    }

# --- Predicción sobre test_data ---
def predict(text):
    tokens = text.split()
    scores = {}
    for label in class_priors:
        score = class_priors[label]
        for word in tokens:
            if word in vocab:
                score += log_probs[label].get(word, math.log(1 / (sum(word_counts[label].values()) + len(vocab))))
        scores[label] = score
    return max(scores, key=scores.get)

y_true = [label for _, label in test_data]
y_pred = [predict(text) for text, _ in test_data]

# --- Evaluación ---
print("📊 Evaluación del modelo:")
print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")
print("\nClassification Report:")
report = classification_report(y_true, y_pred, digits=3)
print(report)

with open('model/metrics_report.txt', 'w') as f:
    f.write("Accuracy: {:.3f}\n\n".format(accuracy_score(y_true, y_pred)))
    f.write(report)

# --- Guardar modelo ---
model = {
    'log_probs': log_probs,
    'class_priors': class_priors,
    'vocab': vocab,
    'word_counts': word_counts
}

with open('model/sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Modelo entrenado y guardado exitosamente.")
