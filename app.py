from flask import Flask, render_template, request
import time
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Configuración de NLTK
ruta_nltk = r"C:\Users\david.rodriguez\Documents\DD\IA\Proyecto_IA_1164619\nltk_data"
nltk.data.path.clear()
nltk.data.path.append(ruta_nltk)

# Cargar modelo entrenado
with open('model/sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Inicializar recursos NLTK
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words]
    return ' '.join(tokens)

def predict_sentiment(tweet):
    # Medir el tiempo de inicio
    start_time = time.time()

    tweet = preprocess(tweet)
    tokens = word_tokenize(tweet.lower())
    log_probs = model['log_probs']
    class_priors = model['class_priors']
    
    # Inicializar variables
    max_log_prob = float('-inf')
    predicted_class = None

    # Calcular probabilidad logarítmica para cada clase
    for label in log_probs:
        prob = class_priors[label]
        for word in tokens:
            if word in log_probs[label]:
                prob += log_probs[label][word]
        if prob > max_log_prob:
            max_log_prob = prob
            predicted_class = label
    
    # Medir el tiempo de fin
    end_time = time.time()
    
    # Calcular el tiempo transcurrido en milisegundos y redondearlo a 2 decimales
    elapsed_time_ms = round((end_time - start_time) * 1000, 2)
    
    return predicted_class, elapsed_time_ms


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    tweet = request.form['tweet']
    sentiment, elapsed_time = predict_sentiment(tweet)
    
    return render_template('results.html', tweet=tweet, sentiment=sentiment, time=elapsed_time)

if __name__ == '__main__':
    app.run(debug=True)
