# Proyecto de Análisis de Sentimientos en Tweets

Este proyecto consiste en un clasificador de sentimientos para tweets (positivo, negativo o neutro), construido desde cero usando el algoritmo Naive Bayes. El modelo se entrena con un dataset real y se expone a través de una aplicación web sencilla usando Flask.

---

## Características principales

- Entrenamiento de un clasificador Naive Bayes sin bibliotecas de machine learning.
- Limpieza, tokenización, lematización y eliminación de stopwords usando NLTK.
- Evaluación del modelo mediante accuracy, precision, recall y F1-Score.
- Interfaz web que permite ingresar un tweet y analizar su sentimiento.
- Tiempo de inferencia mostrado en milisegundos.

---

## Estructura del Proyecto y Contenido

Proyecto_IA/
│
├── dataset/
│   └── Tweets.csv               # Dataset original de tweets con sentimientos
│
├── model/
│   ├── __pycache__/             # Archivos cache generados automáticamente por Python
│   ├── naive_bayes.py           # Implementación del clasificador Naive Bayes desde cero
│   ├── train_model.py           # Script para entrenar el modelo y generar métricas
│   ├── metrics_report.txt       # Reporte de métricas de evaluación (accuracy, precision, recall, F1)
│   └── sentiment_model.pkl      # Modelo entrenado serializado con pickle
│
├── templates/
│   ├── index.html               # Formulario principal para analizar tweets
│   └── results.html             # Página de resultados con sentimiento y tiempo de inferencia
│
├── nltk_data/                   # Carpeta local con recursos descargados de NLTK (stopwords, lemmatizer, etc.)
│
├── app.py                       # Aplicación web Flask que permite análisis de sentimientos en tiempo real
└── README.md                    # Este archivo con la explicación del proyecto
└── test_tweets.txt              # Este archivo contiene diferentes tweets de pruebas (positivos, negativos y neutros)

---

## Requisitos

Python 3.8 o superior

Paquetes necesarios:
- Flask
- nltk
- pandas

Puedes instalar todas las dependencias con el siguiente comando:
- pip install flask nltk pandas

Descargar recursos de NLTK:
- punkt
- stopwords
- wordnet

---

## Métodos del Proyecto

- train_model.py
Este archivo entrena el modelo Naive Bayes utilizando el dataset Tweets.csv. Realiza los siguientes pasos:
-- Limpieza de texto: Se eliminan URLs, menciones de usuarios, hashtags y caracteres no alfabéticos.
-- Preprocesamiento: Se tokeniza el texto, se eliminan las stopwords y se lematizan las palabras.
-- Entrenamiento del modelo: Se entrena un clasificador Naive Bayes basado en la frecuencia de las palabras por clase (positivo, negativo, neutro).
-- Guardado del modelo: El modelo entrenado se guarda en un archivo .pkl para su uso posterior en la aplicación web.

- naive_bayes.py
Este archivo contiene la implementación del clasificador Naive Bayes desde cero. Los métodos incluyen:
- Calcular probabilidades logarítmicas: Calcula las probabilidades logarítmicas para las clases y las palabras dentro de cada clase.
- Clasificación de sentimiento: Usa las probabilidades logarítmicas para predecir el sentimiento de un tweet dado.

- app.py
Este archivo contiene la aplicación web Flask. Permite a los usuarios ingresar un tweet a través de un formulario y recibir un análisis de sentimiento, junto con el tiempo de inferencia en milisegundos. Además, muestra el resultado en una página de resultados.

- metrics_report.txt
Este archivo contiene un informe sobre el desempeño del modelo, con las siguientes métricas:
-- Accuracy: Proporción de clasificaciones correctas.
-- Precision: Medida de exactitud de las predicciones positivas.
-- Recall: Medida de la capacidad del modelo para encontrar todas las instancias positivas.
-- F1-Score: Media armónica entre la precisión y el recall.

- index.html
Página principal con un formulario para ingresar un tweet y analizar su sentimiento.

- results.html
Página que muestra los resultados del análisis de sentimiento y el tiempo de inferencia.

- Evaluación del Modelo
Las métricas del modelo se generaron usando un conjunto de validación. Se encuentran en el archivo model/metrics_report.txt. Incluyen:
-- Accuracy: Proporción de clasificaciones correctas.
-- Precision: Medida de exactitud de las predicciones positivas.
-- Recall: Medida de la capacidad del modelo para encontrar todas las instancias positivas.
-- F1-Score: Media armónica entre la precisión y el recall.

## Instrucciones para ejecutar

- Entrenar el modelo:
python model/train_model.py

- Ejecutar la aplicación Flask:
python app.py

- Abrir la aplicación en tu navegador:
http://localhost:5000

---

==============================================================================================
## Créditos
Dataset usado: Twitter Tweets Sentiment Dataset by yasserh (Kaggle)
URL: https://www.kaggle.com/datasets/yasserh/twitter-tweets-sentiment-dataset

Desarollador:
| - Nombre: David André Rodríguez Cano
| - Carné:  1164619
==============================================================================================