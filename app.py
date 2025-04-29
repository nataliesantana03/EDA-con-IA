import streamlit as st
import pickle
import re
import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier

tweets = [
    "me siento triste y solo", "odio todo", "nada tiene sentido",
    "amo la vida", "qué día tan hermoso", "esto es increíble",
    "estoy llorando", "quiero desaparecer", "me rompieron el corazón",
    "gracias por tanto", "qué emoción", "vamos con todo", "estoy feliz", "logré mi meta"
]
labels = [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1]  

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(tweets)
y = np.array(labels)

model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
model.fit(X, y)

def clean_text(text):
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)  
    text = re.sub(r'#', '', text)               
    text = re.sub(r'RT[\s]+', '', text)         
    text = re.sub(r'https?:\/\/\S+', '', text)   
    return text

def predict_tweet(text):
    text = clean_text(text)
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)

    if prediction[0] == 1:
        reasons = [
            "✅ Contiene palabras poderosas que generan engagement.",
            "✅ Usa emociones que la gente suele compartir.",
            "✅ Tiene buena estructura y es fácil de leer."
        ]
        return "🔥 Viral", random.choice(reasons)
    else:
        reasons = [
            "❌ Falta emoción o llamado a la acción.",
            "❌ El texto es muy genérico o poco atractivo.",
            "❌ Carece de palabras que impulsen interacción."
        ]
        return "🧊 No Viral", random.choice(reasons)

st.set_page_config(page_title="Predicción de Tweets Virales", page_icon="✨", layout="centered")

st.markdown("""
    <body>
<style>
    body {
        background-color: #0E1117;
    }
    .big-title {
        font-size:50px;
        color:#FF4B4B;
        text-align:center;
        font-weight:bold;
        margin-bottom:20px;
        
}
    </style>
    <div class="big-title">🔮 Predicción de Tweets Virales</div>
    </body>

""", unsafe_allow_html=True)

st.write("Escribe tu tweet y predice si se volverá *viral* o no. ¡No subestimes tu poder de influencia! 🚀")

tweet_input = st.text_area("✍ Escribe tu tweet aquí:")

if st.button("🔎 Predecir"):
    if tweet_input.strip() == "":
        st.warning("⚠ Por favor, escribe algo para predecir.")
    else:
        result, reason = predict_tweet(tweet_input)
        st.markdown(f"## 📢 Resultado: *{result}*")
        st.info(f"📋 Razón: {reason}")
        
        st.markdown("""
<style>
.x-button {
    margin-top: 2em;
    display: flex;
    justify-content: center;
}
.x-button a {
    background: black;
    color: white;
    text-decoration: none;
    padding: 12px 24px;
    border-radius: 30px;
    font-weight: bold;
    font-size: 16px;
    display: flex;
    align-items: center;
    gap: 10px;
    transition: all 0.3s ease;
}
.x-button a:hover {
    background: #1A1A1A;
}
.x-button img {
    height: 24px;
    width: 24px;
}
</style>

<div class="x-button">
    <a href="https://x.com/intent/tweet?text=%s" target="_blank">
        <img src="https://abs.twimg.com/favicons/twitter.2.ico" alt="X logo">
        Publica tu tweet
    </a>
</div>
""" % tweet_input, unsafe_allow_html=True)
