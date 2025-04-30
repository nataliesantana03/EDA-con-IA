import streamlit as st
import pickle
import re
import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier

# esta es la base de entrenamiento
tweets = [
"me siento triste y solo", "odio todo", "nada tiene sentido",
"amo la vida", "qué día tan hermoso", "esto es increíble",
"estoy llorando", "quiero desaparecer", "me rompieron el corazón",
"gracias por tanto", "qué emoción", "vamos con todo", "estoy feliz", "logré mi meta"
]
labels = [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1]

sugerencias_positivas = [
"Hoy cumplí una meta que me costó lágrimas. Valió cada segundo.",
"Nada como el sol, el café y la tranquilidad de saber que estás bien.",
"Si estás leyendo esto, te mereces todo lo bueno que viene.",
"Día productivo, mente tranquila. Vamos por más.",
"Son las pequeñas victorias las que hacen grande la vida."
]

sugerencias_negativas = [
"A veces solo quiero desaparecer por un rato y que nadie lo note.",
"No sé si estoy triste o solo muy cansado de fingir que todo está bien.",
"Hoy fue uno de esos días donde nada tiene sentido.",
"Me estoy perdiendo en mí misma y no sé cómo volver.",
"Hay heridas que ni el tiempo quiere curar."
]


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

# recomenda tweets similares a los que la persona escribe
def predict_tweet(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)

    if prediction[0] == 1:
        result = "Viral"
        reason = random.choice([
        "Contiene palabras poderosas que generan engagement.",
        "Usa emociones que la gente suele compartir.",
        "Tiene buena estructura y es fácil de leer."
        ])
    else:
        result = "No Viral"
        reason = random.choice([
        "Falta emoción o llamado a la acción.",
        "El texto es muy genérico o poco atractivo.",
        "Carece de palabras que impulsen interacción."
        ])

    recommendations = random.sample(sugerencias_negativas, 2)
    return result, reason, recommendations

st.set_page_config(page_title="Predicción de Tweets", layout="centered")

# estilo css
st.markdown(""" 
<style>
    html, body, [data-testid="stApp"] {
    background-color: #0d0d0d !important;
    color: white !important;
    font-family: 'Segoe UI', sans-serif;
    }

.title {
    font-size: 3rem;
    text-align: center;
    color: #00ffe7;
    text-shadow: 0 0 20px #00ffe7;
    font-weight: bold;
    margin-bottom: 1rem;
    }

.subtitle {
    text-align: center;
    font-size: 1.2rem;
    color: #b3b3b3;
    margin-bottom: 2rem;
    }

textarea {
    background: linear-gradient(145deg, #1a1a1a, #121212);
    border: 2px solid #00ffff;
    border-radius: 12px;
    color: white;
    padding: 1rem;
    width: 100%;
    font-size: 1rem;
    box-shadow: 0 0 8px #00ffff88;
    transition: all 0.3s ease-in-out;
    resize: none;
    }

textarea:focus {
    outline: none;
    box-shadow: 0 0 15px #00ffffcc;
    border-color: #00ffff;
    }
    
textarea::placeholder {
    color: white;
    opacity: 1.6; /* puedes ajustarlo si quieres que sea más o menos visible */
}

.stButton > button {
    background: linear-gradient(to right, #00b7ff);
    color: white;
    font-weight: bold;
    padding: 0.6em 2em;
    border-radius: 30px;
    border: none;
    font-size: 1rem;
    transition: all 0.3s ease;
    }

.stButton > button:hover {
    box-shadow: 0 0 25px #00ffe7, 0 0 45px #00b7ff;
    transform: scale(1.05);
    color: white;
    cursor: pointer;
    }
    
.stButton > button:focus,
.stButton > button:active {
    color: white !important;
    background: linear-gradient(to right, #00b7ff, #00ffe7) !important;
    box-shadow: 0 0 25px #00ffe7, 0 0 45px #00b7ff !important;
    outline: none !important;
    border: none !important;
    }

.result {
    font-size: 1.5rem;
    font-weight: bold;
    color: #12cdea;
    text-align: center;
    }

.reason {
    background-color: rgba(0, 255, 231, 0.1);
    border-left: 5px solid #00ffe7;
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🔮 Predice la Viralidad de tu Tweet</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Escribe tu tweet y predice si se volverá <i>viral</i> o no. ¡No subestimes tu poder de influencia!</div>', unsafe_allow_html=True)

tweet_input = st.text_area("Escribe tu tweet aquí:")

# la acción para predecir
if st.button("Predecir"):
    if not tweet_input.strip():
        st.warning("Por favor, escribe un tweet primero.")
    else:
        result, reason, recommendations = predict_tweet(tweet_input)
        st.markdown(f"<div class='result'>Resultado: {result}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='reason'>Razón: {reason}</div>", unsafe_allow_html=True)

        st.markdown("### Tweets recomendados basados en tu estilo:")
        for rec in recommendations:
            st.markdown(f"<div class='tweet-card'>{rec}</div>", unsafe_allow_html=True)

        # Botón para publicar en X
        tweet_encoded = tweet_input.replace(" ", "%20")
        st.markdown(f"""
            <div style='text-align:center; margin-top: 30px;'>
            <a href="https://x.com/intent/tweet?text={tweet_encoded}" target="_blank"
            style="background-color:#000; color:#fff; padding:12px 30px;
            border-radius:25px; text-decoration:none; font-weight:bold;">
            Publicar
</a>
</div>
""", unsafe_allow_html=True)
