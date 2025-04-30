Predicción de Tweets Virales con Machine Learning (EDA con IA)

Este proyecto utiliza técnicas de procesamiento de lenguaje natural y modelos de clasificación para predecir si un tweet tiene potencial de volverse viral o no. 
Es una herramienta interactiva creada con (Streamlit) que permite al usuario escribir un tweet y recibir una predicción instantánea sobre su impacto potencial.

Tecnologías y Herramientas

1. Python 
2. Pandas y NumPy
3. Scikit-learn
4. TfidfVectorizer
5. LogisticRegression
6. Streamlit 
7. Pickle
8. CSS 
9. Dataset: https://www.kaggle.com/datasets/kazanova/sentiment140

Cómo funciona

1. Se entrena un modelo de regresión logística con un subconjunto del dataset (Sentiment140) que contiene 1.6 millones de tweets clasificados como positivos o negativos.
2. Se vectorizan los textos usando (TF-IDF)
3. El modelo predice si el contenido tiene características que lo harían viral (positivo) o no (negativo).
4. Se construye una app web con Streamlit donde el usuario puede:
   - Escribir su tweet 
   - Obtener una predicción instantánea 
   - Ver una razón explicativa del resultado 
   - Y compartirlo directamente en la red social X

nota: Desafortunadamente no puede gargar el Data ya que pesa mucho y luego no podia subir el proyecto pero mas arriba deje el link de descarga por si llegase a ser util en algun momento.
