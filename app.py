import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

model       = joblib.load("model.joblib")
vectorizer  = joblib.load("vectorizer.joblib")
emotion_map = joblib.load("emotion_map.joblib")

# reverse the map: {number: emotion}
int_to_emotion = {v: k for k, v in emotion_map.items()}

def clean_text(txt):
    txt = txt.lower()
    txt = txt.translate(str.maketrans('', '', string.punctuation))
    txt = "".join(ch for ch in txt if not ch.isdigit())
    txt = "".join(ch for ch in txt if ch.isascii())
    txt = " ".join(w for w in txt.split() if w not in stop_words)
    return txt

st.title("Emotion Detector")

user_input = st.text_area("Enter text:")

if st.button("Predict"):
    if user_input.strip():
        cleaned    = clean_text(user_input)
        vectorised = vectorizer.transform([cleaned])
        prediction = model.predict(vectorised)[0]
        emotion    = int_to_emotion[prediction]
        st.success(f"Emotion: **{emotion}**")
    else:
        st.warning("Please enter some text.")