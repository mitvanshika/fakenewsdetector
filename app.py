import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# ── Load saved model & vectorizer ─────────────────────────
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ── Same clean function as before ─────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    return ' '.join(tokens)

# ── UI ────────────────────────────────────────────────────
st.set_page_config(page_title="Fake News Detector", page_icon="🔍")

st.title("🔍 Fake News Detector")
st.write("Paste a news article below and the model will predict whether it's real or fake.")

user_input = st.text_area("📰 Paste article text here", height=200)

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please paste some text first!")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        score = model.decision_function(vectorized)[0]
        confidence = min(abs(score) * 20, 100)

        if prediction == 1:
            st.success(f"🟢 REAL NEWS — Confidence: {confidence:.1f}%")
        else:
            st.error(f"🔴 FAKE NEWS — Confidence: {confidence:.1f}%")

        with st.expander("See cleaned text"):
            st.write(cleaned)