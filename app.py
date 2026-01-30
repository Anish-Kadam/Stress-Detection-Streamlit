import streamlit as st
import pickle
import re

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="Stress Detection App",
    page_icon="🧠",
    layout="centered"
)

# -------------------------------
# Load model and vectorizer
# -------------------------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# -------------------------------
# Text cleaning function
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# -------------------------------
# UI
# -------------------------------
st.title("🧠 Stress Detection Using Machine Learning")
st.write("Enter your thoughts to check stress level")

text = st.text_area(
    "📝 Enter text here",
    height=150,
    placeholder="Example: I feel anxious and overwhelmed with work"
)

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Predict"):
    if text.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        cleaned_text = clean_text(text)
        vector = vectorizer.transform([cleaned_text])
        prediction = model.predict(vector)[0]
        confidence = model.predict_proba(vector)[0]

        if prediction == 1:
            st.error(f"⚠️ Stressed (Confidence: {confidence[1]*100:.2f}%)")
        else:
            st.success(f"✅ Not Stressed (Confidence: {confidence[0]*100:.2f}%)")
