import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

# Load word index and model
word_index = imdb.get_word_index()
model = load_model('simple_rnn_imdb.h5')

def preprocess_text(text):
    words = text.lower().split()
    encoded = [word_index.get(w, 2) + 3 for w in words]
    padded = sequence.pad_sequences([encoded], maxlen=500)
    return padded

def predict_sentiment(review):
    processed = preprocess_text(review)
    prob = model.predict(processed)[0][0]
    sentiment = "Positive" if prob >= 0.5 else "Negative"
    return sentiment, prob

# ---- Streamlit App ----
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a review to classify its sentiment:")

user_input = st.text_area("Movie Review")

if st.button("Classify"):
    sentiment, score = predict_sentiment(user_input)
    st.write(f"**Sentiment:** {sentiment}")
    st.write(f"**Score:** {score:.4f}")
