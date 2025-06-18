import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

# Load IMDB word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load model
model = load_model('simple_rnn_imdb.h5')

# ----------------------------
# Define helper functions here
# ----------------------------

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    prob = prediction[0][0]
    sentiment = 'Positive' if prob >= 0.5 else 'Negative'
    return sentiment, prob

# ----------------------------
# Streamlit app starts here
# ----------------------------

st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify it as positive or negative.")

user_input = st.text_area("Movie Review")

if st.button("Classify"):
    sentiment, score = predict_sentiment(user_input)
    
    if sentiment == "Positive":
        st.success(f"Sentiment: {sentiment}")
    else:
        st.error(f"Sentiment: {sentiment}")
        
    st.write(f"Prediction Score: {score}")
