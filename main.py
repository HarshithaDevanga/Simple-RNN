# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation
model = load_model('simple_rnn_imdb.h5')

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


import streamlit as st
## streamlit app
# Streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# User input
user_input = st.text_area('Movie Review')

# Replace the classification button section with:
if st.button('Classify'):
    sentiment, score = predict_sentiment(user_input)
    
    # Display with color coding
    if sentiment == 'Positive':
        st.success(f'Sentiment: {sentiment}')
    else:
        st.error(f'Sentiment: {sentiment}')
    
    # Visualize score on -1 to 1 scale
    st.write(f'Prediction Score: {score:.4f}')
    st.progress((score + 1) / 2)  # Convert -1:1 to 0:1 for progress bar
