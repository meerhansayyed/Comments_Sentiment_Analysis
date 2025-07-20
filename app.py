import streamlit as st
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

from nltk.sentiment.sentiment_analyzer import SentimentAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download the VADER lexicon if not already present
nltk.download('vader_lexicon')

# Initialize the built-in VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Streamlit UI
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")
st.title("🔍 Sentiment Analyzer with VADER (NLTK)")

# Input from user
text = st.text_area("Enter a sentence or paragraph to analyze sentiment:", height=150)

if st.button("Analyze"):
    if text.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        scores = analyzer.polarity_scores(text)
        st.subheader("🔎 Sentiment Scores")
        st.json(scores)

        st.subheader("📊 Analysis Result")
        compound = scores['compound']
        if compound >= 0.05:
            st.success("Positive Sentiment 😊")
        elif compound <= -0.05:
            st.error("Negative Sentiment 😞")
        else:
            st.info("Neutral Sentiment 😐")
