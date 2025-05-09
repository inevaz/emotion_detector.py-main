Emotion Detector
A simple yet effective text-based emotion detection system built using Python and the Natural Language Toolkit (NLTK). This project identifies five primary emotions anger, disgust, fear, joy, and sadness, from a given text input and returns their intensity scores along with the dominant emotion detected.
This script uses VADER (Valence Aware Dictionary and sEntiment Reasoner) , a pre-trained sentiment analysis model optimized for social media and short texts. While VADER gives a general sentiment score (positive, neutral, negative), this project extends it by mapping those results to more specific emotional states.

The logic applies weights and conditional boosts based on keywords to simulate basic emotion detection in a rule-based way. It also handles edge cases like empty inputs gracefully.

- Features
Detects 5 core emotions: anger, disgust, fear, joy, and sadness.
Uses VADER sentiment scores as a base for emotion inference.
Applies keyword-based boosting to improve accuracy.
Handles blank or invalid input with appropriate feedback.
Returns the dominant emotion detected.

- Technologies Used
Python
NLTK (Natural Language Toolkit)
SentimentIntensityAnalyzer (from NLTK's VADER)

- This project was developed as part of the IBM FullStack Developer Professional Certificate program on Coursera.
